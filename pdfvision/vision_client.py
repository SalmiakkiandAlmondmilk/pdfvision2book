"""DeepSeek 视觉模型客户端(OpenAI 兼容 Chat Completions)。

仅用标准库 urllib 实现, 支持 base64 内联图片; 对 429/5xx/网络抖动做指数退避重试。
负责: 逐页转写/翻译, 解析 [[FIG:n]] 占位符与 <<<FIGURES>>> 之后的插图 JSON。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .http_client import ChatHTTP, VisionError, http_detail  # noqa: F401 (re-export)
from .pdf_ingest import PageImage
from .prompts import (FIGURES_SEP, MARKER_ORIGINAL, MARKER_TRANSLATION,
                      build_vision_prompt)

log = logging.getLogger("pdfvision")

_FIG_RE = re.compile(r"\[\[\s*FIG\s*:?\s*(\d+)\s*\]\]", re.I)


@dataclass
class Figure:
    """页面中的一幅插图。"""
    n: int = 0
    pos: str = ""            # 九宫格位置(top-left … bottom-right)
    kind: str = "other"      # photo/chart/diagram/map/comic/other
    caption: str = ""        # 原文描述
    text: str = ""           # 图中文字(原文逐字转写)
    caption_t: str = ""      # 译文描述
    text_t: str = ""         # 图中文字译文
    rect: Optional[Tuple[float, float, float, float]] = None  # 几何区域(PDF 坐标)
    asset_href: str = ""     # 嵌入 EPUB 时的资源路径

    def body_text(self, translated: bool) -> str:
        """图中文字(优先译文)。"""
        if translated and self.text_t:
            return self.text_t
        return self.text or (self.caption_t if translated else self.caption)

    def caption_text(self, translated: bool) -> str:
        if translated and self.caption_t:
            return self.caption_t
        return self.caption

    def to_dict(self) -> dict:
        return {"n": self.n, "pos": self.pos, "kind": self.kind,
                "caption": self.caption, "text": self.text,
                "caption_t": self.caption_t, "text_t": self.text_t,
                "rect": list(self.rect) if self.rect else None,
                "asset_href": self.asset_href}

    @classmethod
    def from_dict(cls, d: dict) -> "Figure":
        rect = d.get("rect")
        return cls(n=int(d.get("n", 0)), pos=str(d.get("pos", "")),
                   kind=str(d.get("kind", "other")),
                   caption=str(d.get("caption", "")), text=str(d.get("text", "")),
                   caption_t=str(d.get("caption_t", "")),
                   text_t=str(d.get("text_t", "")),
                   rect=tuple(rect) if rect else None,
                   asset_href=str(d.get("asset_href", "")))


@dataclass
class PageResult:
    """一页(或一张图)的识别结果。

    blocks:  [(label, text)]        label 为 "原文"/"译文" 或兜底标签
    figures: 该页识别出的插图
    usage:   该请求消耗的 tokens
    """
    page_no: int
    label: str
    blocks: List[tuple]
    figures: List[Figure] = field(default_factory=list)
    usage: Dict[str, int] = None  # type: ignore[assignment]

    @property
    def original(self) -> Optional[str]:
        for lab, txt in self.blocks:
            if lab == MARKER_ORIGINAL:
                return txt
        return None

    @property
    def translation(self) -> Optional[str]:
        for lab, txt in self.blocks:
            if lab == MARKER_TRANSLATION:
                return txt
        return None

    @property
    def primary(self) -> str:
        if not self.blocks:
            return ""
        return self.blocks[0][1]

    def body(self, translate: bool, with_original: bool) -> str:
        """取该页最终要展示的正文(已含 [[FIG:n]] 占位符)。"""
        if not translate:
            return self.primary
        if with_original:
            parts = []
            if self.original:
                parts.append(self.original)
            if self.translation:
                parts.append(self.translation)
            return "\n\n".join(parts)
        return self.translation or self.primary

    def to_dict(self) -> dict:
        return {"page_no": self.page_no, "label": self.label,
                "blocks": [[lab, txt] for lab, txt in self.blocks],
                "figures": [f.to_dict() for f in self.figures],
                "usage": self.usage or {}}

    @classmethod
    def from_dict(cls, d: dict) -> "PageResult":
        return cls(page_no=int(d.get("page_no", 0)),
                   label=str(d.get("label", "")),
                   blocks=[(str(lab), str(txt)) for lab, txt in d.get("blocks", [])],
                   figures=[Figure.from_dict(f) for f in d.get("figures", [])],
                   usage=d.get("usage") or {})


class VisionSession(ChatHTTP):
    """视觉模型会话(含累计用量统计)。"""

    def transcribe_image(self, img: PageImage, translate: bool,
                         with_original: bool, target: str = "",
                         candidate_count: int = 0) -> PageResult:
        """把一张图片交给视觉模型, 返回转写/翻译 + 插图信息。"""
        text = build_vision_prompt(translate=translate, with_original=with_original,
                                   target=target, candidate_count=candidate_count)
        content = [{"type": "text", "text": text},
                   {"type": "image_url",
                    "image_url": {"url": img.data_url, "detail": "original"}}]
        raw = self.chat([{"role": "user", "content": content}], temperature=0.2)
        answer = self.content_of(raw)
        text_part, figures = split_answer(answer)
        return PageResult(page_no=img.page_no, label=img.label,
                          blocks=self._split_blocks(text_part, translate, with_original),
                          figures=figures, usage=raw.get("usage") or {})

    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_blocks(answer: str, translate: bool,
                      with_original: bool) -> List[tuple]:
        """按 【原文】/【译文】 标记切分; 没有标记时给出兜底。"""
        text = answer.replace("（原文）", f"【{MARKER_ORIGINAL}】") \
                     .replace("（译文）", f"【{MARKER_TRANSLATION}】")
        head = "【" + MARKER_ORIGINAL + "】"
        trans = "【" + MARKER_TRANSLATION + "】"
        found: List[tuple] = []
        if head in text:
            rest = text.split(head, 1)[1]
            if trans in rest:
                orig, tail = rest.split(trans, 1)
                found = [(MARKER_ORIGINAL, orig.strip()),
                         (MARKER_TRANSLATION, tail.strip())]
        elif trans in text:
            _, tail = text.split(trans, 1)
            found = [(MARKER_TRANSLATION, tail.strip())]
        if found:
            return [b for b in found if b[1]]
        if not translate:
            return [(MARKER_ORIGINAL, answer)]
        return [(MARKER_TRANSLATION, answer)]


# --------------------------------------------------------------------------- #
# 解析工具(视觉与文本客户端共用)
# --------------------------------------------------------------------------- #
def parse_json_loose(text: str) -> Optional[dict]:
    """尽量从模型输出里取出 JSON 对象(容忍 ```json 包裹、前后杂字符)。"""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I | re.M).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    start, end = t.find("{"), t.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(t[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    return None


def split_answer(answer: str) -> Tuple[str, List[Figure]]:
    """把模型回答拆成 (正文, 插图列表)。"""
    text_part, json_part = answer, ""
    if FIGURES_SEP in answer:
        text_part, json_part = answer.split(FIGURES_SEP, 1)
    figures: List[Figure] = []
    obj = parse_json_loose(json_part) if json_part.strip() else None
    if obj:
        raw_list = obj.get("figures")
        if isinstance(raw_list, list):
            for i, item in enumerate(raw_list, start=1):
                if not isinstance(item, dict):
                    continue
                try:
                    n = int(item.get("n", i) or i)
                except (TypeError, ValueError):
                    n = i
                figures.append(Figure(
                    n=n, pos=str(item.get("pos", "") or ""),
                    kind=str(item.get("type", item.get("kind", "other")) or "other"),
                    caption=str(item.get("caption", "") or ""),
                    text=str(item.get("text", "") or ""),
                    caption_t=str(item.get("caption_t", "") or ""),
                    text_t=str(item.get("text_t", "") or "")))
    elif "<<<FIGURE" in answer or _FIG_RE.search(answer):
        # JSON 缺失但有占位符: 保留占位符信息, 仅内联文字说明
        for m in _FIG_RE.finditer(answer):
            n = int(m.group(1))
            if all(f.n != n for f in figures):
                figures.append(Figure(n=n))
    figures.sort(key=lambda f: f.n)
    return text_part.strip(), figures
