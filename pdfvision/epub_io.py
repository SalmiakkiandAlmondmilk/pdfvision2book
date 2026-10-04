"""EPUB 输入: 仅翻译文字, 图片与行内结构原位保留。

做法:
  1. 把 EPUB 当作 zip 读入, 找出需翻译的 XHTML/HTML 文档(全书所有 .xhtml/.html);
  2. 对每个文档做标签级分词, 只在"文字片段"上调用纯文本模型翻译,
     标签、属性、图片引用、行内结构全部原样保留 —— 因此图片自然留在原位置;
  3. 其余资源(图片、字体、CSS、OPF)原样拷贝到新 EPUB, 只替换文字被翻译的文档。
"""
from __future__ import annotations

import html
import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

TOKEN_RE = re.compile(r"(<!--.*?-->|<!\[CDATA\[.*?\]\]>|<\?.*?\?>|<[^>]*>)", re.S)
TAG_NAME_RE = re.compile(r"</?\s*([A-Za-z0-9:_-]+)")
DOC_EXTS = (".xhtml", ".html", ".htm", ".xml")

SKIP_TAGS = {"script", "style", "pre", "code", "svg", "math"}
VOID_TAGS = {"img", "br", "hr", "meta", "link", "input", "source", "area",
             "base", "col", "embed", "param", "track", "wbr"}
BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th",
              "tr", "blockquote", "section", "article", "figcaption", "figure",
              "dd", "dt", "aside", "header", "footer", "nav", "title", "caption",
              "body", "html"}

TranslateFn = Callable[[List[Tuple[str, str]]], List[str]]


class EpubError(RuntimeError):
    pass


@dataclass
class EpubBook:
    path: Path
    entries: List[Tuple[str, bytes]] = field(default_factory=list)   # (name, data)
    doc_names: List[str] = field(default_factory=list)               # 需翻译的文档
    opf_name: str = ""

    @property
    def entry_map(self) -> Dict[str, bytes]:
        return {n: d for n, d in self.entries}

    def text_docs(self) -> List[Tuple[str, str]]:
        """[(name, text)] 文档名 -> 解码后的文本。"""
        out = []
        emap = self.entry_map
        for name in self.doc_names:
            data = emap.get(name)
            if data is None:
                continue
            out.append((name, data.decode("utf-8", "replace")))
        return out


def read_epub(path) -> EpubBook:
    p = Path(path)
    try:
        with zipfile.ZipFile(p) as zf:
            entries = [(i.filename, zf.read(i.filename)) for i in zf.infolist()
                       if not i.is_dir()]
    except zipfile.BadZipFile as exc:
        raise EpubError(f"EPUB 不是合法的 zip 包: {exc}") from exc
    names = [n for n, _ in entries]
    doc_names = [n for n in names if n.lower().endswith(DOC_EXTS)
                 and not n.startswith("META-INF/")
                 and not n.lower().endswith((".opf", ".ncx"))]
    if not doc_names:
        raise EpubError("EPUB 里没有找到可翻译的 XHTML 文档。")
    # 按 OPF spine 顺序排一下(找不到就用 zip 顺序)
    opf_name = _find_opf(names, {n: d for n, d in entries})
    ordered = _spine_order(entries, opf_name)
    if ordered:
        doc_names = [n for n in ordered if n in doc_names] + \
                    [n for n in doc_names if n not in ordered]
    return EpubBook(path=p, entries=entries, doc_names=doc_names, opf_name=opf_name)


def _find_opf(names: List[str], emap: Dict[str, bytes]) -> str:
    container = emap.get("META-INF/container.xml", b"")
    m = re.search(rb'full-path="([^"]+)"', container)
    if m:
        return m.group(1).decode("utf-8", "replace")
    for n in names:
        if n.lower().endswith(".opf"):
            return n
    return ""


def _spine_order(entries, opf_name: str) -> List[str]:
    if not opf_name:
        return []
    emap = {n: d for n, d in entries}
    try:
        opf = emap[opf_name].decode("utf-8", "replace")
    except KeyError:
        return []
    base = posixpath.dirname(opf_name)
    id_href: Dict[str, str] = {}
    for m in re.finditer(r"<item\b[^>]*>", opf):
        tag = m.group(0)
        idm = re.search(r'id="([^"]+)"', tag)
        hm = re.search(r'href="([^"]+)"', tag)
        if idm and hm:
            id_href[idm.group(1)] = _norm_href(base, hm.group(1))
    order: List[str] = []
    for m in re.finditer(r"<itemref\b[^>]*>", opf):
        idm = re.search(r'idref="([^"]+)"', m.group(0))
        if idm and idm.group(1) in id_href:
            order.append(id_href[idm.group(1)])
    return order


def _norm_href(base: str, href: str) -> str:
    href = href.split("#", 1)[0]
    joined = posixpath.normpath(posixpath.join(base, href)) if base else href
    return joined.lstrip("./")


# --------------------------------------------------------------------------- #
# 标签级分词翻译
# --------------------------------------------------------------------------- #
# 合并时允许跨过的行内标签(不打断句子); img/br/hr 等结构性标签会打断合并
INLINE_MERGE_TAGS = {"a", "b", "i", "em", "strong", "span", "small", "sup", "sub",
                     "u", "s", "ruby", "rb", "rt", "rp", "font", "big", "mark",
                     "cite", "q", "abbr", "time", "wbr"}
STRUCT_TAGS = {"img", "br", "hr", "svg", "image", "table", "tr", "td", "th",
               "figure", "video", "audio", "object", "iframe"}
RUBY_RE = re.compile(r"<(rt|rp)\b[^>]*>.*?</\1\s*>", re.S | re.I)


def strip_ruby(text: str) -> str:
    """去掉日语注音 <rt>/<rp> 的内容(翻译成中文后再显示假名没有意义)。"""
    return RUBY_RE.sub("", text)


def _scan(text: str):
    """扫描 XHTML, 返回 (slots, groups)。

    slots : [{"start","end","text","ctx","group","first"}]
    groups: [{"text","ctx"}]  —— 同一块级元素内被行内标签分隔的文字合并为一组
             (日语小说常见: 一句被 <ruby>/<span> 切碎, 逐片翻译既慢又差)
    """
    tokens = list(TOKEN_RE.finditer(text))
    slots: List[Dict] = []
    groups: List[Dict] = []
    cur: List[Dict] = []          # 当前组的 slots
    skip_depth = 0
    ctx_buf = ""

    def close_group():
        nonlocal cur
        if cur:
            merged = "".join(s["text"] for s in cur).strip()
            if merged:
                groups.append({"text": merged, "ctx": cur[0]["ctx"], "slots": cur})
                for j, s in enumerate(cur):
                    s["group"] = len(groups) - 1
                    s["first"] = (j == 0)
            cur = []

    pos = 0
    for m in tokens:
        if m.start() > pos:
            chunk = text[pos:m.start()]
            if skip_depth == 0 and _has_letters(chunk):
                slots.append({"start": pos, "end": m.start(), "text": chunk,
                              "ctx": ctx_buf, "group": -1, "first": False})
                cur.append(slots[-1])
            ctx_buf = (ctx_buf + " " + chunk).strip()
            if len(ctx_buf) > 400:
                ctx_buf = ctx_buf[-400:]
        tag = m.group(0)
        nm = TAG_NAME_RE.match(tag)
        if nm:
            name = nm.group(1).lower()
            closing = tag.startswith("</")
            self_closing = tag.endswith("/>")
            if name in SKIP_TAGS:
                skip_depth = max(0, skip_depth - 1) if closing else (
                    skip_depth if self_closing else skip_depth + 1)
                close_group()          # 不跨过 script/style/pre/code 合并
            elif name in BLOCK_TAGS or name in STRUCT_TAGS:
                close_group()
            if name in BLOCK_TAGS and not closing:
                ctx_buf = ""
        pos = m.end()
    if pos < len(text):
        chunk = text[pos:]
        if skip_depth == 0 and _has_letters(chunk):
            slots.append({"start": pos, "end": len(text), "text": chunk,
                          "ctx": ctx_buf, "group": -1, "first": False})
            cur.append(slots[-1])
    close_group()
    return slots, groups


def collect_text_slots(text: str) -> List[Dict]:
    """提取需要翻译的文字片段(逐片, 不合并)。"""
    slots, _groups = _scan(text)
    return slots


def collect_translation_units(text: str) -> List[Dict]:
    """提取翻译单元(同一块级元素内合并), 用于预估请求数。"""
    slots, groups = _scan(text)
    return groups


def translate_xhtml(text: str, translate_fn: TranslateFn, target: str,
                    strip_ruby_text: bool = False) -> str:
    """翻译 XHTML 里的文字, 标签/属性/图片原样保留。

    同一 <p>/<div> 内被行内标签(含 <ruby> 注音)切碎的文字会**合并成一句**翻译,
    译文写入该组第一个文字节点, 其余节点内容清空 —— 既减少请求, 也避免把句子切碎。
    """
    if strip_ruby_text:
        text = strip_ruby(text)
    slots, groups = _scan(text)
    if not groups:
        return text
    translated = translate_fn([(g["text"], g["ctx"]) for g in groups])
    out, cursor = [], 0
    for slot in slots:
        out.append(text[cursor:slot["start"]])
        raw = slot["text"]
        lead = raw[:len(raw) - len(raw.lstrip())]
        trail = raw[len(raw.rstrip()):]
        gi = slot.get("group", -1)
        body = ""
        if gi >= 0 and slot.get("first"):
            trans = translated[gi] if gi < len(translated) else ""
            body = (trans or raw).strip()
            if not _has_letters(body):      # 译文异常 -> 保留原文
                body = raw.strip()
        elif gi < 0:                        # 未成组的片段: 保持原文
            body = raw.strip()
        out.append(lead + html.escape(body, quote=False) + trail)
        cursor = slot["end"]
    out.append(text[cursor:])
    return "".join(out)


def _has_letters(s: str) -> bool:
    return any(c.isalpha() for c in s)


# --------------------------------------------------------------------------- #
# 写回
# --------------------------------------------------------------------------- #
def write_epub_copy(book: EpubBook, dst_path, replacements: Dict[str, bytes],
                    language: str = "") -> None:
    """把原 EPUB 复制一份, 用 replacements 覆盖指定条目(文字翻译后的文档)。"""
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    lang_done = False
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
        first = True
        for name, data in book.entries:
            if name == "mimetype":
                zf.writestr(zipfile.ZipInfo("mimetype"), data,
                            compress_type=zipfile.ZIP_STORED)
                first = False
                continue
            payload = replacements.get(name, data)
            if language and book.opf_name and name == book.opf_name and not lang_done:
                payload = _set_language(payload, language)
                lang_done = True
            if first:      # 极端情况: 原包里没有 mimetype, 补在最前面
                zf.writestr(zipfile.ZipInfo("mimetype"), b"application/epub+zip",
                            compress_type=zipfile.ZIP_STORED)
                first = False
            zf.writestr(name, payload)
        if first:
            zf.writestr(zipfile.ZipInfo("mimetype"), b"application/epub+zip",
                        compress_type=zipfile.ZIP_STORED)


def _set_language(opf_bytes: bytes, language: str) -> bytes:
    text = opf_bytes.decode("utf-8", "replace")
    new = re.sub(r"(<dc:language[^>]*>)[^<]*(</dc:language>)",
                 rf"\g<1>{language}\g<2>", text, count=1)
    if new == text:
        return opf_bytes
    return new.encode("utf-8")


def plain_text_from_docs(docs: List[Tuple[str, str]]) -> str:
    """把(已翻译的)XHTML 文档转成纯文本。"""
    chunks: List[str] = []
    for _name, text in docs:
        stripped = re.sub(r"<(script|style)\b.*?</\1>", " ", text, flags=re.S | re.I)
        stripped = re.sub(r"<br\s*/?>", "\n", stripped, flags=re.I)
        stripped = re.sub(r"</(p|div|h[1-6]|li|tr|blockquote|section|article)>", "\n",
                          stripped, flags=re.I)
        stripped = re.sub(r"<[^>]+>", "", stripped)
        stripped = html.unescape(stripped)
        stripped = re.sub(r"[ \t\u00a0]+", " ", stripped)
        stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()
        if stripped:
            chunks.append(stripped)
    return "\n\n".join(chunks)
