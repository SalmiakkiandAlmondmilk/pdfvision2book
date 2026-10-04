"""主流程。

  * PDF / 图片  -> 视觉模型(deepseek-v4-flash-vision-exp): 逐页转写(+可选翻译),
                   识别插图并裁切, 在 EPUB 中原位嵌入;
  * EPUB / TXT  -> 纯文本模型(deepseek-v4-flash): 仅翻译;
                   EPUB 的图片、行内结构原位保留。

输出 TXT / EPUB3, 正文里不出现任何页/图编号标记。
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .bookbuilder import write_epub, write_text_file, write_txt
from .config import Settings
from .epub_io import (collect_text_slots, collect_translation_units, plain_text_from_docs,
                      read_epub, translate_xhtml, write_epub_copy)
from .pdf_ingest import PdfError, crop_rect, figure_candidates, zone_of
from .prompts import MARKER_ORIGINAL, PROMPT_VERSION
from .sources import (KIND_DESC, KIND_EPUB, KIND_IMAGE, KIND_PDF, KIND_TXT,
                      detect_kind, load_visual_images, read_txt_text)
from .text_client import (CACHE_RECORD_VERSION, TextSession, cache_value_usable,
                          make_batches, split_long)
from .vision_client import PageResult, VisionError, VisionSession

log = logging.getLogger("pdfvision")

MODE_DESC = {"page": "整页渲染后视觉阅读", "images": "提取嵌入图片后视觉阅读"}


# --------------------------------------------------------------------------- #
# 输出目录 / 断点
# --------------------------------------------------------------------------- #
def _job_key(settings: Settings, target: str, kind: str) -> str:
    """同一批"任务设定"下的唯一键; 设定变化(翻译/模式/dpi/模型/插图开关)则不复用旧断点。"""
    raw = "|".join([PROMPT_VERSION, kind, settings.mode, str(settings.dpi),
                    str(target), str(settings.with_original), str(settings.keep_figures),
                    settings.vision_model, settings.text_model])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def resolve_output_dir(settings: Settings, src: Path):
    """按优先级寻找可写的输出目录并预检可写性, 返回 (目录, 提示文本)。"""
    stem = src.stem

    def probe(p: Path):
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe_file = p / ".write_probe"
            probe_file.write_text("ok", encoding="utf-8")
            probe_file.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    candidates = []
    if settings.outdir:
        candidates.append(Path(settings.outdir).resolve())
    else:
        candidates.append((src.parent / (stem + "_vision")).resolve())
    candidates.append((Path.cwd() / (stem + "_vision")).resolve())
    candidates.append((Path(tempfile.gettempdir()) / (stem + "_vision")).resolve())

    seen = set()
    chosen = None
    for c in candidates:
        key = str(c).lower()
        if key in seen:
            continue
        seen.add(key)
        if probe(c):
            chosen = c
            break
    if chosen is None:
        raise PdfError("找不到任何可写的输出目录, 请用 --outdir 指定一个可写路径。")

    note = ""
    default_hint = (src.parent / (stem + "_vision")).resolve()
    if chosen != default_hint and not settings.outdir:
        note = f"默认输出目录不可写({default_hint}), 输出已改存到: {chosen}"
    elif settings.outdir and Path(settings.outdir).resolve() != chosen:
        note = f"指定的输出目录不可写, 输出已改存到: {chosen}"
    return chosen, note


class _JsonlStore:
    """极简 JSONL 追加式存储(断点续跑用), 目录被删时自动重建。"""

    def __init__(self, path: Path, lock: threading.Lock):
        self.path = path
        self.lock = lock

    def read_lines(self) -> List[dict]:
        if not self.path.exists():
            return []
        out: List[dict] = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue        # 半行损坏
                if isinstance(obj, dict):
                    out.append(obj)
        except OSError:
            return []
        return out

    def append(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with self.lock:
            try:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line)
            except OSError:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a", encoding="utf-8") as f:
                        f.write(line)
                except OSError:
                    log.warning("checkpoint 写入失败(忽略): %s", self.path)


# --------------------------------------------------------------------------- #
# 视觉通道: PDF / 图片
# --------------------------------------------------------------------------- #
def process_visual(settings: Settings, kind: str, progress=None) -> Dict:
    t0 = time.time()
    src = Path(settings.pdf_path)
    outdir, outdir_note = resolve_output_dir(settings, src)
    pages_dir = None
    if settings.save_pages:
        pages_dir = outdir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

    def report(done, total, msg):
        if progress:
            progress(done, total, msg)
        else:
            log.info(msg)
            if total == 0 or total <= 10 or done == total:
                print(f"  进度 {done}/{total} {msg}", flush=True)

    report(0, 0, "正在解析输入文件并生成图片 …")
    if outdir_note:
        report(0, 0, outdir_note)
    images = load_visual_images(settings, save_dir=pages_dir)
    if not images:
        raise PdfError("没有提取到任何页面/图片, 无法处理。")
    total = len(images)
    report(0, total, f"共 {total} 张图片待处理")

    # 插图候选区域(仅 PDF 整页模式)
    cand_map: Dict[int, List[Tuple[tuple, str]]] = {}
    if settings.keep_figures and kind == KIND_PDF and settings.mode == "page":
        cand_map = _collect_figure_candidates(src, [im.page_no for im in images])
        got = sum(len(v) for v in cand_map.values())
        if got:
            report(0, total, f"版面分析: 检测到 {got} 个候选插图区域")

    translate = bool(settings.translate)
    target = settings.translate or ""
    session = VisionSession(api_key=settings.api_key, base_url=settings.base_url,
                            model=settings.vision_model, timeout=settings.timeout,
                            max_attempts=settings.max_attempts,
                            max_output_tokens_per_request=settings.max_output_tokens_per_request,
                            thinking=settings.vision_thinking)
    results: List[Optional[PageResult]] = [None] * total
    job_key = _job_key(settings, target, kind)
    store = _JsonlStore(outdir / ".checkpoint.jsonl", threading.Lock())

    cache: Dict[str, PageResult] = {}
    cached_usage: Dict[str, int] = {}
    if settings.resume:
        seen = set()
        for rec in store.read_lines():
            k = str(rec.get("k", ""))
            if not k.startswith(job_key + ":") or k in seen:
                continue
            seen.add(k)
            try:
                cache[k] = PageResult.from_dict(rec.get("result") or {})
            except (TypeError, ValueError):
                continue
            for kk, vv in (rec.get("usage") or {}).items():
                if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                    cached_usage[kk] = cached_usage.get(kk, 0) + int(vv)

    pending: List[Tuple[int, object]] = []
    resume_count = 0
    for i, img in enumerate(images):
        hit = cache.get(job_key + ":" + img.digest)
        if hit is not None:
            results[i] = hit
            resume_count += 1
        else:
            pending.append((i, img))
    if resume_count:
        for k, v in cached_usage.items():
            session.total_usage[k] = session.total_usage.get(k, 0) + v
        report(resume_count, total,
               f"断点续跑: 复用已完成 {resume_count} 张, 仍需处理 {len(pending)} 张")

    ck_lock = threading.Lock()

    def run_one(item):
        idx, img = item
        try:
            result = session.transcribe_image(
                img, translate=translate, with_original=settings.with_original,
                target=target, candidate_count=len(cand_map.get(img.page_no, [])))
            with ck_lock:
                store.append({"k": job_key + ":" + img.digest,
                              "result": result.to_dict(),
                              "usage": result.usage or {}})
            return idx, result
        except VisionError as exc:
            return idx, exc

    done = resume_count
    with ThreadPoolExecutor(max_workers=max(1, settings.workers)) as pool:
        futures = [pool.submit(run_one, it) for it in pending]
        for fut in as_completed(futures):
            idx, outcome = fut.result()
            done += 1
            if isinstance(outcome, Exception):
                report(done, total, f"第 {idx + 1} 张失败: {outcome}")
                raise outcome
            results[idx] = outcome
            report(done, total, f"完成第 {idx + 1} 张")

    pages: List[PageResult] = [r for r in results if r is not None]
    if not pages:
        raise VisionError("没有任何页面被成功识别。")

    # 插图: 把模型的插图描述对应到几何区域, 裁切出来准备嵌入
    assets: List[dict] = []
    if settings.keep_figures and cand_map:
        assets = _crop_figures(src, pages, cand_map, settings, report)

    meta = {
        "source": src.name,
        "mode_desc": KIND_DESC.get(kind, kind) + " · " + MODE_DESC.get(settings.mode, ""),
        "translate": target,
        "model": settings.vision_model,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "outdir_note": outdir_note or "",
    }
    title = settings.document_title
    report(0, 0, "正在生成输出文件 …")
    summary = _write_outputs(settings, pages, translate, title, meta, outdir,
                             assets, session.total_usage, report)
    summary.update({"images": total, "pages_done": len(pages),
                    "requests": session.requests,
                    "output_tokens": session.total_usage.get("completion_tokens", 0),
                    "seconds": round(time.time() - t0, 1), "kind": kind})
    report(total, total, "全部完成")
    return summary


def _collect_figure_candidates(src: Path, page_nos: List[int]):
    """一次性打开 PDF, 收集这些页的插图候选区域。"""
    out: Dict[int, List[Tuple[tuple, str]]] = {}
    try:
        import pymupdf as fitz
    except ImportError:  # pragma: no cover
        import fitz  # type: ignore
    try:
        doc = fitz.open(src)
    except Exception:  # noqa: BLE001
        return out
    try:
        for pno in sorted(set(page_nos)):
            if pno - 1 >= doc.page_count:
                continue
            page = doc.load_page(pno - 1)
            rects = figure_candidates(page)
            if rects:
                out[pno] = [(tuple(r), zone_of(r, page.rect)) for r in rects]
    finally:
        doc.close()
    return out


def _crop_figures(src: Path, pages: List[PageResult], cand_map, settings: Settings,
                  report) -> List[dict]:
    """把模型识别出的插图与几何候选区域配对并裁切成 PNG 资源。"""
    try:
        import pymupdf as fitz
    except ImportError:  # pragma: no cover
        import fitz  # type: ignore
    assets: List[dict] = []
    try:
        doc = fitz.open(src)
    except Exception:  # noqa: BLE001
        return assets
    try:
        for p in pages:
            cands = cand_map.get(p.page_no, [])
            if not cands or not p.figures:
                continue
            page = doc.load_page(min(p.page_no - 1, doc.page_count - 1))
            used: set = set()
            ordered = list(cands)
            # 1) 先按九宫格位置配对
            for fig in p.figures:
                if fig.rect is not None or not fig.pos:
                    continue
                for i, (rect, zone) in enumerate(ordered):
                    if i in used or zone != fig.pos:
                        continue
                    fig.rect = tuple(rect)
                    used.add(i)
                    break
            # 2) 剩余的按顺序配对
            for fig in p.figures:
                if fig.rect is not None:
                    continue
                for i, (rect, _zone) in enumerate(ordered):
                    if i in used:
                        continue
                    fig.rect = tuple(rect)
                    used.add(i)
                    break
            # 3) 裁切
            for fig in p.figures:
                if not fig.rect:
                    continue
                try:
                    data, _w, _h = crop_rect(page, fig.rect, dpi=settings.figure_dpi,
                                             max_side=settings.figure_max_side)
                except Exception as exc:  # noqa: BLE001
                    log.warning("插图裁切失败(第 %s 张): %s", p.page_no, exc)
                    continue
                href = f"figures/fig_{p.page_no:04d}_{fig.n}.png"
                fig.asset_href = href
                assets.append({"id": f"fig{p.page_no}_{fig.n}", "href": href,
                               "media_type": "image/png", "data": data})
            if any(f.asset_href for f in p.figures):
                report(0, 0, f"已嵌入插图: 第 {p.page_no} 张共 "
                             f"{sum(1 for f in p.figures if f.asset_href)} 幅")
    finally:
        doc.close()
    return assets


def _write_outputs(settings: Settings, pages: List[PageResult], translate: bool,
                   title: str, meta: Dict, outdir: Path, assets: List[dict],
                   usage: Dict[str, int], report) -> Dict:
    """写 TXT / EPUB(目录被删时自动重建; 仍失败则改存系统临时目录)。"""

    def write_all(dir_path: Path, meta_dict: Dict) -> bool:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
        try:
            if "txt" in settings.formats:
                write_txt(pages, translate, settings.with_original, title,
                          meta_dict, dir_path / f"{title}.txt")
            if "epub" in settings.formats:
                write_epub(pages, translate, settings.with_original, title,
                           meta_dict, dir_path / f"{title}.epub",
                           per_chapter=settings.pages_per_chapter,
                           lang=settings.target_language_tag, assets=assets)
            _write_meta(dir_path, settings, usage)
            return True
        except OSError:
            return False

    final_outdir = outdir
    if not write_all(final_outdir, meta):
        fallback = Path(tempfile.mkdtemp(prefix="pdfvision_"))
        note = f"输出目录写入失败({final_outdir}), 结果已改存到: {fallback}"
        report(0, 0, note)
        meta["outdir_note"] = (meta.get("outdir_note") + " | " if meta.get("outdir_note")
                               else "") + note
        if not write_all(fallback, meta):
            raise PdfError("所有输出目录均写入失败(含系统临时目录)。")
        final_outdir = fallback

    summary = {"title": title, "outdir": str(final_outdir),
               "meta": str(final_outdir / "meta.json"),
               "usage": _usage_dict(usage)}
    if "txt" in settings.formats:
        summary["txt"] = str(final_outdir / f"{title}.txt")
    if "epub" in settings.formats:
        summary["epub"] = str(final_outdir / f"{title}.epub")
    return summary


def _write_meta(outdir: Path, settings: Settings, usage: Dict[str, int],
                extra: Optional[Dict] = None) -> None:
    payload = {"settings": {**vars(settings), "api_key": "***"},
               "usage": dict(usage)}
    if extra:
        payload.update(extra)
    (outdir / "meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _usage_dict(usage: Dict[str, int]) -> Dict[str, int]:
    return {"prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("prompt_tokens", 0))
                            + int(usage.get("completion_tokens", 0))}


# --------------------------------------------------------------------------- #
# 文本通道: EPUB / TXT(仅翻译)
# --------------------------------------------------------------------------- #
def process_text(settings: Settings, kind: str, progress=None) -> Dict:
    t0 = time.time()
    src = Path(settings.pdf_path)
    outdir, outdir_note = resolve_output_dir(settings, src)

    def report(done, total, msg):
        if progress:
            progress(done, total, msg)
        else:
            log.info(msg)
            if total == 0 or total <= 10 or done == total:
                print(f"  进度 {done}/{total} {msg}", flush=True)

    target = (settings.translate or settings.default_text_target or "简体中文").strip()
    if not settings.translate:
        report(0, 0, f"文本类输入仅做翻译, 未指定 --translate, 使用默认目标语言: {target}")
    if outdir_note:
        report(0, 0, outdir_note)

    job_key = _job_key(settings, target, kind)
    store = _JsonlStore(outdir / ".checkpoint_text.jsonl", threading.Lock())
    cache: Dict[str, str] = {}
    rejected_cache = 0
    if settings.resume:
        seen_keys = set()
        for rec in store.read_lines():
            if rec.get("v") != CACHE_RECORD_VERSION:
                # 旧格式(无法判断内容是否可信)一律忽略; 键里也带了版本号, 双重保险
                continue
            k, t = rec.get("k"), rec.get("t")
            if not isinstance(k, str) or not isinstance(t, str) or k in seen_keys:
                continue
            seen_keys.add(k)
            if cache_value_usable(t, target):
                cache[k] = t
            else:
                rejected_cache += 1
        if cache or rejected_cache:
            report(0, 0, f"译文缓存: 可用 {len(cache)} 条"
                         + (f", 忽略不可信 {rejected_cache} 条(原文/拒答)" if rejected_cache else ""))
    else:
        report(0, 0, "已关闭断点续跑: 忽略已有译文缓存, 全部重新翻译")

    doc_idx = [1]      # 当前文档序号(供批次级进度显示)
    doc_total = [1]
    progress_lock = threading.Lock()

    def _batch_progress(d: int, t: int, note: str) -> None:
        with progress_lock:
            doc_idx_snapshot = doc_idx[0]
            doc_total_snapshot = doc_total[0]
            eta = ""
            if session.requests and d:
                per = (time.time() - t0) / max(session.requests, 1)
                remain = (t - d) * per
                if remain > 0:
                    eta = f" · 预计还需 {remain / 60:.1f} 分钟"
            report(max(d - 1, 0), max(t, 1),
                   f"文档 {doc_idx_snapshot}/{doc_total_snapshot} · 批次 {d}/{t} · "
                   f"{note}{eta}")

    session = TextSession(api_key=settings.api_key, base_url=settings.base_url,
                          model=settings.text_model, timeout=settings.timeout,
                          max_attempts=min(settings.max_attempts, 3), cache=cache,
                          on_cache=(lambda k, v: store.append(
                              {"k": k, "t": v, "v": CACHE_RECORD_VERSION}))
                          if settings.resume else None,
                          max_requests=settings.max_requests,
                          max_output_tokens=settings.max_output_tokens,
                          stall_limit=settings.stall_limit,
                          transport_fail_limit=settings.transport_fail_limit,
                          max_output_tokens_per_request=settings.max_output_tokens_per_request,
                          thinking=settings.text_thinking,
                          workers=settings.workers,
                          on_batch=_batch_progress)
    aborted = ""

    title = settings.document_title
    body = ""
    units = 1
    translated: List[str] = []
    if kind == KIND_EPUB:
        report(0, 0, "正在读取 EPUB(文字翻译, 图片原位保留)…")
        book = read_epub(src)
        docs = book.text_docs()
        units = len(docs)
        doc_total[0] = len(docs)
        replacements: Dict[str, bytes] = {}
        translated_docs: List[Tuple[str, str]] = []
        for i, (name, text) in enumerate(docs, start=1):
            doc_idx[0] = i
            try:
                new_text = translate_xhtml(
                    text, lambda segs: session.translate_segments(segs, target), target,
                    strip_ruby_text=settings.strip_ruby)
            except VisionError as exc:
                aborted = session.abort_reason or str(exc)
                report(i - 1, len(docs), f"已中止: {aborted}")
                break
            replacements[name] = new_text.encode("utf-8")
            translated_docs.append((name, new_text))
            report(i, len(docs), f"已翻译文档 {i}/{len(docs)}")
        meta = {"source": src.name,
                "mode_desc": KIND_DESC[kind],
                "translate": target, "model": settings.text_model,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "outdir_note": outdir_note or ""}
    else:
        report(0, 0, "正在读取 TXT 并翻译 …")
        raw = read_txt_text(src)
        blocks = [b for b in _split_blocks(raw) if b.strip()]
        if not blocks:
            raise PdfError("TXT 文件里没有可翻译的文字。")
        units = len(blocks)
        doc_total[0] = 1
        doc_idx[0] = 1
        contexts = [b[:400] for b in blocks]
        try:
            translated = session.translate_segments(list(zip(blocks, contexts)), target)
        except VisionError as exc:
            aborted = session.abort_reason or str(exc)
            report(0, 1, f"已中止: {aborted}")
            translated = list(blocks)          # 中止时保留原文, 仍输出已完成部分
        meta = {"source": src.name,
                "mode_desc": KIND_DESC[kind],
                "translate": target, "model": settings.text_model,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "outdir_note": outdir_note or ""}
        body = "\n\n".join(translated)
        translated_docs = []
        if not aborted:
            report(len(blocks), len(blocks), f"已翻译 {len(blocks)} 段")

    # 写输出
    def write_all(dir_path: Path, meta_dict: Dict) -> bool:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
        try:
            if kind == KIND_EPUB:
                if "epub" in settings.formats:
                    write_epub_copy(book, dir_path / f"{title}.epub", replacements,
                                    language=_lang_tag(settings, target))
                if "txt" in settings.formats:
                    write_text_file(title, meta_dict,
                                    plain_text_from_docs(translated_docs),
                                    dir_path / f"{title}.txt")
            else:
                if "txt" in settings.formats:
                    write_text_file(title, meta_dict, body, dir_path / f"{title}.txt")
                if "epub" in settings.formats:
                    write_epub(_blocks_to_pages(translated),
                               translate=False, with_original=False, title=title,
                               meta=meta_dict, path=dir_path / f"{title}.epub",
                               per_chapter=settings.pages_per_chapter,
                               lang=_lang_tag(settings, target))
            _write_meta(dir_path, settings, session.total_usage)
            return True
        except OSError:
            return False

    final_outdir = outdir
    if not write_all(final_outdir, meta):
        fallback = Path(tempfile.mkdtemp(prefix="pdfvision_"))
        note = f"输出目录写入失败({final_outdir}), 结果已改存到: {fallback}"
        report(0, 0, note)
        meta["outdir_note"] = (meta.get("outdir_note") + " | " if meta.get("outdir_note")
                               else "") + note
        if not write_all(fallback, meta):
            raise PdfError("所有输出目录均写入失败(含系统临时目录)。")
        final_outdir = fallback

    summary = {"title": title, "outdir": str(final_outdir), "kind": kind,
               "meta": str(final_outdir / "meta.json"),
               "pages_done": units, "images": units,
               "usage": _usage_dict(session.total_usage),
               "requests": session.requests,
               "output_tokens": session.output_tokens,
               "translated_units": session.translated_units,
               "cached_units": session.cached_units,
               "skipped_units": session.skipped_units,
               "total_pieces": session.total_pieces,
               "rejected_cache": rejected_cache + session.rejected_cache,
               "seconds": round(time.time() - t0, 1)}
    warning = ""
    failed_path = ""
    if session.failures:
        lines = ["# 未能翻译的片段(已保留原文)", ""]
        for i, f in enumerate(session.failures, start=1):
            lines += [f"[{i}] 原因: {f['why']}",
                      f"    原文: {f['source']}",
                      f"    模型答复: {f['answer'] or '(空)'}", ""]
        try:
            failed_path = str(final_outdir / "failed_units.txt")
            Path(failed_path).write_text("\n".join(lines), encoding="utf-8")
        except OSError:
            failed_path = ""
    if session.skipped_units:
        pct = session.skipped_units / max(session.total_pieces, 1) * 100
        warning = (f"有 {session.skipped_units}/{session.total_pieces} 段未能翻译({pct:.1f}%), "
                   f"这些位置仍是原文。原因与模型原始答复见 "
                   f"{Path(failed_path).name if failed_path else 'stdout 日志'}。"
                   f"建议: 直接重跑同一条命令(已完成的会命中缓存, 只补未译部分)。")
        report(0, 0, f"⚠ {warning}")
    summary["warning"] = warning
    summary["failed_units"] = failed_path
    _write_meta(final_outdir, settings, session.total_usage, extra={
        "requests": session.requests, "output_tokens": session.output_tokens,
        "translated_units": session.translated_units,
        "cached_units": session.cached_units,
        "skipped_units": session.skipped_units,
        "total_pieces": session.total_pieces,
        "rejected_cache": rejected_cache + session.rejected_cache,
        "failed_units_file": failed_path,
        "warning": warning, "aborted": aborted})
    if "txt" in settings.formats:
        summary["txt"] = str(final_outdir / f"{title}.txt")
    if "epub" in settings.formats:
        summary["epub"] = str(final_outdir / f"{title}.epub")
    if aborted:
        # 预算护栏触发: 已把完成的译文写出, 再以错误形式告知用户
        report(0, 0, f"⚠ {aborted}")
        raise VisionError(f"{aborted} 已完成部分已保存到: {final_outdir}")
    report(0, 0, "全部完成")
    return summary


def _lang_tag(settings: Settings, target: str) -> str:
    old = settings.translate
    settings.translate = target
    try:
        return settings.target_language_tag
    finally:
        settings.translate = old


def _split_blocks(text: str) -> List[str]:
    """按空行切段(保留段落内部换行)。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return [b.strip() for b in text.split("\n\n")]


def _blocks_to_pages(blocks: List[str]) -> List[PageResult]:
    """把已翻译的段落包装成 PageResult(仅供 write_epub 复用)。"""
    pages: List[PageResult] = []
    for i, b in enumerate(blocks, start=1):
        pages.append(PageResult(page_no=i, label="",
                                blocks=[(MARKER_ORIGINAL, b)], figures=[]))
    return pages


# --------------------------------------------------------------------------- #
# 预估(不调用 API)
# --------------------------------------------------------------------------- #
def plan_input(settings: Settings) -> Dict:
    """预估本次处理量: 单元数 / 请求数 / 单次输出上限 / 最大输出 token。

    不发送任何 API 请求, 用于花钱之前先看清楚规模。
    """
    src = Path(settings.pdf_path or "")
    if not src.exists():
        raise PdfError(f"找不到输入文件: {src}")
    kind = detect_kind(src)
    plan: Dict = {"kind": kind, "source": src.name,
                  "max_output_tokens_per_request": settings.max_output_tokens_per_request}

    if kind in (KIND_PDF, KIND_IMAGE):
        images = load_visual_images(settings)
        plan.update({
            "units": len(images), "unit_name": "页面/图片",
            "segments": len(images), "pieces": len(images),
            "requests_est": len(images), "requests_worst": len(images),
            "output_tokens_max": len(images) * settings.max_output_tokens_per_request,
            "note": f"视觉通道({settings.vision_model}): 每张图 1 次请求"
                    + ("(含插图识别)" if settings.keep_figures else ""),
        })
        return plan

    target = (settings.translate or settings.default_text_target or "简体中文").strip()
    segments: List[Tuple[str, str]] = []
    units = 0
    if kind == KIND_EPUB:
        book = read_epub(src)
        docs = book.text_docs()
        units = len(docs)
        for _name, text in docs:
            for group in collect_translation_units(text):
                segments.append((group["text"], group["ctx"]))
    else:
        blocks = [b for b in _split_blocks(read_txt_text(src)) if b.strip()]
        units = 1
        segments = [(b, b[:400]) for b in blocks]

    pieces = 0
    piece_items: List[Tuple[int, int, str, str]] = []
    for i, (text, ctx) in enumerate(segments):
        for k, piece in enumerate(split_long(text)):
            piece_items.append((i, k, piece, ctx))
    pieces = len(piece_items)
    batches = len(list(make_batches(piece_items))) if piece_items else 0
    total_chars = sum(len(t) for t, _ in segments)
    # 估算(不是硬上限): 中日文约 1 字 ≈ 1 token; 每个请求另有指令+JSON 框架开销
    input_est = int(total_chars * 0.9) + batches * 500
    output_est = int(total_chars * 1.1)
    plan.update({
        "units": units, "unit_name": "文档" if kind == KIND_EPUB else "文件",
        "segments": len(segments), "pieces": pieces,
        "requests_est": batches,
        "requests_worst": batches * (1 + 3),   # 含二分重试的最坏估计
        "total_chars": total_chars,
        "input_tokens_est": input_est,
        "output_tokens_est": output_est,
        "output_tokens_max": batches * settings.max_output_tokens_per_request,
        "target": target,
        "note": f"文本通道({settings.text_model}): 仅翻译, 思考模式={settings.text_thinking}, "
                f"并发 {settings.workers}"
                + ("，已去除注音 <rt>" if settings.strip_ruby else ""),
        "single_longest": max((len(t) for t, _ in segments), default=0),
        "max_piece": max((len(p[2]) for p in piece_items), default=0),
    })
    return plan


def format_plan(plan: Dict) -> str:
    """把预估结果排版成可读文本。"""
    lines = [
        "=" * 58,
        "预 估 结 果(未调用任何 API, 不产生费用)",
        "=" * 58,
        f"  输入      : {plan['source']}",
        f"  类型      : {KIND_DESC.get(plan['kind'], plan['kind'])}",
        f"  {plan['unit_name']:<9} : {plan['units']}",
        f"  文字片段  : {plan['segments']}",
        f"  切片后条目: {plan['pieces']}",
    ]
    if plan.get("single_longest"):
        lines.append(f"  最长原始片段: {plan['single_longest']} 字"
                     f"(已切成每片 ≤{plan.get('max_piece', 0)} 字发送)")
    lines += [
        f"  预计请求数: {plan['requests_est']}(最坏含重试约 {plan['requests_worst']})",
        f"  单次输出上限: {plan['max_output_tokens_per_request']} tokens",
        f"  预计 token : 输入 ≈{plan.get('input_tokens_est', 0)} · "
        f"输出 ≈{plan.get('output_tokens_est', 0)}"
        f"(硬上限 {plan['output_tokens_max']})",
    ]
    if plan.get("input_tokens_est"):
        # 参考价(官方 deepseek-flash 缓存未命中: 输入 $0.15/M、输出 $0.60/M, 高峰翻倍)
        lo = (plan["input_tokens_est"] * 0.15 + plan["output_tokens_est"] * 0.60) / 1e6
        hi = (plan["input_tokens_est"] * 0.30 + plan["output_tokens_est"] * 1.20) / 1e6
        lines.append(f"  参考费用  : 约 ${lo:.3f} ~ ${hi:.3f}(低峰~高峰, 以账单为准)")
    lines += [
        f"  说明      : {plan['note']}",
        "=" * 58,
        "确认规模合理后再正式运行; 可用 --max-requests / --max-output-tokens 设硬上限。",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def process_input(settings: Settings, progress=None) -> Dict:
    """按输入类型分派: PDF/图片 -> 视觉通道; EPUB/TXT -> 文本通道。"""
    src = Path(settings.pdf_path or "")
    if not src.exists():
        raise PdfError(f"找不到输入文件: {src}")
    if settings.dry_run:
        return plan_input(settings)
    kind = detect_kind(src)
    if kind in (KIND_PDF, KIND_IMAGE):
        return process_visual(settings, kind, progress=progress)
    return process_text(settings, kind, progress=progress)


# 向后兼容的旧名字
process_pdf = process_input


def run_cli(settings: Settings) -> int:
    """命令行入口, 返回退出码。"""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if settings.dry_run:
        try:
            print(format_plan(plan_input(settings)))
        except (PdfError, VisionError) as exc:
            print(f"[错误] {exc}", file=sys.stderr)
            return 1
        return 0
    try:
        summary = process_input(settings)
    except (PdfError, VisionError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        return 130

    u = summary["usage"]
    print()
    print("=" * 58)
    print("处理完成 ✅")
    print(f"  文档    : {summary['title']}")
    print(f"  输入类型: {KIND_DESC.get(summary.get('kind', ''), summary.get('kind', ''))}")
    if summary.get("images"):
        print(f"  处理量  : {summary['pages_done']} 个单元")
    print(f"  输出目录: {summary['outdir']}")
    if summary.get("txt"):
        print(f"  TXT     : {summary['txt']}")
    if summary.get("epub"):
        print(f"  EPUB    : {summary['epub']}")
    print(f"  耗时    : {summary['seconds']} 秒")
    if u["total_tokens"]:
        print(f"  Token   : 提示 {u['prompt_tokens']} + 补全 {u['completion_tokens']} "
              f"= {u['total_tokens']}")
    print("=" * 58)
    return 0
