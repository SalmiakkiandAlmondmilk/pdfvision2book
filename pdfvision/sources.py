"""输入类型探测与分派。

支持四类输入:
  * PDF        -> 视觉模型(整页渲染 / 仅内嵌图片)
  * 图片文件    -> 视觉模型(仅支持静态图; GIF/WebP/APNG 动图拒绝)
  * EPUB       -> 纯文本模型, 仅翻译, 图片原位保留
  * TXT        -> 纯文本模型, 仅翻译
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from .config import TEXT_EXTS, VISUAL_EXTS, Settings
from .pdf_ingest import (PdfError, PageImage, extract_embedded_images,
                         figure_candidates, load_image_file, render_pages)

KIND_PDF = "pdf"
KIND_IMAGE = "image"
KIND_EPUB = "epub"
KIND_TXT = "txt"

KIND_DESC = {
    KIND_PDF: "PDF(视觉模型逐页阅读)",
    KIND_IMAGE: "图片(视觉模型识别)",
    KIND_EPUB: "EPUB(纯文本模型仅翻译, 图片原位保留)",
    KIND_TXT: "TXT(纯文本模型仅翻译)",
}


def detect_kind(path) -> str:
    """按扩展名 + 文件头判断输入类型。"""
    p = Path(path)
    if not p.exists():
        raise PdfError(f"找不到输入文件: {p}")
    ext = p.suffix.lower()
    head = p.open("rb").read(8)
    if head[:5] == b"%PDF-":
        return KIND_PDF
    if head[:4] == b"PK\x03\x04" or ext == ".epub":
        return KIND_EPUB
    if ext in VISUAL_EXTS:
        return KIND_IMAGE
    if ext in TEXT_EXTS:
        return KIND_TXT
    raise PdfError(f"不支持的文件类型: {p.name}(支持 PDF / 图片 / EPUB / TXT)")


def load_visual_images(settings: Settings, save_dir=None) -> List[PageImage]:
    """视觉通道取图。"""
    p = Path(settings.pdf_path)
    kind = detect_kind(p)
    if kind == KIND_IMAGE:
        return load_image_file(p)
    if settings.mode == "images":
        return list(extract_embedded_images(p, max_pages=settings.max_pages,
                                            save_dir=save_dir))
    return list(render_pages(p, dpi=settings.dpi, max_pages=settings.max_pages,
                             save_dir=save_dir))


def page_figure_candidates(settings: Settings, page_no: int, save_dir=None):
    """返回 [(rect, zone)]: 该页插图候选区域(仅 PDF 且开启插图保留时调用)。"""
    if not settings.keep_figures or settings.mode != "page":
        return []
    p = Path(settings.pdf_path)
    if detect_kind(p) != KIND_PDF:
        return []
    import pymupdf as _fitz  # noqa: F401  (确认可用)
    try:
        import pymupdf as fitz
    except ImportError:  # pragma: no cover
        import fitz  # type: ignore
    doc = fitz.open(p)
    try:
        if page_no - 1 >= doc.page_count:
            return []
        page = doc.load_page(page_no - 1)
        cands = figure_candidates(page)
        from .pdf_ingest import zone_of
        return [(tuple(r), zone_of(r, page.rect)) for r in cands]
    finally:
        doc.close()


def read_txt_text(path) -> str:
    """读取 TXT(自动尝试常见编码)。"""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "big5", "utf-16"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")
