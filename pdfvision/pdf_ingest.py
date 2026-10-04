"""PDF / 图片 解析: 变成可送入视觉模型的图片(JPEG/PNG 字节)。

  * page   整页渲染 —— 推荐。扫描件/含图片的版面都能按阅读顺序完整阅读。
  * images 仅提取 PDF 内嵌的位图图片(跳过重复图片), 适合只想识别图中文字的场景。
另外提供: 图片文件输入(动图拒绝)、插图候选区域检测与裁切。
"""
from __future__ import annotations

import hashlib
import io
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

try:  # pymupdf>=1.24 推荐模块名
    import pymupdf as fitz
except ImportError:  # 老版本兼容
    import fitz  # type: ignore

MAX_PAGE_SIDE = 4096          # 送到 API 前单边最大像素
MAX_EMBEDDED_SIDE = 2048      # 内嵌图片提取后单边最大像素
MEGA_PIXEL = 16_000_000
MIN_FIGURE_RATIO = 0.01       # 插图候选最小面积占比
MAX_FIGURE_RATIO = 0.72       # 超过该占比视为整页背景, 不作为插图裁切


@dataclass
class PageImage:
    """一张可发送给视觉模型的图片。"""
    page_no: int            # 1 起
    kind: str               # "page" | "embedded"
    mime: str               # image/jpeg | image/png
    data: bytes
    width: int
    height: int
    label: str = ""         # 人类可读标签, 如 "第 3 页"、"第 2 页 · 图片 1"

    @property
    def data_url(self) -> str:
        import base64
        return f"data:{self.mime};base64,{base64.b64encode(self.data).decode('ascii')}"

    @property
    def digest(self) -> str:
        """图片内容指纹(用于断点续跑时判断该图是否已处理过)。"""
        return hashlib.md5(self.data).hexdigest()


class PdfError(RuntimeError):
    pass


def _downscale_pixmap(pix, max_side: int) -> "fitz.Pixmap":
    """把 pixmap 按比例缩小, 直到短边不大于 max_side(调用方负责先转换 RGB)。"""
    guard = 0
    while max(pix.width, pix.height) > max_side and guard < 12:
        pix.shrink(2)  # 每边减半
        guard += 1
    return pix


def _pixmap_to_rgb(pix) -> "fitz.Pixmap":
    if pix.colorspace is None or pix.colorspace.n > 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    if pix.alpha:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    return pix


def _open_doc(pdf_path) -> "fitz.Document":
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # noqa: BLE001
        raise PdfError(f"无法打开 PDF: {exc}") from exc
    if doc.needs_pass:
        doc.close()
        raise PdfError("PDF 已加密且需要密码, 请先解除密码再处理。")
    return doc


# --------------------------------------------------------------------------- #
# 模式一: 整页渲染
# --------------------------------------------------------------------------- #
def render_pages(pdf_path, dpi: int = 150, max_pages: Optional[int] = None,
                 save_dir: Optional[Path] = None) -> Iterator[PageImage]:
    """把每一页渲染成 PNG 字节流。save_dir 非空时同时把图片落盘。"""
    doc = _open_doc(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    total = doc.page_count
    if max_pages:
        total = min(total, max_pages)
    try:
        for i in range(total):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix = _pixmap_to_rgb(pix)
            pix = _downscale_pixmap(pix, MAX_PAGE_SIDE)
            png = pix.tobytes("png")
            img = PageImage(
                page_no=i + 1, kind="page", mime="image/png", data=png,
                width=pix.width, height=pix.height, label=f"第 {i + 1} 页",
            )
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                (save_dir / f"page_{i + 1:04d}.png").write_bytes(png)
            yield img
    finally:
        doc.close()


# --------------------------------------------------------------------------- #
# 模式二: 提取内嵌图片
# --------------------------------------------------------------------------- #
def extract_embedded_images(pdf_path, max_pages: Optional[int] = None,
                            save_dir: Optional[Path] = None) -> Iterator[PageImage]:
    """按阅读顺序提取页面内嵌的位图图片, 全局去重(完全相同的内容只处理一次)。"""
    doc = _open_doc(pdf_path)
    total = doc.page_count
    if max_pages:
        total = min(total, max_pages)
    seen_md5: set = set()
    try:
        for i in range(total):
            page = doc.load_page(i)
            infos = page.get_image_info(xrefs=True)
            infos.sort(key=lambda it: (round(it["bbox"][1]), round(it["bbox"][0])))
            for order, info in enumerate(infos, start=1):
                xref = int(info.get("xref") or 0)
                if xref <= 0:
                    continue  # 内联图片无法单独提取, 请用整页渲染模式
                img = _extract_one(doc, page, xref, info, page_no=i + 1, order=order)
                if img is None:
                    continue
                digest = hashlib.md5(img.data).hexdigest()
                if digest in seen_md5:
                    continue
                seen_md5.add(digest)
                if save_dir is not None:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    ext = "jpg" if img.mime == "image/jpeg" else "png"
                    (save_dir / f"p{i + 1:04d}_img{order:03d}.{ext}").write_bytes(img.data)
                yield img
    finally:
        doc.close()


def _extract_one(doc, page, xref: int, info: dict, page_no: int, order: int) -> Optional[PageImage]:
    label = f"第 {page_no} 页 · 图片 {order}"
    try:
        raw = doc.extract_image(xref)
        ext = str(raw.get("ext", "")).lower()
        blob: Optional[bytes] = raw.get("image")
        if blob and ext in ("jpeg", "jpg"):
            return PageImage(page_no=page_no, kind="embedded", mime="image/jpeg",
                             data=blob, width=int(raw.get("width") or 0),
                             height=int(raw.get("height") or 0), label=label)
        if blob:
            # 其它格式(PNG/JPX/GIF/CCITT 等)统一转成 PNG
            return _blob_to_png(page_no, blob, ext, label)
    except Exception:  # noqa: BLE001  提取失败 -> 走页面区域裁剪兜底
        pass
    return _crop_fallback(page, info, page_no, order, label)


def _blob_to_png(page_no: int, blob: bytes, ext: str, label: str) -> Optional[PageImage]:
    try:
        pix = fitz.Pixmap(blob)
    except Exception:  # noqa: BLE001
        return None
    pix = _pixmap_to_rgb(pix)
    pix = _downscale_pixmap(pix, MAX_EMBEDDED_SIDE)
    return PageImage(page_no=page_no, kind="embedded", mime="image/png",
                     data=pix.tobytes("png"), width=pix.width, height=pix.height, label=label)


def _crop_fallback(page, info: dict, page_no: int, order: int, label: str) -> Optional[PageImage]:
    """无法直接解码时, 以较高分辨率渲染图片所在区域。"""
    try:
        import fitz as _  # noqa: F401
        bbox = fitz.Rect(info["bbox"])
    except Exception:  # noqa: BLE001
        return None
    if bbox.is_empty or bbox.width < 2 or bbox.height < 2:
        return None
    zoom = 300 / 72.0
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=bbox, alpha=False)
    except Exception:  # noqa: BLE001
        return None
    pix = _pixmap_to_rgb(pix)
    pix = _downscale_pixmap(pix, MAX_EMBEDDED_SIDE)
    return PageImage(page_no=page_no, kind="embedded", mime="image/png",
                     data=pix.tobytes("png"), width=pix.width, height=pix.height, label=label)


# --------------------------------------------------------------------------- #
# 工具: 页面文本快速探测(仅为日志提示, 不替代视觉阅读)
# --------------------------------------------------------------------------- #
def page_has_text_layer(pdf_path, page_no: int) -> bool:
    doc = _open_doc(pdf_path)
    try:
        txt = doc.load_page(page_no - 1).get_text("text").strip()
        return len(txt) >= 20
    except Exception:  # noqa: BLE001
        return False
    finally:
        doc.close()


# --------------------------------------------------------------------------- #
# 图片文件输入(动图除外)
# --------------------------------------------------------------------------- #
ANIMATED_MSG = ("检测到动图(GIF/WebP/APNG 多帧), 本程序不支持动图; "
                "请先另存为静态图片(如 PNG/JPG)再处理。")


def detect_animated(data: bytes) -> bool:
    """检测 GIF / WebP / APNG 是否多帧(动图)。"""
    try:
        if data[:6] in (b"GIF87a", b"GIF89a"):
            return _gif_frame_count(data) > 1
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return b"acTL" in data          # APNG 动画控制块
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return _webp_is_animated(data)
    except Exception:  # noqa: BLE001
        return False
    return False


def _gif_frame_count(data: bytes) -> int:
    """按 GIF 块结构统计图像帧数。"""
    i = 13 if len(data) > 13 else len(data)   # 跳过 6 字节签名 + 7 字节逻辑屏幕描述符
    frames = 0
    while i < len(data):
        b = data[i]
        if b == 0x3B:                          # trailer
            break
        if b == 0x21:                          # 扩展块
            i += 2
            while i < len(data) and data[i]:   # 子块
                i += data[i] + 1
            i += 1
        elif b == 0x2C:                        # 图像描述符 = 一帧
            frames += 1
            if frames > 1:
                return frames
            i += 1                             # 越过 0x2C
            if i + 9 > len(data):
                break
            packed = data[i + 8]
            i += 9                             # 图片描述符 9 字节
            if packed & 0x80:                  # 局部颜色表
                i += 3 * (2 << (packed & 0x07))
            i += 1                             # LZW 最小码长
            while i < len(data) and data[i]:   # 图像数据子块
                i += data[i] + 1
            i += 1
        else:
            i += 1
    return frames


def _webp_is_animated(data: bytes) -> bool:
    """遍历 RIFF chunk, 检测 VP8X 动画标志或 ANIM 块。"""
    # RIFF(4) + size(4) + WEBP(4)
    i = 12
    while i + 8 <= len(data):
        fourcc = data[i:i + 4]
        size = struct.unpack("<I", data[i + 4:i + 8])[0]
        if fourcc == b"ANIM":
            return True
        if fourcc == b"VP8X" and i + 12 <= len(data):
            flags = data[i + 8]
            if flags & 0x02:                   # animation flag
                return True
        i += 8 + size + (size & 1)             # chunk 按偶数字节对齐
    return False


def load_image_file(path, dpi: int = 150) -> List[PageImage]:
    """把一张静态图片文件转成 PageImage(动图直接报错)。"""
    p = Path(path)
    data = p.read_bytes()
    if detect_animated(data):
        raise PdfError(ANIMATED_MSG)
    try:
        doc = fitz.open(p)
    except Exception as exc:  # noqa: BLE001
        raise PdfError(f"无法打开图片: {exc}") from exc
    try:
        if doc.page_count < 1:
            raise PdfError("图片没有可用画面。")
        page = doc.load_page(0)
        zoom = max(dpi / 72.0, 1.0)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pix = _pixmap_to_rgb(pix)
        pix = _downscale_pixmap(pix, MAX_PAGE_SIDE)
        return [PageImage(page_no=1, kind="image", mime="image/png",
                          data=pix.tobytes("png"), width=pix.width,
                          height=pix.height, label=p.stem)]
    finally:
        doc.close()


# --------------------------------------------------------------------------- #
# 插图: 候选区域检测 / 九宫格定位 / 裁切
# --------------------------------------------------------------------------- #
def _rect_key(r) -> Tuple[float, float, float, float]:
    return (round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1))


def figure_candidates(page) -> List["fitz.Rect"]:
    """几何方式找出页面上可能是插图的区域(嵌入位图 + 矢量绘图簇), 按阅读顺序排序。"""
    page_rect = page.rect
    page_area = max(page_rect.width * page_rect.height, 1.0)
    cands: List["fitz.Rect"] = []

    # 1) 嵌入位图
    try:
        for info in page.get_image_info(xrefs=True):
            r = fitz.Rect(info["bbox"])
            if r.width < 30 or r.height < 30:
                continue
            ratio = (r.width * r.height) / page_area
            if MIN_FIGURE_RATIO <= ratio <= MAX_FIGURE_RATIO:
                cands.append(r)
    except Exception:  # noqa: BLE001
        pass

    # 2) 矢量绘图簇(图表/示意图/装饰画)
    try:
        clusters = page.cluster_drawings()
        for r in clusters:
            r = fitz.Rect(r)
            if r.width < 40 or r.height < 40:
                continue
            ratio = (r.width * r.height) / page_area
            if MIN_FIGURE_RATIO <= ratio <= MAX_FIGURE_RATIO:
                cands.append(r)
    except Exception:  # noqa: BLE001
        pass

    # 3) 去重: 被已有区域大面积覆盖的丢弃; 相近的合并保留大的
    cands.sort(key=lambda r: (-(r.width * r.height),))
    kept: List["fitz.Rect"] = []
    for r in cands:
        r_area = r.width * r.height
        dup = False
        for k in kept:
            inter = fitz.Rect(r) & k
            if inter.is_empty:
                continue
            inter_area = inter.width * inter.height
            if inter_area / max(r_area, 1.0) > 0.7:
                dup = True
                break
        if not dup:
            kept.append(r)
    kept.sort(key=lambda r: (round(r.y0), round(r.x0)))
    return kept[:8]


def zone_of(rect, page_rect=None) -> str:
    """九宫格位置: top-left … bottom-right(用于把模型的插图描述对应到几何区域)。"""
    if page_rect is None:
        page_rect = rect
    cx = (rect.x0 + rect.x1) / 2
    cy = (rect.y0 + rect.y1) / 2
    col = 0 if cx < page_rect.x0 + page_rect.width / 3 else (
        1 if cx < page_rect.x0 + page_rect.width * 2 / 3 else 2)
    row = 0 if cy < page_rect.y0 + page_rect.height / 3 else (
        1 if cy < page_rect.y0 + page_rect.height * 2 / 3 else 2)
    return ["top-left", "top", "top-right",
            "middle-left", "center", "middle-right",
            "bottom-left", "bottom", "bottom-right"][row * 3 + col]


def candidate_zones(cands: List["fitz.Rect"], page_rect) -> List[str]:
    return [zone_of(r, page_rect) for r in cands]


def crop_rect(page, rect, dpi: int = 200, max_side: int = 1600) -> Tuple[bytes, int, int]:
    """把页面上的一个区域裁切渲染成 PNG。"""
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=fitz.Rect(rect), alpha=False)
    pix = _pixmap_to_rgb(pix)
    pix = _downscale_pixmap(pix, max_side)
    return pix.tobytes("png"), pix.width, pix.height
