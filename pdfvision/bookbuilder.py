"""文档装配与输出: Markdown/插图 -> TXT / EPUB3。

设计要点:
  * 输出里**不再出现** "第 N 页""第 N 页 · 图片 N""第 1–8 页" 这类页/图标记,
    正文连排到底; 插图以图片 + 图中文字的形式就地插入。
  * EPUB 用标准库 zipfile 手工打包(EPUB3 + NCX 兼容), 插图作为资源写入 manifest。
"""
from __future__ import annotations

import html
import re
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .vision_client import Figure, PageResult

FIG_RE = re.compile(r"\[\[\s*FIG\s*:?\s*(\d+)\s*\]\]", re.I)

# --------------------------------------------------------------------------- #
# 极简 Markdown -> XHTML
# --------------------------------------------------------------------------- #
_ATX = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
_QUOTE = re.compile(r"^\s*>\s?(.*)$")
_HR = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?(:?-{2,}:?\|)+\s*$")
_FENCE = re.compile(r"^```|^~~~")
_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")


def _inline(text: str) -> str:
    """把已转义文本里的行内标记转成 HTML。text 必须已 html.escape。"""
    text = _LINK.sub(lambda m: f'<a href="{html.escape(m.group(2))}">{m.group(1)}</a>', text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"(?<!_)_([^_\n]+)_(?!_)", r"<em>\1</em>", text)
    return text


def _parse_table(lines: List[str]) -> Tuple[str, int]:
    rows_html: List[str] = []
    header: Optional[List[str]] = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("|") and "|" not in line:
            break
        if _TABLE_SEP.match(line):
            i += 1
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        cells = [_inline(html.escape(c)) for c in cells]
        if header is None:
            header = cells
        else:
            rows_html.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        i += 1
    if header is None:
        return "", 0
    thead = "<tr>" + "".join(f"<th>{c}</th>" for c in header) + "</tr>"
    return (f'<table><thead>{thead}</thead><tbody>'
            + "".join(rows_html) + "</tbody></table>"), i


def markdown_to_html(md_text: str, heading_offset: int = 0) -> str:
    """把页面结果(模型输出的 Markdown)转成干净的 XHTML 片段。"""
    md_text = md_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = md_text.split("\n")
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if _FENCE.match(stripped):
            i += 1
            buf: List[str] = []
            while i < n and not _FENCE.match(lines[i].strip()):
                buf.append(html.escape(lines[i]))
                i += 1
            i += 1
            out.append("<pre>" + "\n".join(buf) + "</pre>")
            continue

        if stripped.startswith("|"):
            tbl, consumed = _parse_table(lines[i:])
            if consumed:
                out.append(tbl)
                i += consumed
                continue

        m = _ATX.match(line)
        if m:
            level = min(len(m.group(1)) + heading_offset, 6)
            out.append(f"<h{level}>{_inline(html.escape(m.group(2).strip()))}</h{level}>")
            i += 1
            continue

        if _HR.match(stripped):
            out.append("<hr/>")
            i += 1
            continue

        if _QUOTE.match(line):
            buf = []
            while i < n and _QUOTE.match(lines[i]):
                buf.append(_QUOTE.match(lines[i]).group(1))  # type: ignore[union-attr]
                i += 1
            out.append("<blockquote><p>" + _inline(html.escape(" ".join(buf))) + "</p></blockquote>")
            continue

        if _LIST.match(line):
            items_html: List[str] = []
            ordered = None
            while i < n:
                m2 = _LIST.match(lines[i])
                if not m2:
                    break
                _, marker, text = m2.groups()
                if ordered is None:
                    ordered = marker.isdigit()
                items_html.append("<li>" + _inline(html.escape(text)) + "</li>")
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(items_html) + f"</{tag}>")
            continue

        if stripped:
            buf2 = [stripped]
            i += 1
            while i < n and lines[i].strip() and not _ATX.match(lines[i]) \
                    and not _LIST.match(lines[i]) and not _QUOTE.match(lines[i]) \
                    and not _HR.match(lines[i]) and not _FENCE.match(lines[i]) \
                    and not lines[i].strip().startswith("|"):
                buf2.append(lines[i].strip())
                i += 1
            joined = _inline(html.escape(" ".join(buf2)))
            if joined:
                out.append(f"<p>{joined}</p>")
            continue

        i += 1
    body = "\n".join(out)
    return body if body.strip() else "<p></p>"


def markdown_to_plain(md_text: str) -> str:
    body = markdown_to_html(md_text)
    body = re.sub(r"<br\s*/?>", "\n", body)
    body = re.sub(r"</(p|h[1-6]|li|tr|div|pre|blockquote)>", "\n", body)
    body = re.sub(r"<[^>]+>", "", body)
    body = (body.replace("&lt;", "<").replace("&gt;", ">")
                .replace("&quot;", '"').replace("&#39;", "'")
                .replace("&amp;", "&"))
    return re.sub(r"\n{3,}", "\n\n", body).strip()


# --------------------------------------------------------------------------- #
# 插图渲染(按 [[FIG:n]] 占位符就地替换, 不带任何编号文字)
# --------------------------------------------------------------------------- #
def _figures_of(p: PageResult) -> Dict[int, Figure]:
    return {f.n: f for f in (p.figures or [])}


def _figure_caption_html(fig: Figure, translated: bool) -> str:
    parts: List[str] = []
    body = fig.body_text(translated)
    if body:
        parts.append(html.escape(body).replace("\n", "<br/>"))
    cap = fig.caption_text(translated)
    if cap and cap != body:
        parts.append(f'<span class="fig-cap">{html.escape(cap)}</span>')
    if not parts:
        return ""
    return "<figcaption>" + " ".join(parts) + "</figcaption>"


def figure_html(fig: Optional[Figure], translated: bool) -> str:
    if fig is None:
        return ""
    cap = _figure_caption_html(fig, translated)
    if fig.asset_href:
        return (f'<figure class="fig"><img src="{html.escape(fig.asset_href)}" '
                f'alt=""/>{cap}</figure>')
    if not cap:
        return ""
    return f'<p class="fig-note">{cap}</p>'


def figure_plain(fig: Optional[Figure], translated: bool) -> str:
    if fig is None:
        return ""
    body = fig.body_text(translated)
    cap = fig.caption_text(translated)
    if body and cap and cap != body:
        return f"{body}\n〔{cap}〕"
    return body or cap


def _md_with_figures_html(md: str, figs: Dict[int, Figure], translated: bool,
                          heading_offset: int = 2) -> str:
    parts = FIG_RE.split(md)
    out: List[str] = []
    for i, seg in enumerate(parts):
        if i % 2 == 0:
            out.append(markdown_to_html(seg, heading_offset=heading_offset))
        else:
            try:
                n = int(seg)
            except ValueError:
                continue
            out.append(figure_html(figs.get(n), translated))
    return "\n".join(x for x in out if x.strip())


def _md_with_figures_plain(md: str, figs: Dict[int, Figure], translated: bool) -> str:
    parts = FIG_RE.split(md)
    out: List[str] = []
    for i, seg in enumerate(parts):
        if i % 2 == 0:
            out.append(markdown_to_plain(seg))
        else:
            try:
                n = int(seg)
            except ValueError:
                continue
            out.append(figure_plain(figs.get(n), translated))
    return "\n\n".join(x for x in out if x.strip())


# --------------------------------------------------------------------------- #
# 页面 -> 文本片段
# --------------------------------------------------------------------------- #
def page_html(p: PageResult, translate: bool, with_original: bool) -> str:
    figs = _figures_of(p)
    parts: List[str] = []
    if not translate or with_original:
        orig = p.original if translate else p.primary
        if orig:
            parts.append(_md_with_figures_html(orig, figs, translated=False))
    if translate:
        trans = p.translation
        if trans:
            if with_original and parts:
                parts.append('<p class="tr-sep">— 译文 —</p>')
            parts.append(_md_with_figures_html(trans, figs, translated=True))
    return "\n".join(x for x in parts if x.strip()) or "<p><i>(本页无可用内容)</i></p>"


def page_plain(p: PageResult, translate: bool, with_original: bool) -> str:
    figs = _figures_of(p)
    parts: List[str] = []
    if not translate or with_original:
        orig = p.original if translate else p.primary
        if orig:
            parts.append(_md_with_figures_plain(orig, figs, translated=False))
    if translate:
        trans = p.translation
        if trans:
            if with_original and parts:
                parts.append("— 译文 —")
            parts.append(_md_with_figures_plain(trans, figs, translated=True))
    return "\n\n".join(x for x in parts if x.strip()) or "(本页无可用内容)"


def group_pages_into_chapters(pages: List[PageResult], per_chapter: int) -> List[List[PageResult]]:
    """把逐页结果分组(仅用于 EPUB 内部分段, 不产生任何可见标题)。"""
    if per_chapter <= 0:
        return [[p] for p in pages]
    return [pages[i:i + per_chapter] for i in range(0, len(pages), per_chapter)]


# --------------------------------------------------------------------------- #
# 输出: TXT / EPUB
# --------------------------------------------------------------------------- #
_HEADER_KEYS = (("source", "来源"), ("mode_desc", "处理方式"), ("translate", "目标语言"),
                ("model", "模型"), ("time", "生成时间"), ("outdir_note", "提示"))


def _header_lines(title: str, meta: Dict[str, str]) -> List[str]:
    lines = [title, "=" * max(len(title), 4), ""]
    for key, label in _HEADER_KEYS:
        val = (meta or {}).get(key)
        if val:
            lines.append(f"{label}: {val}")
    if len(lines) > 3:
        lines.append("")
    return lines


def write_txt(pages: List[PageResult], translate: bool, with_original: bool,
              title: str, meta: Dict[str, str], path: Path) -> None:
    """正文连排, 无页/图标记。"""
    chunks = [page_plain(p, translate, with_original) for p in pages]
    body = "\n\n".join(c for c in chunks if c.strip())
    Path(path).write_text("\n".join(_header_lines(title, meta)) + body + "\n",
                          encoding="utf-8")


def write_text_file(title: str, meta: Dict[str, str], body: str, path: Path) -> None:
    """写纯文本输出(EPUB/TXT 输入翻译用)。"""
    Path(path).write_text("\n".join(_header_lines(title, meta)) + body.strip() + "\n",
                          encoding="utf-8")


def _xml(s: str) -> str:
    return html.escape(s)


def _wrap_xhtml(title: str, body: str, lang: str, css_href: str = "style.css") -> str:
    return (f'<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n'
            f'<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub='
            f'"http://www.idpf.org/2007/ops" lang="{_xml(lang)}">\n'
            f'<head><title>{_xml(title)}</title>'
            f'<link rel="stylesheet" type="text/css" href="{css_href}"/></head>\n'
            f'<body>\n{body}\n</body></html>')


_CSS = """
body { font-family: Georgia, "Source Han Serif SC", "Noto Serif CJK SC", serif;
       line-height: 1.75; margin: 5% 6%; }
h1 { font-size: 1.6em; } h2 { font-size: 1.35em; margin-top: 1.4em; }
h3 { font-size: 1.15em; } h4, h5, h6 { font-size: 1em; }
p  { text-align: justify; margin: 0.6em 0; }
blockquote { border-left: 3px solid #bbb; margin-left: 1em; padding-left: 1em;
             color: #444; }
table { border-collapse: collapse; margin: 0.8em auto; }
th, td { border: 1px solid #999; padding: 0.3em 0.6em; font-size: 0.92em; }
pre { font-size: 0.85em; white-space: pre-wrap; background: #f4f4f4; padding: .5em; }
code { font-family: Consolas, monospace; }
.tr-sep { color: #777; font-style: italic; margin: 1em 0 .4em;
          border-bottom: 1px dotted #ccc; }
figure.fig { margin: 1em auto; text-align: center; page-break-inside: avoid; }
figure.fig img { max-width: 100%; height: auto; }
figcaption { font-size: 0.88em; color: #444; margin-top: .4em; text-align: center; }
.fig-cap { display: block; color: #666; font-style: italic; }
p.fig-note { font-size: 0.9em; color: #555; }
"""


def write_epub(pages: List[PageResult], translate: bool, with_original: bool,
               title: str, meta: Dict[str, str], path: Path,
               per_chapter: int = 8, lang: str = "zh-CN",
               assets: Optional[List[dict]] = None) -> None:
    """写 EPUB3。章节只用于内部分段与目录, 正文里不出现页/图编号文字。"""
    chapters = group_pages_into_chapters(pages, per_chapter)
    assets = assets or []
    book_id = "urn:uuid:" + str(uuid.uuid4())
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    chapter_files: List[Tuple[str, str, str]] = []
    for idx, chunk in enumerate(chapters, start=1):
        body = "\n".join(page_html(p, translate, with_original) for p in chunk)
        fname = f"chap{idx:04d}.xhtml"
        chapter_files.append((fname, f"chapter-{idx}",
                              _wrap_xhtml(title, body, lang)))

    manifest = ['<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
                '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" '
                'properties="nav"/>',
                '<item id="css" href="style.css" media-type="text/css"/>']
    spine: List[str] = []
    for idx, (fname, cid, _x) in enumerate(chapter_files, start=1):
        manifest.append(f'<item id="{cid}" href="{fname}" media-type="application/xhtml+xml"/>')
        spine.append(f'<itemref idref="{cid}"/>')
    for asset in assets:
        manifest.append(f'<item id="{_xml(asset["id"])}" href="{_xml(asset["href"])}" '
                        f'media-type="{_xml(asset.get("media_type", "image/png"))}"/>')

    opf = f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">{book_id}</dc:identifier>
    <dc:title>{_xml(title)}</dc:title>
    <dc:language>{_xml(lang)}</dc:language>
    <dc:creator>pdfvision2book</dc:creator>
    <meta property="dcterms:modified">{now}</meta>
    <meta name="generator" content="pdfvision2book"/>
    <meta name="source-document" content="{_xml(meta.get('source', ''))}"/>
  </metadata>
  <manifest>
    {chr(10).join(manifest)}
  </manifest>
  <spine toc="ncx">
    {chr(10).join(spine)}
  </spine>
</package>"""

    nav_items = "".join(
        f'<li><a href="{fname}">{idx}</a></li>'
        for idx, (fname, _cid, _x) in enumerate(chapter_files, start=1))
    nav_xhtml = _wrap_xhtml("目录",
                            f'<nav epub:type="toc"><ol>{nav_items}</ol></nav>', lang)
    nav_points = "".join(
        f'<navPoint id="np{idx}" playOrder="{idx}"><navLabel><text>{idx}</text>'
        f'</navLabel><content src="{fname}"/></navPoint>'
        for idx, (fname, _cid, _x) in enumerate(chapter_files, start=1))
    ncx = f"""<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head><meta name="dtb:uid" content="{book_id}"/></head>
  <docTitle><text>{_xml(title)}</text></docTitle>
  <navMap>{nav_points}</navMap>
</ncx>"""

    container = ('<?xml version="1.0" encoding="utf-8"?>\n'
                 '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:'
                 'xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" '
                 'media-type="application/oebps-package+xml"/></rootfiles></container>')

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip",
                    compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container)
        zf.writestr("OEBPS/content.opf", opf)
        zf.writestr("OEBPS/nav.xhtml", nav_xhtml)
        zf.writestr("OEBPS/toc.ncx", ncx)
        zf.writestr("OEBPS/style.css", _CSS)
        for fname, _cid, xhtml_body in chapter_files:
            zf.writestr(f"OEBPS/{fname}", xhtml_body)
        for asset in assets:
            zf.writestr(f"OEBPS/{asset['href']}", asset["data"])
