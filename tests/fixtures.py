# -*- coding: utf-8 -*-
"""测试用素材生成: 样例 PDF / 带文字的图片 / 动图 / 简单 EPUB。"""
from __future__ import annotations

import struct
import zipfile
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore


def build_text_png(path: Path, text: str = "HELLO VISION 12345") -> Path:
    """生成一张只有文字(图片)的 PNG, 用于图片输入测试。"""
    doc = fitz.open()
    page = doc.new_page(width=400, height=160)
    page.insert_text((30, 90), text, fontsize=20)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    pix.save(str(path))
    doc.close()
    return path


def build_animated_gif(path: Path) -> Path:
    """手工构造一个 2 帧 GIF(仅用于动图检测测试)。"""
    buf = bytearray()
    buf += b"GIF89a"
    buf += struct.pack("<HHBBB", 1, 1, 0xF0, 0, 0)   # 逻辑屏幕描述符 + 2 色全局色表
    buf += bytes([0, 0, 0, 255, 255, 255])
    buf += b"\x21\xF9\x04\x00\x0A\x00\x00\x00"        # 图形控制扩展
    for _ in range(2):                                 # 两帧 -> 动图
        buf += b"\x2C"
        buf += struct.pack("<HHHHB", 0, 0, 1, 1, 0x00)
        buf += b"\x02\x02\x4C\x01\x00"
    buf += b"\x3B"
    path.write_bytes(bytes(buf))
    return path


def build_animated_webp(path: Path) -> Path:
    """构造带 ANIM 块的 WebP 头(仅用于动图检测测试)。"""
    vp8x = b"VP8X" + struct.pack("<I", 10) + b"\x02" + b"\x00" * 9
    anim = b"ANIM" + struct.pack("<I", 6) + b"\x00" * 6
    body = b"WEBP" + vp8x + anim
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    return path


def build_simple_epub(path: Path, with_image: bool = True) -> Path:
    """构造一个含 2 章 + 1 张图片的最小 EPUB3。"""
    png = _tiny_png()
    container = ('<?xml version="1.0" encoding="utf-8"?>\n'
                 '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:'
                 'xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" '
                 'media-type="application/oebps-package+xml"/></rootfiles></container>')
    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:test-epub-0001</dc:identifier>
    <dc:title>Test Book</dc:title>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="c1" href="chap1.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="chap2.xhtml" media-type="application/xhtml+xml"/>
    <item id="pic" href="images/pic.png" media-type="image/png"/>
  </manifest>
  <spine>
    <itemref idref="c1"/>
    <itemref idref="c2"/>
  </spine>
</package>"""
    chap1 = ('<?xml version="1.0" encoding="utf-8"?>\n'
             '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>One</title></head>'
             '<body><h1>Chapter One</h1>'
             '<p>Hello world, this is the first paragraph.</p>'
             + ('<p><img src="images/pic.png" alt="pic"/></p>' if with_image else '')
             + '<p>The <b>second</b> paragraph stays inline.</p></body></html>')
    chap2 = ('<?xml version="1.0" encoding="utf-8"?>\n'
             '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Two</title></head>'
             '<body><h1>Chapter Two</h1><p>More text here.</p></body></html>')
    nav = ('<?xml version="1.0" encoding="utf-8"?>\n'
           '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub='
           '"http://www.idpf.org/2007/ops"><head><title>toc</title></head><body>'
           '<nav epub:type="toc"><ol><li><a href="chap1.xhtml">One</a></li>'
           '<li><a href="chap2.xhtml">Two</a></li></ol></nav></body></html>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip",
                    compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container)
        zf.writestr("OEBPS/content.opf", opf)
        zf.writestr("OEBPS/nav.xhtml", nav)
        zf.writestr("OEBPS/chap1.xhtml", chap1)
        zf.writestr("OEBPS/chap2.xhtml", chap2)
        if with_image:
            zf.writestr("OEBPS/images/pic.png", png)
    return path


def build_ruby_epub(path: Path, blocks: int = 48) -> Path:
    """构造一个日语小说风 EPUB: 每个 <p> 内含 <ruby>/<rt> 注音(会被切碎成多片)。"""
    paras = []
    for i in range(blocks):
        paras.append('<p class="calibre5">　それは<ruby>一九三九'
                     '<rt>いちきゅうさんきゅう</rt></ruby>年にさかのぼる。'
                     f'第{i}段落。</p>')
        paras.append('<p class="calibre5"><br class="main"/></p>')
    chap = ('<?xml version="1.0" encoding="utf-8"?>\n'
            '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="ja">'
            '<head><title>第一章</title></head><body class="vrtl">'
            + "".join(paras) + '</body></html>')
    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
 <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:identifier id="uid">urn:uuid:ruby-test-1</dc:identifier>
  <dc:title>ルビテスト</dc:title><dc:language>ja</dc:language></metadata>
 <manifest><item id="c1" href="text/chap1.html" media-type="application/xhtml+xml"/></manifest>
 <spine><itemref idref="c1"/></spine></package>"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip",
                    compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml",
                    '<?xml version="1.0"?><container version="1.0" '
                    'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                    '<rootfile full-path="OEBPS/content.opf" '
                    'media-type="application/oebps-package+xml"/></rootfiles></container>')
        zf.writestr("OEBPS/content.opf", opf)
        zf.writestr("OEBPS/text/chap1.html", chap)
    return path


def _tiny_png() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=20, height=20)
    page.draw_rect(fitz.Rect(2, 2, 18, 18), color=(1, 0, 0), width=1)
    pix = page.get_pixmap(alpha=False)
    data = pix.tobytes("png")
    doc.close()
    return data
