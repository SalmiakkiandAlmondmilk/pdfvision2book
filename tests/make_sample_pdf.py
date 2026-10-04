# -*- coding: utf-8 -*-
"""生成一个 3 页的样例 PDF(含文字层 + 纯图片页 + 表格), 供本地冒烟测试。

用法: python tests/make_sample_pdf.py [输出路径]
"""
import sys
from pathlib import Path

import pymupdf as fitz


def build(out: Path) -> None:
    doc = fitz.open()
    W, H = 595, 842  # A4

    # ---------------- 页 1: 常规文字页(标题+段落+绘图) ----------------
    page = doc.new_page(width=W, height=H)
    page.insert_text((60, 70), "Sample Technical Report", fontsize=22)
    page.insert_text((60, 110), "Chapter 1 - Introduction", fontsize=16)
    for i, para in enumerate([
        "This is the first paragraph. It should be transcribed faithfully "
        "by the vision model, word by word, including numbers 12345 and "
        "special symbols like @#$.",
        "Machine learning is a method of data analysis that automates "
        "analytical model building. It is a branch of artificial "
        "intelligence based on the idea that systems can learn from data.",
    ]):
        page.insert_textbox(fitz.Rect(60, 150 + i * 90, W - 60, 300 + i * 60),
                            para, fontsize=11)
    rect = fitz.Rect(60, 380, 300, 560)
    page.draw_rect(rect, color=(0, 0, 1), width=1)
    page.draw_line((80, 500), (280, 430), color=(1, 0, 0), width=2)
    page.insert_text((rect.x0 + 10, rect.y1 - 12),
                     "Figure 1: sample chart placeholder", fontsize=9)

    # ---------------- 页 2: 纯图片页(模拟扫描件, 无文字层) ----------------
    tmpdoc = fitz.open()
    tp = tmpdoc.new_page(width=W, height=H)
    tp.insert_text((60, 90), "SCANNED PAGE - 01", fontsize=18)
    tp.insert_textbox(fitz.Rect(60, 120, W - 60, 320),
                      "This page has no text layer on purpose. Only an "
                      "embedded raster image exists, which is what the "
                      "vision model must read. Keep every word.", fontsize=11)
    tp.draw_rect(fitz.Rect(60, 340, 300, 470), color=(0, 0.5, 0), width=1.5)
    tp.insert_text((75, 365), "Diagram A", fontsize=11)
    pm = tp.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
    tmpdoc.close()

    page2 = doc.new_page(width=W, height=H)
    page2.insert_image(page2.rect, pixmap=pm)  # 该页唯一内容 = 这张图

    # ---------------- 页 3: 表格示例页 ----------------
    page = doc.new_page(width=W, height=H)
    page.insert_text((60, 70), "Chapter 2 - Data Table", fontsize=16)
    rows = [("Item", "Qty", "Price", "Total"),
            ("Widget A", "12", "9.50", "114.00"),
            ("Widget B", "5", "3.25", "16.25")]
    y = 120
    for row in rows:
        x = 60
        for cell in row:
            page.draw_rect(fitz.Rect(x, y, x + 130, y + 34), color=(0, 0, 0), width=0.8)
            page.insert_text((x + 6, y + 22), cell, fontsize=11)
            x += 130
        y += 36

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    doc.close()
    print(f"样例 PDF 已生成: {out} (共 3 页, 第 2 页为纯图片页)")


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1
          else Path(__file__).parent / "sample.pdf")
