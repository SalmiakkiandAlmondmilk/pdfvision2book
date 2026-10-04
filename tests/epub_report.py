# -*- coding: utf-8 -*-
"""EPUB 结构诊断: 逐文档统计文字片段/长度/切片数, 找出异常文档。

用法: python tests/epub_report.py <文件.epub>
不调用任何 API。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdfvision.epub_io import collect_translation_units, read_epub, strip_ruby  # noqa: E402
from pdfvision.text_client import (BATCH_CHARS, BATCH_ITEMS, MAX_ITEM_CHARS,  # noqa: E402
                                   make_batches, split_long)


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python tests/epub_report.py <文件.epub> [--keep-ruby]")
        return 2
    path = Path(sys.argv[1])
    keep_ruby = "--keep-ruby" in sys.argv
    book = read_epub(path)
    print(f"文件: {path.name}  ({path.stat().st_size / 1048576:.2f} MB)")
    print(f"条目总数: {len(book.entries)}  待翻译文档: {len(book.doc_names)}"
          f"  注音: {'保留' if keep_ruby else '去除'}\n")
    print(f"{'#':>3} {'文档':<30} {'KB':>7} {'翻译单元':>8} {'总字符':>8} "
          f"{'最长单元':>8} {'批次':>6}")
    total_units = total_chars = total_batches = 0
    for i, (name, raw_text) in enumerate(book.text_docs(), start=1):
        text = raw_text if keep_ruby else strip_ruby(raw_text)
        units = collect_translation_units(text)
        chars = sum(len(u["text"]) for u in units)
        longest = max((len(u["text"]) for u in units), default=0)
        pieces = []
        for gi, u in enumerate(units):
            for k, p in enumerate(split_long(u["text"])):
                pieces.append((gi, k, p, u["ctx"]))
        batches = len(list(make_batches(pieces))) if pieces else 0
        total_units += len(units)
        total_chars += chars
        total_batches += batches
        flag = "  <-- 大文档" if len(units) > 800 else ""
        print(f"{i:>3} {name[-30:]:<30} {len(text.encode('utf-8')) / 1024:>7.0f} "
              f"{len(units):>8} {chars:>8} {longest:>8} {batches:>6}{flag}")
    print(f"\n合计: 翻译单元 {total_units} · 字符 {total_chars} · "
          f"批次(=预计请求数) {total_batches}")
    print(f"单片上限 {MAX_ITEM_CHARS} 字 · 每批最多 {BATCH_ITEMS} 条 / {BATCH_CHARS} 字")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
