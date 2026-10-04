# -*- coding: utf-8 -*-
"""CLI 参数映射测试(离线, mock API): PDF / 图片 / TXT / EPUB 四种输入各跑一次。

用法: python tests/cli_test.py
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main as cli  # noqa: E402
from tests import fixtures  # noqa: E402
from tests.stub_api import start as start_stub  # noqa: E402

FAILS: list = []
LABEL_PATTERNS = [r"第\s*\d+\s*页", r"图片\s*\d+", r"\[\[FIG"]


def check(name: str, cond: bool, detail: str = ""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -> {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def no_labels(text: str) -> bool:
    return not any(re.search(p, text) for p in LABEL_PATTERNS)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="pdfvision_cli_"))
    stub = start_stub(8768)
    base = f"http://127.0.0.1:{stub.server_address[1]}"
    pdf = ROOT / "tests" / "sample.pdf"
    if not pdf.exists():
        from tests.make_sample_pdf import build
        build(pdf)
    png = fixtures.build_text_png(tmp / "pic.png")
    epub = fixtures.build_simple_epub(tmp / "book.epub")
    txt = tmp / "note.txt"
    txt.write_text("Hello CLI world.\n\nSecond line.", encoding="utf-8")
    common = ["--api-key", "k", "--base-url", base]

    # 1) PDF: 限量 1 页, 只出 EPUB
    d1 = tmp / "out_pdf"
    code = cli.main([str(pdf), *common, "--outdir", str(d1), "--max-pages", "1",
                     "--format", "epub", "--title", "CLI-PDF"])
    check("CLI PDF 退出码 0", code == 0, str(code))
    check("CLI PDF 生成 EPUB 且无 TXT",
          (d1 / "CLI-PDF.epub").exists() and not list(d1.glob("*.txt")))

    # 2) 图片: 只出 TXT
    d2 = tmp / "out_png"
    code = cli.main([str(png), *common, "--outdir", str(d2), "--format", "txt",
                     "--title", "CLI-PNG"])
    txt2 = d2 / "CLI-PNG.txt"
    check("CLI 图片退出码 0", code == 0, str(code))
    check("CLI 图片只出 TXT", txt2.exists() and not list(d2.glob("*.epub")))
    if txt2.exists():
        body2 = txt2.read_text(encoding="utf-8")
        check("CLI 图片 TXT 无页/图标记", no_labels(body2))

    # 3) TXT: 翻译成英文, 双格式
    d3 = tmp / "out_txt"
    code = cli.main([str(txt), *common, "--outdir", str(d3), "--translate", "English",
                     "--title", "CLI-TXT"])
    check("CLI TXT 退出码 0", code == 0, str(code))
    check("CLI TXT 双格式输出", (d3 / "CLI-TXT.txt").exists() and (d3 / "CLI-TXT.epub").exists())
    if (d3 / "CLI-TXT.txt").exists():
        check("CLI TXT 已翻译", "【译】" in (d3 / "CLI-TXT.txt").read_text(encoding="utf-8"))

    # 4) EPUB: 翻译, --no-figures 不影响图片保留
    d4 = tmp / "out_epub"
    code = cli.main([str(epub), *common, "--outdir", str(d4), "--translate", "简体中文",
                     "--no-figures", "--title", "CLI-EPUB"])
    check("CLI EPUB 退出码 0", code == 0, str(code))
    out_epub = d4 / "CLI-EPUB.epub"
    check("CLI EPUB 输出存在", out_epub.exists())
    if out_epub.exists():
        import zipfile
        with zipfile.ZipFile(out_epub) as zf:
            names = zf.namelist()
            body = "".join(zf.read(n).decode("utf-8", "replace")
                           for n in names if n.endswith((".xhtml", ".xml")))
        check("CLI EPUB 图片原位保留", "OEBPS/images/pic.png" in names)
        check("CLI EPUB 已翻译", "【译】" in body)

    stub.shutdown()
    print("\n" + ("CLI 检查全部通过 ✅" if not FAILS else f"有 {len(FAILS)} 项失败 ❌"))
    print(f"临时产物目录: {tmp}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
