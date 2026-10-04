# -*- coding: utf-8 -*-
"""离线端到端冒烟测试(不需要真实 API Key)。

覆盖: PDF/图片 -> 视觉通道(转写/翻译/插图嵌入), EPUB/TXT -> 文本通道(仅翻译),
动图拒绝, 断点续跑, 输出中不含页/图编号标记。
用法: python tests/smoke.py
"""
from __future__ import annotations

import re
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdfvision import pipeline  # noqa: E402
from pdfvision.config import Settings  # noqa: E402
from pdfvision.pdf_ingest import PdfError  # noqa: E402
from pdfvision.pipeline import process_input  # noqa: E402
from pdfvision.vision_client import Figure, PageResult  # noqa: E402
from tests import fixtures, stub_api  # noqa: E402
from tests.stub_api import start as start_stub  # noqa: E402

FAILS: list = []
LABEL_PATTERNS = [r"第\s*\d+\s*页", r"图片\s*\d+",
                  r"第\s*\d+\s*[–\-—~至]\s*\d+\s*页", r"\[\[FIG"]


def check(name: str, cond: bool, detail: str = ""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  -> {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def no_labels(text: str) -> bool:
    return not any(re.search(p, text) for p in LABEL_PATTERNS)


def epub_texts(path: Path):
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        body = "".join(zf.read(n).decode("utf-8", "replace")
                       for n in names if n.lower().endswith((".xhtml", ".html", ".xml")))
    return names, body


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="pdfvision_smoke_"))
    sample_pdf = ROOT / "tests" / "sample.pdf"
    if not sample_pdf.exists():
        print("正在生成样例 PDF …")
        from tests.make_sample_pdf import build as build_sample
        build_sample(sample_pdf)

    stub = start_stub()
    base = f"http://127.0.0.1:{stub.server_address[1]}"
    print(f"mock api: {base}\n")

    def S(name, **kw):
        kw.setdefault("api_key", "test-key")
        kw.setdefault("base_url", base)
        kw.setdefault("outdir", str(tmp / name))
        kw.setdefault("mode", "page")
        kw.setdefault("dpi", 150)
        kw.setdefault("workers", 1)
        return Settings(pdf_path=str(sample_pdf), **kw)

    # ---------------- 1) PDF 整页: 仅转写原文 ----------------
    print("场景 1: PDF 整页 · 仅转写原文(含插图嵌入)")
    stub_api.reset()
    r1 = process_input(S("s1", max_pages=2, title="冒烟-原文"))
    txt1 = Path(r1["txt"]).read_text(encoding="utf-8")
    names1, body1 = epub_texts(Path(r1["epub"]))
    check("S1 处理 2 页", r1["pages_done"] == 2, str(r1["pages_done"]))
    check("S1 TXT 含转写内容", "Mock Heading" in txt1 and "12345" in txt1)
    check("S1 TXT 含图中文字", "图内文字" in txt1)
    check("S1 TXT 无页/图标记", no_labels(txt1), txt1[:200])
    check("S1 EPUB 无页/图标记", no_labels(body1))
    check("S1 EPUB 含插图资源", any(n.startswith("OEBPS/figures/") for n in names1),
          str([n for n in names1 if "fig" in n]))
    check("S1 EPUB 用 <figure> 嵌入", "<figure" in body1)
    check("S1 EPUB 结构完整", {"mimetype", "META-INF/container.xml",
                              "OEBPS/content.opf", "OEBPS/nav.xhtml"} <= set(names1))
    check("S1 只用了视觉模型", set(stub_api.MODELS) == {"deepseek-v4-flash-vision-exp"},
          str(set(stub_api.MODELS)))

    # ---------------- 2) PDF 仅内嵌图片 ----------------
    print("\n场景 2: PDF images 模式")
    stub_api.reset()
    r2 = process_input(S("s2", mode="images", title="冒烟-图片"))
    names2, _ = epub_texts(Path(r2["epub"]))
    check("S2 处理 1 张内嵌图", r2["pages_done"] == 1, str(r2["pages_done"]))
    check("S2 无插图嵌入(未做版面分析)",
          not any(n.startswith("OEBPS/figures/") for n in names2))

    # ---------------- 3) 翻译 + 原文对照 ----------------
    print("\n场景 3: 翻译 + 原文对照")
    stub_api.reset()
    r3 = process_input(S("s3", max_pages=1, translate="简体中文", with_original=True,
                         title="冒烟-双语"))
    txt3 = Path(r3["txt"]).read_text(encoding="utf-8")
    _, body3 = epub_texts(Path(r3["epub"]))
    check("S3 TXT 含原文", "Transcribed paragraph" in txt3)
    check("S3 TXT 含译文", "模拟译文" in txt3)
    check("S3 插图译文嵌入", "(译)" in body3)
    check("S3 无页/图标记", no_labels(txt3) and no_labels(body3))

    # ---------------- 4) 仅译文 ----------------
    print("\n场景 4: 仅译文")
    stub_api.reset()
    r4 = process_input(S("s4", max_pages=1, translate="简体中文", title="冒烟-仅译文"))
    txt4 = Path(r4["txt"]).read_text(encoding="utf-8")
    check("S4 TXT 是译文", "模拟译文" in txt4)
    check("S4 无原文残留", "Transcribed paragraph" not in txt4)

    # ---------------- 5) 仅 EPUB 输出 ----------------
    print("\n场景 5: 输出格式仅 EPUB")
    stub_api.reset()
    r5 = process_input(S("s5", max_pages=1, title="冒烟-仅EPUB", formats=["epub"]))
    leftover = list(Path(r5["outdir"]).glob("*.txt"))
    check("S5 未生成 TXT", "txt" not in r5 and not leftover, str(leftover))
    check("S5 生成 EPUB", Path(r5["epub"]).exists())

    # ---------------- 6) 断点续跑 ----------------
    print("\n场景 6: 断点续跑")
    stub_api.reset()
    s6 = S("s6", max_pages=1, title="冒烟-断点")
    process_input(s6)
    check("S6 第一次 1 次视觉调用", stub_api._COUNT["vision"] == 1,
          str(stub_api._COUNT["vision"]))
    s6.max_pages = 2
    r6b = process_input(s6)
    check("S6 续跑只补 1 次", stub_api._COUNT["vision"] == 2, str(stub_api._COUNT["vision"]))
    txt6 = Path(r6b["txt"]).read_text(encoding="utf-8")
    check("S6 两页内容都在", txt6.count("Mock Heading") == 2, str(txt6.count("Mock Heading")))

    # ---------------- 7) 图片输入 ----------------
    print("\n场景 7: 图片文件输入(PNG)")
    stub_api.reset()
    png = fixtures.build_text_png(tmp / "hello.png", "HELLO VISION 12345")
    r7 = process_input(Settings(api_key="test-key", base_url=base,
                                pdf_path=str(png), outdir=str(tmp / "s7"),
                                workers=1, title="冒烟-图片输入"))
    txt7 = Path(r7["txt"]).read_text(encoding="utf-8")
    check("S7 识别为图片输入", r7["kind"] == "image", r7["kind"])
    check("S7 调用视觉模型 1 次", stub_api._COUNT["vision"] == 1)
    check("S7 TXT 含转写", "Mock Heading" in txt7)

    # ---------------- 8) 动图拒绝 ----------------
    print("\n场景 8: 动图(GIF/WebP)拒绝")
    gif = fixtures.build_animated_gif(tmp / "anim.gif")
    webp = fixtures.build_animated_webp(tmp / "anim.webp")
    for p in (gif, webp):
        try:
            process_input(Settings(api_key="test-key", base_url=base,
                                   pdf_path=str(p), outdir=str(tmp / "s8"), workers=1))
            check(f"S8 {p.suffix} 被拒绝", False, "未报错")
        except PdfError as exc:
            check(f"S8 {p.suffix} 被拒绝", "动图" in str(exc), str(exc))

    # ---------------- 9) TXT 输入(仅翻译, 文本模型) ----------------
    print("\n场景 9: TXT 输入 · 仅翻译")
    stub_api.reset()
    txt_in = tmp / "input.txt"
    txt_in.write_text("Hello world.\n\nSecond paragraph here.", encoding="utf-8")
    r9 = process_input(Settings(api_key="test-key", base_url=base, pdf_path=str(txt_in),
                                outdir=str(tmp / "s9"), workers=1, title="冒烟-TXT"))
    out9 = Path(r9["txt"]).read_text(encoding="utf-8")
    _, body9 = epub_texts(Path(r9["epub"]))
    check("S9 识别为 TXT", r9["kind"] == "txt", r9["kind"])
    check("S9 用文本模型", set(stub_api.MODELS) == {"deepseek-v4-flash"},
          str(set(stub_api.MODELS)))
    check("S9 未调用视觉模型", stub_api._COUNT["vision"] == 0)
    check("S9 TXT 已翻译", "【译】Hello world." in out9)
    check("S9 EPUB 已生成并翻译", "【译】" in body9)
    check("S9 无页/图标记", no_labels(out9) and no_labels(body9))

    # ---------------- 10) EPUB 输入(仅翻译, 图片原位保留) ----------------
    print("\n场景 10: EPUB 输入 · 仅翻译(图片原位保留)")
    stub_api.reset()
    src_epub = fixtures.build_simple_epub(tmp / "book.epub")
    r10 = process_input(Settings(api_key="test-key", base_url=base, pdf_path=str(src_epub),
                                 outdir=str(tmp / "s10"), workers=1,
                                 translate="简体中文", title="冒烟-EPUB"))
    out_epub = Path(r10["epub"])
    with zipfile.ZipFile(src_epub) as zsrc, zipfile.ZipFile(out_epub) as zdst:
        names10 = zdst.namelist()
        pic_src = zsrc.read("OEBPS/images/pic.png")
        pic_dst = zdst.read("OEBPS/images/pic.png") if "OEBPS/images/pic.png" in names10 else b""
        body10 = "".join(zdst.read(n).decode("utf-8", "replace")
                         for n in names10 if n.endswith((".xhtml", ".xml")))
        opf10 = zdst.read("OEBPS/content.opf").decode("utf-8", "replace")
    check("S10 输出 EPUB 有效", out_epub.exists() and "mimetype" in names10)
    check("S10 图片原样保留", pic_src == pic_dst and len(pic_src) > 0)
    check("S10 图片引用仍指向原路径", 'src="images/pic.png"' in body10)
    check("S10 文本已翻译", "【译】Hello world" in body10)
    check("S10 语言元数据已更新", "<dc:language>zh-CN</dc:language>" in opf10)
    check("S10 只用文本模型", set(stub_api.MODELS) == {"deepseek-v4-flash"},
          str(set(stub_api.MODELS)))
    txt10 = Path(r10["txt"]).read_text(encoding="utf-8")
    check("S10 TXT 也有译文", "【译】" in txt10)

    # ---------------- 11) 插图区域配对(九宫格) ----------------
    print("\n场景 11: 插图区域配对")
    cands = pipeline._collect_figure_candidates(sample_pdf, [1, 2])
    check("S11 第 1 页找到候选区域", bool(cands.get(1)), str(cands))
    check("S11 整页扫描图不作为插图裁切(第 2 页)", not cands.get(2), str(cands.get(2)))
    if cands.get(1):
        zone = cands[1][0][1]
        pr = PageResult(page_no=1, label="", blocks=[("原文", "")],
                        figures=[Figure(n=1, pos=zone, caption="c", text="t")])
        assets = pipeline._crop_figures(sample_pdf, [pr], cands, S("s11"), lambda *a: None)
        check("S11 按位置配对并裁切成功",
              bool(assets) and bool(pr.figures[0].asset_href)
              and assets[0]["data"][:4] == b"\x89PNG",
              str(assets[0]["href"] if assets else "无资源"))

    stub.shutdown()
    print("\n" + ("所有检查通过 ✅" if not FAILS else f"有 {len(FAILS)} 项失败 ❌"))
    print(f"临时产物目录: {tmp}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
