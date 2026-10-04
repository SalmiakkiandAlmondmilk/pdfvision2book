# -*- coding: utf-8 -*-
"""真实模型测验: 插图内文字出现在不同位置时的识别效果。

做法: 为九宫格中的每个位置生成一页"插图 + 图中文字"页面(PDF),
把页面交给 deepseek-v4-flash-vision-exp, 检查:
  1. 是否识别出插图(返回了 [[FIG:n]] 占位符与 <<<FIGURES>>> JSON);
  2. 是否读出了插图中的文字(图的 text 字段包含植入的编号);
  3. 几何区域与模型描述的九宫格位置能否正确配对(裁切区域是否命中插图框)。

需要真实 API Key(会消耗少量 token); 未配置 Key 时自动跳过并以 0 退出。
用法:
    python tests/live_figure_positions.py                 # 9 个位置全测
    python tests/live_figure_positions.py --positions TL,MC,BR --min-accuracy 0.6
    python tests/live_figure_positions.py --keep-images    # 保留生成的页面图便于人工核对
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

from pdfvision.config import default_settings  # noqa: E402
from pdfvision.pdf_ingest import figure_candidates, render_pages, zone_of  # noqa: E402
from pdfvision.vision_client import VisionSession  # noqa: E402

GRID = ["TL", "TC", "TR", "ML", "MC", "MR", "BL", "BC", "BR"]
ROW_OF = {"T": 0, "M": 1, "B": 2}
COL_OF = {"L": 0, "C": 1, "R": 2}
FIG_BOX = (90.0, 380.0, 505.0, 700.0)      # 插图框(A4 页面坐标)
CODE_SUFFIX = "7351"


def build_page_pdf(path: Path, code_pos: str) -> tuple:
    """生成一页: 正文 + 一个插图框 + 框内指定位置的文字。

    返回 (pdf_path, 插图框 Rect, 植入的字符串)。
    """
    code = f"ZQ{code_pos}{CODE_SUFFIX}"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    # 正文
    page.insert_text((60, 70), "Position Test Page", fontsize=18)
    page.insert_textbox(fitz.Rect(60, 90, 535, 200),
                        "This page contains body text at the top. The illustration "
                        "below has a code printed inside it. Transcribe the code "
                        "exactly as it appears.", fontsize=11)
    page.insert_textbox(fitz.Rect(60, 700, 535, 800),
                        "Body text below the illustration as well. Keep reading "
                        "order intact.", fontsize=11)
    # 插图框(填充 + 边框 + 装饰线, 便于版面分析识别为插图区域)
    fr = fitz.Rect(*FIG_BOX)
    page.draw_rect(fr, color=(0, 0, 0), width=1.2, fill=(0.96, 0.97, 1.0))
    page.draw_line((fr.x0 + 15, fr.y1 - 40), (fr.x1 - 15, fr.y0 + 40),
                   color=(0.55, 0.6, 0.9), width=1.5)
    page.draw_circle((fr.x0 + 45, fr.y0 + 45), 18, color=(0.8, 0.5, 0.5), width=1.2)
    page.draw_rect(fitz.Rect(fr.x1 - 90, fr.y1 - 90, fr.x1 - 25, fr.y1 - 25),
                   color=(0.4, 0.7, 0.5), width=1.0)
    # 框内文字: 按九宫格位置放置
    row, col = code_pos[0], code_pos[1]
    cw, ch = fr.width / 3, fr.height / 3
    cell = fitz.Rect(fr.x0 + COL_OF[col] * cw + 8, fr.y0 + ROW_OF[row] * ch + 8,
                     fr.x0 + (COL_OF[col] + 1) * cw - 8, fr.y0 + (ROW_OF[row] + 1) * ch - 8)
    page.insert_textbox(cell, code, fontsize=15, align=COL_OF[col])
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)
    doc.close()
    return path, fr, code


def norm(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", s or "").upper()


def main() -> int:
    ap = argparse.ArgumentParser(description="插图内文字位置识别测验(真实 API)")
    ap.add_argument("--positions", default=",".join(GRID),
                    help="要测的位置, 逗号分隔, 取值: " + ",".join(GRID))
    ap.add_argument("--dpi", type=int, default=200, help="页面渲染分辨率(默认 200)")
    ap.add_argument("--min-accuracy", type=float, default=0.7,
                    help="低于该识别率则退出码非 0(默认 0.7)")
    ap.add_argument("--keep-images", action="store_true", help="保留生成的页面 PDF/图片")
    args = ap.parse_args()

    settings = default_settings()
    if not settings.api_key:
        print("未配置 DEEPSEEK_API_KEY, 跳过真实模型测验。")
        print("配置后可运行: python tests/live_figure_positions.py")
        return 0

    positions = [p.strip().upper() for p in args.positions.split(",") if p.strip()]
    bad = [p for p in positions if p not in GRID]
    if bad:
        print(f"无效位置: {bad}; 可选: {GRID}")
        return 2

    work = Path(tempfile.mkdtemp(prefix="pdfvision_live_"))
    print(f"临时目录: {work}")
    print(f"模型: {settings.vision_model} · 位置数: {len(positions)} · DPI: {args.dpi}")
    print("注意: 本测验会真实调用 API 并计费(每页 1 次请求)。\n")

    session = VisionSession(api_key=settings.api_key, base_url=settings.base_url,
                            model=settings.vision_model, timeout=settings.timeout)
    rows = []
    for pos in positions:
        pdf_path, fr, code = build_page_pdf(work / f"pos_{pos}.pdf", pos)
        images = list(render_pages(pdf_path, dpi=args.dpi))
        img = images[0]
        doc = fitz.open(pdf_path)
        try:
            page = doc.load_page(0)
            cands = figure_candidates(page)
            zones = [zone_of(r, page.rect) for r in cands]
            expect_zone = zone_of(fr, page.rect)
            result = session.transcribe_image(img, translate=False, with_original=False,
                                              candidate_count=len(cands))
        finally:
            doc.close()
        answer_all = " ".join(t for _l, t in result.blocks)
        fig_text = " ".join(f.text for f in result.figures)
        found = norm(code) in norm(fig_text) or norm(code) in norm(answer_all)
        placeholder = bool(re.search(r"\[\[\s*FIG\s*:?\s*\d+\s*\]\]", answer_all, re.I))
        # 位置配对: 模型给的 pos 能否落到插图框所在九宫格
        matched = None
        if result.figures:
            f0 = result.figures[0]
            if f0.pos:
                matched = (f0.pos == expect_zone)
            elif cands:
                # 无 pos 时按顺序配对: 第一个候选框就是插图框?
                r0 = cands[0]
                matched = (abs(r0.x0 - fr.x0) < 25 and abs(r0.y0 - fr.y0) < 25)
        rows.append({"pos": pos, "code": code, "expect_zone": expect_zone,
                     "figures": len(result.figures), "text": fig_text.strip()[:40],
                     "found": found, "marker": placeholder, "zone_ok": matched,
                     "cand": len(cands)})
        print(f"  {pos}: 候选区域 {len(cands)} · 模型插图 {len(result.figures)} · "
              f"读出={'是' if found else '否'} · 占位符={'有' if placeholder else '无'} · "
              f"区域配对={matched}")

    ok = sum(1 for r in rows if r["found"])
    n = len(rows)
    acc = ok / n if n else 0.0
    print("\n位置  期望区域     模型插图  图中文字           读出  占位符  区域配对")
    for r in rows:
        print(f"{r['pos']:>4}  {r['expect_zone']:<12} {r['figures']:>6}  "
              f"{r['text']:<18} {'✔' if r['found'] else '✘':^4}  "
              f"{'✔' if r['marker'] else '✘':^5}  "
              f"{('✔' if r['zone_ok'] else '✘' if r['zone_ok'] is False else '-'):^6}")
    usage = session.total_usage
    print(f"\n识别率: {ok}/{n} = {acc:.0%}   token: 提示 {usage.get('prompt_tokens', 0)}"
          f" + 补全 {usage.get('completion_tokens', 0)}")
    print(f"生成的测试文件保留在 {work}")
    if acc < args.min_accuracy:
        print(f"❌ 低于阈值 {args.min_accuracy:.0%}")
        return 1
    print("✅ 达到阈值")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
