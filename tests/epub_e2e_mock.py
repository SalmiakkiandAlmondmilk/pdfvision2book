# -*- coding: utf-8 -*-
"""对真实 EPUB 做一次"离线端到端"演练(用 mock API, 不花钱, 不消耗余额)。

用途: 遇到"某个 EPUB 跑很久/疑似卡住"时, 先用它验证:
  * 这本书会被切成多少个翻译单元/多少次请求(与 --dry-run 一致);
  * 全流程能否跑通、输出 EPUB 是否完好、图片是否原位保留、注音是否已去掉;
  * 实际请求数、token 数、耗时。

用法:
    python tests/epub_e2e_mock.py <文件.epub> [--workers 4] [--outdir DIR]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdfvision.config import Settings  # noqa: E402
from pdfvision.epub_io import read_epub, strip_ruby  # noqa: E402
from pdfvision.pipeline import plan_input, process_input  # noqa: E402
from tests import stub_api  # noqa: E402
from tests.stub_api import start as start_stub  # noqa: E402

FAILS: list = []


def check(name: str, cond: bool, detail: str = ""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -> {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    ap = argparse.ArgumentParser(description="真实 EPUB 的离线端到端演练(mock API)")
    ap.add_argument("epub")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    src = Path(args.epub)
    outdir = Path(args.outdir) if args.outdir else Path(tempfile.mkdtemp(prefix="epub_e2e_"))
    stub = start_stub()
    base = f"http://127.0.0.1:{stub.server_address[1]}"

    settings = Settings(api_key="mock", base_url=base, pdf_path=str(src),
                        outdir=str(outdir), translate="简体中文", workers=args.workers,
                        title="E2E-" + src.stem[:20])
    plan = plan_input(settings)
    print(f"文件: {src.name}")
    print(f"预估: 翻译单元 {plan['segments']} · 切片 {plan['pieces']} · "
          f"请求 {plan['requests_est']} · 最长单元 {plan.get('single_longest')} 字\n")

    t0 = time.time()
    summary = process_input(settings)
    secs = time.time() - t0

    out_epub = Path(summary["epub"])
    with zipfile.ZipFile(src) as zs, zipfile.ZipFile(out_epub) as zd:
        src_imgs = {n: zs.read(n) for n in zs.namelist() if "/images/" in n or n.endswith(
            (".jpg", ".jpeg", ".png", ".gif"))}
        dst_names = zd.namelist()
        dst_imgs = {n: zd.read(n) for n in dst_names if n in src_imgs}
        body = "".join(zd.read(n).decode("utf-8", "replace")
                       for n in dst_names if n.lower().endswith((".xhtml", ".html")))

    expected = plan["requests_est"]
    print(f"\n实际: 请求 {summary['requests']} 次 · 输出 token {summary['output_tokens']} · "
          f"耗时 {secs:.1f}s · 已译 {summary['translated_units']} 段 · "
          f"重复句命中缓存 {summary.get('cached_units', 0)} 段 · "
          f"放弃 {summary['skipped_units']} 段")
    check("流程跑通", out_epub.exists())
    check("所有翻译单元都有结果(已译+缓存=总数)",
          summary["translated_units"] + summary.get("cached_units", 0)
          + summary["skipped_units"] >= plan["segments"],
          f"{summary['translated_units']}+{summary.get('cached_units', 0)}"
          f"+{summary['skipped_units']} < {plan['segments']}")
    check("请求数与预估一致(±10%)",
          abs(summary["requests"] - expected) <= max(3, expected * 0.1),
          f"预估 {expected} 实际 {summary['requests']}")
    check("图片全部原位保留",
          len(src_imgs) > 0 and all(dst_imgs.get(n) == src_imgs[n] for n in src_imgs),
          f"源图 {len(src_imgs)} 张, 输出 {len(dst_imgs)} 张")
    check("正文已翻译", "【译】" in body)
    check("注音 <rt> 已去除", "<rt" not in body.lower())
    check("文本请求均关闭思考模式",
          all(t == "disabled" for t in stub_api.THINKING),
          f"thinking 取值={set(stub_api.THINKING)}")
    check("没有放弃的段落", summary["skipped_units"] == 0, str(summary["skipped_units"]))

    stub.shutdown()
    print("\n" + ("离线演练通过 ✅" if not FAILS else f"有 {len(FAILS)} 项失败 ❌"))
    print(f"输出目录: {outdir}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
