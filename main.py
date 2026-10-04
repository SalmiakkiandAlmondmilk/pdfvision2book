#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pdfvision2book 入口。

支持四类输入, 自动识别:
  * PDF / 图片(jpg png webp bmp tif gif 静态图, 动图不支持)
      -> 视觉模型 deepseek-v4-flash-vision-exp: 识别文字(+可选翻译),
         识别插图并裁切, 在 EPUB 中原位嵌入;
  * EPUB / TXT
      -> 纯文本模型 deepseek-v4-flash: 仅翻译; EPUB 的图片原位保留。

命令行示例:
    python main.py 扫描件.pdf                       # 转写原文, 输出 TXT + EPUB
    python main.py book.pdf --translate 简体中文     # 读出后翻成中文
    python main.py book.pdf --translate 简体中文 --with-original
    python main.py 插图页.png --translate 简体中文    # 图片输入: 识别文字+翻译
    python main.py 英文书.epub --translate 简体中文    # EPUB: 仅翻译, 图片原位保留
    python main.py 笔记.txt --translate English       # TXT: 仅翻译
    python main.py book.pdf --format epub --no-figures
    python main.py book.pdf --mode images --outdir out

本地网页上传界面:
    python main.py --serve [--host 0.0.0.0] [--port 8000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pdfvision.config import Settings, default_settings, load_dotenv
from pdfvision.pipeline import run_cli


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdfvision2book",
        description="DeepSeek 视觉/文本模型驱动的转译程序: PDF/图片 -> 转写(可翻译) + 插图嵌入; "
                    "EPUB/TXT -> 仅翻译。输出 TXT 与 EPUB3(独立程序, 非插件)。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("input", nargs="?", help="输入文件: PDF / 图片 / EPUB / TXT")
    p.add_argument("--serve", action="store_true",
                   help="启动本地网页上传界面(浏览器打开 http://127.0.0.1:8000)")
    p.add_argument("--host", default="127.0.0.1", help="网页模式监听地址(默认 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="网页模式端口(默认 8000)")

    g_api = p.add_argument_group("API 配置")
    g_api.add_argument("--api-key", default=None,
                       help="DeepSeek API Key(默认读环境变量 DEEPSEEK_API_KEY 或 .env)")
    g_api.add_argument("--base-url", default=None,
                       help="API 地址(默认 https://api.deepseek.com)")
    g_api.add_argument("--vision-model", "--model", dest="vision_model", default=None,
                       help="读图模型(默认 deepseek-v4-flash-vision-exp)")
    g_api.add_argument("--text-model", default=None,
                       help="纯文本翻译模型(默认 deepseek-v4-flash)")
    g_api.add_argument("--timeout", type=int, default=None, help="单次请求超时秒数")

    g_in = p.add_argument_group("读图设置(仅 PDF / 图片输入)")
    g_in.add_argument("--mode", choices=["page", "images"], default="page",
                      help="PDF: page=整页渲染阅读(默认, 适合扫描件); images=仅读内嵌图片")
    g_in.add_argument("--dpi", type=int, default=None, help="整页渲染分辨率, 默认 150")
    g_in.add_argument("--max-pages", type=int, default=None, help="只处理前 N 页(试跑用)")
    g_in.add_argument("--save-pages", action="store_true",
                      help="把送入模型的图片保存到输出目录 pages/ 便于核对")
    g_in.add_argument("--keep-figures", dest="keep_figures", action="store_true",
                      default=None, help="识别插图并裁切, 在 EPUB 中原位嵌入(默认开)")
    g_in.add_argument("--no-figures", dest="keep_figures", action="store_false",
                      help="不嵌入插图, 仅保留图中文字")
    g_in.add_argument("--figure-dpi", type=int, default=None,
                      help="插图裁切分辨率, 默认 200")

    g_out = p.add_argument_group("转译与输出")
    g_out.add_argument("--outdir", default=None,
                       help="输出目录(默认在输入文件旁建 <文件名>_vision 文件夹)")
    g_out.add_argument("--title", default=None, help="输出书名(默认用文件名)")
    g_out.add_argument("--format", dest="fmt", choices=["txt", "epub", "both"],
                       default="both", help="输出格式: txt / epub / both(默认 both)")
    g_out.add_argument("--translate", default=None,
                       help="目标语言名, 例如: 简体中文 / English / 日本語。"
                            "PDF/图片: 不填=仅转写原文; EPUB/TXT: 不填=默认翻成简体中文")
    g_out.add_argument("--with-original", action="store_true",
                       help="翻译时保留原文对照(与 --translate 一起使用)")
    g_out.add_argument("--pages-per-chapter", type=int, default=None,
                       help="EPUB 每章包含的页数, 默认 8; 0 = 每页一章")
    g_out.add_argument("--no-resume", action="store_true",
                       help="不使用断点续跑, 已完成的页面也重新识别(默认会自动续跑)")
    g_out.add_argument("--workers", type=int, default=None,
                       help="并发请求数, 默认 2(留意账号速率限制)")

    g_safe = p.add_argument_group("费用护栏(强烈建议按需设置)")
    g_safe.add_argument("--dry-run", action="store_true",
                        help="只预估处理量/请求数/token 上限, 不调用 API(不花钱)")
    g_safe.add_argument("--max-requests", type=int, default=0,
                        help="本次最多请求次数, 达到即中止; 0=不限制")
    g_safe.add_argument("--max-output-tokens", type=int, default=0,
                        help="累计补全 token 上限, 达到即中止; 0=不限制")
    g_safe.add_argument("--stall-limit", type=int, default=25,
                        help="连续多少次请求没有新译文就中止(默认 25, 防空转烧钱)")
    g_safe.add_argument("--max-output-tokens-per-request", type=int, default=8192,
                        help="单次请求输出上限(默认 8192, 防止单次生成失控)")
    g_safe.add_argument("--text-thinking", choices=["disabled", "enabled", "auto"],
                        default="disabled",
                        help="文本翻译的思考模式(默认 disabled: 推理 token 既贵又慢)")
    g_safe.add_argument("--vision-thinking", choices=["disabled", "enabled", "auto"],
                        default="auto",
                        help="读图请求的思考模式(默认 auto, 用服务端默认)")
    g_safe.add_argument("--keep-ruby", action="store_true",
                        help="保留日语注音 <rt>(默认在翻译时去掉, 译文不需要假名)")
    return p


def main(argv=None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.serve and not args.input:
        parser.error("请指定输入文件(PDF / 图片 / EPUB / TXT), 或使用 --serve 启动网页上传模式。\n"
                     "示例: python main.py 我的扫描件.pdf")

    defaults = default_settings()
    settings = Settings()
    settings.api_key = args.api_key or defaults.api_key
    settings.base_url = args.base_url or defaults.base_url
    settings.vision_model = args.vision_model or defaults.vision_model
    settings.text_model = args.text_model or defaults.text_model
    settings.timeout = args.timeout or defaults.timeout
    settings.pdf_path = args.input
    settings.mode = args.mode
    settings.dpi = args.dpi if args.dpi else defaults.dpi
    settings.max_pages = args.max_pages
    settings.save_pages = args.save_pages
    settings.keep_figures = (defaults.keep_figures if args.keep_figures is None
                             else args.keep_figures)
    settings.figure_dpi = args.figure_dpi or defaults.figure_dpi
    settings.translate = args.translate
    settings.with_original = args.with_original
    settings.pages_per_chapter = (args.pages_per_chapter
                                  if args.pages_per_chapter is not None
                                  else defaults.pages_per_chapter)
    settings.workers = args.workers if args.workers else defaults.workers
    settings.resume = not args.no_resume
    settings.outdir = args.outdir
    settings.title = args.title or ""
    settings.formats = {"txt": ["txt"], "epub": ["epub"],
                        "both": ["txt", "epub"]}[args.fmt]
    # 费用护栏
    settings.dry_run = args.dry_run
    settings.max_requests = args.max_requests
    settings.max_output_tokens = args.max_output_tokens
    settings.stall_limit = args.stall_limit
    settings.max_output_tokens_per_request = args.max_output_tokens_per_request
    settings.text_thinking = args.text_thinking
    settings.vision_thinking = args.vision_thinking
    settings.strip_ruby = not args.keep_ruby

    if args.serve:
        from pdfvision.webui import serve
        return serve(settings, host=args.host, port=args.port)
    return run_cli(settings)


if __name__ == "__main__":
    sys.exit(main())
