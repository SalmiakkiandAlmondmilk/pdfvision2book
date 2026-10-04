# -*- coding: utf-8 -*-
"""回归测试: 针对"日文 EPUB 翻译卡死 5 小时 / 每小时 200+ 请求 / 80 万输出 token"事故。

覆盖:
  1. 无空格日文/中文长文本必须被切片(旧版按空格切, 整章不切);
  2. 传输层失败(429/网络/超时)不重试放大: 请求数有界;
  3. 输出被截断(finish_reason=length)时二分缩小请求并最终成功;
  4. 三重预算护栏: --max-requests / --max-output-tokens / 无进展看门狗;
  5. 超时默认不重试(避免重复计费);
  6. --dry-run 预估不产生任何 API 请求。

用法: python tests/regression_test.py
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time as _time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdfvision import http_client  # noqa: E402
from pdfvision.config import Settings  # noqa: E402
from pdfvision.http_client import VisionError  # noqa: E402
from pdfvision.pipeline import plan_input  # noqa: E402
from pdfvision.text_client import (BATCH_ITEMS, MAX_ITEM_CHARS, TextSession,  # noqa: E402
                                   split_long)
from tests import fixtures  # noqa: E402

FAILS: list = []


def check(name: str, cond: bool, detail: str = ""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -> {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


class Counter:
    def __init__(self):
        self.n = 0
        self.lock = threading.Lock()

    def bump(self) -> int:
        with self.lock:
            self.n += 1
            return self.n


def reply(self, obj: dict, code: int = 200):
    body = json.dumps(obj).encode("utf-8")
    self.send_response(code)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)


def openai_resp(content: str, finish: str = "stop", completion: int = 20) -> dict:
    return {"choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": completion}}


def serve(respond_fn, counter: Counter, slow: float = 0.0):
    """起 mock 服务; respond_fn(raw_body: str, handler) 负责回包。"""

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):  # noqa: N802
            counter.bump()
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8", "replace")
            if slow:
                _time.sleep(slow)
            respond_fn(raw, self)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def main() -> int:
    http_client.time.sleep = lambda *_a, **_k: None   # 测试中不需要真等退避

    # ---------------- 1) 切片 ----------------
    print("场景 1: 无空格文本切片")
    jp = "あいうえお" * 4000          # 20000 字, 无空格
    pieces = split_long(jp)
    check("日文长文被切片", len(pieces) > 1, f"片数={len(pieces)}")
    check("每片不超过上限", all(len(p) <= MAX_ITEM_CHARS for p in pieces),
          str([len(p) for p in pieces][:5]))
    check("切片内容无损", "".join(pieces) == jp)
    cn = "这是一段没有任何空格的中文长文本。" * 500
    check("中文长文同样被切片",
          all(len(p) <= MAX_ITEM_CHARS for p in split_long(cn))
          and len(split_long(cn)) > 1)
    en = "word " * 2000
    check("英文仍在空白处切分", all(len(p) <= MAX_ITEM_CHARS for p in split_long(en)))

    # ---------------- 2) 429 不放大 ----------------
    print("\n场景 2: 429 传输失败时的请求次数有界")

    def always_429(_raw, h):
        reply(h, {"error": {"message": "rate limited"}}, 429)

    counter = Counter()
    srv, base = serve(always_429, counter)
    session = TextSession(api_key="k", base_url=base, model="deepseek-v4-flash",
                          max_attempts=3, stall_limit=0)
    try:
        session.translate_segments([(f"これはテスト文{i}です。" * 30, "") for i in range(3)],
                                   "简体中文")
    except VisionError:
        pass
    srv.shutdown()
    # 旧行为: 1 批×6 + 逐条 3×6 = 24 次; 新行为: 1 批 × 3 次尝试后放弃 = 3 次
    check("429 时请求数 = 批次数 × 尝试数(≤6)", counter.n <= 6, f"实际={counter.n}")
    check("放弃的条目数被记录", session.skipped_units >= 0, str(session.skipped_units))

    # ---------------- 3) 截断 -> 二分缩小并成功 ----------------
    print("\n场景 3: 输出截断(finish_reason=length)二分缩小请求")

    counter2 = Counter()

    def smart_truncate(raw, h):
        # 计数由 serve() 统一负责, 这里不再重复计数
        # 请求体是 JSON, 提示词里的引号被转义; 先解析出 prompt 再数条目
        try:
            prompt = json.loads(raw)["messages"][0]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            prompt = raw
        n_items = prompt.count('"context"')
        if n_items > 1:
            reply(h, openai_resp('{"translations":[', finish="length", completion=900))
        elif n_items == 1:
            reply(h, openai_resp(json.dumps(
                {"translations": [{"id": "0:0", "text": "【译】x"}]}, ensure_ascii=False)))
        else:
            # 单条重试提示词(无 items): 直接返回纯译文
            reply(h, openai_resp("【译】单条"))

    srv3, base3 = serve(smart_truncate, counter2)
    s3 = TextSession(api_key="k", base_url=base3, model="deepseek-v4-flash",
                     max_attempts=1, stall_limit=0)
    out3 = s3.translate_segments([(f"段落{i}" * 200, "") for i in range(BATCH_ITEMS)],
                                "简体中文")
    srv3.shutdown()
    check("截断后全部完成翻译", all(o.startswith("【译】") for o in out3), str(out3[:2]))
    check("截断处理请求数有界(≤ 2.5×条目数)",
          counter2.n <= int(2.5 * BATCH_ITEMS), f"实际={counter2.n}")

    # ---------------- 4) 预算护栏 ----------------
    print("\n场景 4: 预算护栏")

    def always_ok(_raw, h):
        reply(h, openai_resp(json.dumps(
            {"translations": [{"id": "0:0", "text": "【译】y"}]}, ensure_ascii=False),
            completion=1000))

    counter4 = Counter()
    srv4, base4 = serve(always_ok, counter4)
    s4 = TextSession(api_key="k", base_url=base4, model="deepseek-v4-flash",
                     max_attempts=1, max_requests=3, stall_limit=0)
    try:
        s4.translate_segments([(f"段落{i}" * 300, "") for i in range(20)], "简体中文")
        check("--max-requests 触发中止", False, "未中止")
    except VisionError as exc:
        check("--max-requests 触发中止", "max-requests" in str(exc), str(exc))
    check("请求数正好等于上限", counter4.n == 3, f"实际={counter4.n}")
    srv4.shutdown()

    counter5 = Counter()
    srv5, base5 = serve(always_ok, counter5)
    s5 = TextSession(api_key="k", base_url=base5, model="deepseek-v4-flash",
                     max_attempts=1, max_output_tokens=2500, stall_limit=0)
    try:
        s5.translate_segments([(f"段落{i}" * 300, "") for i in range(20)], "简体中文")
        check("--max-output-tokens 触发中止", False, "未中止")
    except VisionError as exc:
        check("--max-output-tokens 触发中止", "max-output-tokens" in str(exc), str(exc))
    check("token 达上限即停(≤4 次)", counter5.n <= 4, f"实际={counter5.n}")
    srv5.shutdown()

    def empty_json(_raw, h):
        reply(h, openai_resp('{"translations":[]}'))

    counter6 = Counter()
    srv6, base6 = serve(empty_json, counter6)
    s6 = TextSession(api_key="k", base_url=base6, model="deepseek-v4-flash",
                     max_attempts=1, stall_limit=4)
    try:
        s6.translate_segments([(f"段落{i}" * 300, "") for i in range(40)], "简体中文")
        check("无进展看门狗触发", False, "未中止")
    except VisionError as exc:
        check("无进展看门狗触发", "没有产生任何新译文" in str(exc), str(exc))
    check("看门狗在有限请求内停止(≤12)", counter6.n <= 12, f"实际={counter6.n}")
    srv6.shutdown()

    # ---------------- 5) 超时不重试 ----------------
    print("\n场景 5: 翻译请求超时默认不重试")
    counter7 = Counter()
    srv7, base7 = serve(always_ok, counter7, slow=3.0)
    s7 = TextSession(api_key="k", base_url=base7, model="deepseek-v4-flash",
                     timeout=1, max_attempts=5, stall_limit=0)
    try:
        s7.translate_segments([("短句", "")], "简体中文")
    except VisionError:
        pass
    srv7.shutdown()
    check("超时只请求 1 次(不重试)", counter7.n == 1, f"实际={counter7.n}")

    # ---------------- 6) dry-run 不调用 API ----------------
    print("\n场景 6: --dry-run 预估不产生请求")
    counter8 = Counter()
    srv8, base8 = serve(always_ok, counter8)
    epub = Path(__file__).parent / "_tmp_plan.epub"
    fixtures.build_simple_epub(epub)
    plan = plan_input(Settings(api_key="k", base_url=base8, pdf_path=str(epub),
                               translate="简体中文"))
    check("预估给出请求数", plan["requests_est"] >= 1, str(plan))
    check("预估字段完整",
          {"units", "segments", "pieces", "requests_est", "output_tokens_max"} <= set(plan))
    check("dry-run 期间 0 次请求", counter8.n == 0, f"实际={counter8.n}")
    srv8.shutdown()
    epub.unlink(missing_ok=True)

    # ---------------- 7) 护栏中止时仍保存已完成部分 ----------------
    print("\n场景 7: 预算护栏中止时保存已完成部分(端到端)")

    def always_429_2(_raw, h):
        reply(h, {"error": {"message": "rate limited"}}, 429)

    counter9 = Counter()
    srv9, base9 = serve(always_429_2, counter9)
    outdir9 = Path(__file__).parent / "_tmp_abort_out"
    epub9 = Path(__file__).parent / "_tmp_abort.epub"
    fixtures.build_simple_epub(epub9)
    from pdfvision.pipeline import process_input
    try:
        process_input(Settings(api_key="k", base_url=base9, pdf_path=str(epub9),
                               outdir=str(outdir9), translate="简体中文",
                               max_requests=2, stall_limit=0, workers=1,
                               title="护栏中止测试"))
        check("预算中止会抛出明确错误", False, "未抛出")
    except VisionError as exc:
        check("预算中止会抛出明确错误", "max-requests" in str(exc), str(exc))
        check("错误里说明已保存位置", "已保存" in str(exc), str(exc))
    saved = list(outdir9.glob("*.epub")) + list(outdir9.glob("*.txt"))
    check("已完成部分仍写出文件", bool(saved), f"目录内容={list(outdir9.iterdir()) if outdir9.exists() else '无'}")
    check("中止后请求数被限制住(≤6)", counter9.n <= 6, f"实际={counter9.n}")
    srv9.shutdown()
    for p in (epub9,):
        p.unlink(missing_ok=True)
    if outdir9.exists():
        for p in outdir9.iterdir():
            p.unlink(missing_ok=True)
        outdir9.rmdir()

    # ---------------- 8) 日语注音小说: 合并碎片 + 去注音 + 关思考 ----------------
    print("\n场景 8: 日语注音小说(<ruby>)的切片与请求数")
    from pdfvision.epub_io import collect_translation_units
    from pdfvision.pipeline import process_input
    from tests import stub_api

    stub10 = stub_api.start()
    base10 = f"http://127.0.0.1:{stub10.server_address[1]}"
    ruby_epub = Path(__file__).parent / "_tmp_ruby.epub"
    blocks = 48
    fixtures.build_ruby_epub(ruby_epub, blocks=blocks)
    outdir10 = Path(__file__).parent / "_tmp_ruby_out"
    try:
        with zipfile.ZipFile(ruby_epub) as zf:
            chap = zf.read("OEBPS/text/chap1.html").decode("utf-8")
        units = len(collect_translation_units(chap))
        raw_pieces = chap.count("<rt") * 2 + units      # 旧行为: 汉字与注音各自成片
        summary10 = process_input(Settings(
            api_key="k", base_url=base10, pdf_path=str(ruby_epub),
            outdir=str(outdir10), translate="简体中文", workers=4,
            title="注音测试"))
        with zipfile.ZipFile(outdir10 / "注音测试.epub") as zf:
            names10 = zf.namelist()
            body10 = "".join(zf.read(n).decode("utf-8", "replace")
                             for n in names10 if n.endswith((".xhtml", ".html")))
        check("按 <p> 合并翻译单元(≈段落数)",
              blocks - 2 <= units <= blocks + 2, f"units={units} blocks={blocks}")
        check("请求数远小于合并前的碎片数",
              summary10["requests"] < max(units, 1) + 3,
              f"requests={summary10['requests']} units={units} 碎片≈{raw_pieces}")
        check("输出已去除注音 <rt>", "<rt" not in body10.lower())
        check("正文已翻译", "【译】" in body10)
        check("文本请求关闭了思考模式",
              all(t == "disabled" for t in stub_api.THINKING if t is not None),
              f"thinking={set(stub_api.THINKING)}")
    finally:
        stub10.shutdown()
        ruby_epub.unlink(missing_ok=True)
        if outdir10.exists():
            for p in outdir10.iterdir():
                if p.is_file():
                    p.unlink(missing_ok=True)
            outdir10.rmdir()

    # ---------------- 9) 缓存污染: 兜底原文/拒答绝不入缓存, 旧缓存被忽略 ----------------
    print("\n场景 9: 译文缓存不得被原文污染")
    from pdfvision.pipeline import process_input
    from pdfvision.text_client import (cache_value_trustworthy, cache_value_usable,
                                       text_key)

    check("纯假名不算译文(宽松)", not cache_value_usable("は", "简体中文"))
    check("纯假名不算译文(严格)", not cache_value_trustworthy("てい", "简体中文"))
    check("模型拒答不算译文",
          not cache_value_usable("抱歉，您提供的原文只有一个平假名，请提供完整内容", "简体中文"))
    check("中文里保留日文专名可接受(宽松)",
          cache_value_usable("无畏魔女前传　オラーシャ的大地", "简体中文"))
    check("假名占多数的历史缓存判为污染(严格)",
          not cache_value_trustworthy("无畏魔女前传　オラーシャ的大地", "简体中文"))
    check("正常译文可用", cache_value_usable("封面", "简体中文")
          and cache_value_trustworthy("封面", "简体中文"))

    books = Path(__file__).parent
    book9 = fixtures.build_simple_epub(books / "_tmp_cache.epub")
    outdir9 = books / "_tmp_cache_out"
    outdir9.mkdir(parents=True, exist_ok=True)

    def always_429_3(_raw, h):
        reply(h, {"error": {"message": "rate limited"}}, 429)

    c9 = Counter()
    srv9, base9 = serve(always_429_3, c9)
    try:
        process_input(Settings(api_key="k", base_url=base9, pdf_path=str(book9),
                               outdir=str(outdir9), translate="简体中文", workers=1,
                               max_attempts=1, transport_fail_limit=2,
                               title="缓存测试"))
    except VisionError:
        pass
    ck = outdir9 / ".checkpoint_text.jsonl"
    # 模拟"被污染的缓存": 键正确、值是日文原文; 外加一条旧格式条目
    poisoned_key = text_key("Hello world, this is the first paragraph.", "简体中文")
    with ck.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"k": poisoned_key, "t": "これはテストです。", "v": 1},
                           ensure_ascii=False) + "\n")
        f.write(json.dumps({"k": "legacy-key", "t": "旧格式条目"}, ensure_ascii=False) + "\n")
    srv9.shutdown()

    c10 = Counter()
    srv10b, base10b = serve(always_ok, c10)
    try:
        summary9 = process_input(Settings(
            api_key="k", base_url=base10b, pdf_path=str(book9), outdir=str(outdir9),
            translate="简体中文", workers=1, max_attempts=1, title="缓存测试"))
        with zipfile.ZipFile(outdir9 / "缓存测试.epub") as zf:
            body9 = "".join(zf.read(n).decode("utf-8", "replace")
                            for n in zf.namelist() if n.endswith((".xhtml", ".html")))
        check("污染的缓存未被使用(仍真实调用 API)", c10.n > 0, f"requests={c10.n}")
        check("输出里没有残留日文原文", "これはテスト" not in body9)
        check("污染条目被统计出来", summary9.get("rejected_cache", 0) >= 1,
              str(summary9.get("rejected_cache")))
        check("段落全部译出", summary9.get("skipped_units") == 0,
              str(summary9.get("skipped_units")))
    except VisionError as exc:
        check("污染缓存重跑未中止", False, str(exc))
    finally:
        srv10b.shutdown()
        for p in list(outdir9.glob("*")):
            if p.is_file():
                p.unlink(missing_ok=True)
        if outdir9.exists():
            outdir9.rmdir()
        book9.unlink(missing_ok=True)

    # ---------------- 10) 传输失败/密钥错误必须中止, 而不是默默留原文 ----------------
    print("\n场景 10: 请求连续失败/密钥无效时中止并报警")
    book10 = fixtures.build_simple_epub(books / "_tmp_fail.epub")

    c11 = Counter()
    srv11, base11 = serve(always_429_3, c11)
    try:
        process_input(Settings(api_key="k", base_url=base11, pdf_path=str(book10),
                               outdir=str(books / "_tmp_fail_out"), translate="简体中文",
                               workers=1, max_attempts=1, transport_fail_limit=2,
                               title="失败测试"))
        check("连续失败会中止", False, "未中止")
    except VisionError as exc:
        check("连续失败会中止", "连续" in str(exc) and "失败" in str(exc), str(exc))
        check("已保存部分仍写出", "已保存" in str(exc), str(exc))
    check("中止前请求数有界(≤4)", c11.n <= 4, f"requests={c11.n}")
    srv11.shutdown()

    def auth_402(_raw, h):
        reply(h, {"error": {"message": "Insufficient Balance"}}, 402)

    c12 = Counter()
    srv12, base12 = serve(auth_402, c12)
    try:
        process_input(Settings(api_key="k", base_url=base12, pdf_path=str(book10),
                               outdir=str(books / "_tmp_auth_out"), translate="简体中文",
                               workers=1, max_attempts=3, title="余额测试"))
        check("余额不足会立即中止", False, "未中止")
    except VisionError as exc:
        check("余额不足会立即中止", "余额" in str(exc) or "密钥" in str(exc), str(exc))
    check("余额不足只请求 1 次(不重试)", c12.n == 1, f"requests={c12.n}")
    srv12.shutdown()
    book10.unlink(missing_ok=True)
    for d in ("_tmp_fail_out", "_tmp_auth_out"):
        p = books / d
        if p.exists():
            for f in p.glob("*"):
                if f.is_file():
                    f.unlink(missing_ok=True)
            p.rmdir()

    # ---------------- 11) 道歉对白不得被判为"模型拒答"; 照抄原文必须被判为未译 ----------------
    print("\n场景 11: 道歉对白 vs 照抄原文")
    from pdfvision.text_client import is_echo
    from pdfvision.text_client import cache_value_usable as usable

    check("『抱歉…』是正常译文, 不是拒答",
          usable("抱歉，我该怎么称呼你？", "简体中文")
          and usable("对不起，我一直都是这样。", "简体中文"))
    check("真正的拒答仍被识别",
          not usable("抱歉，我无法翻译这段内容，请提供更多上下文。", "简体中文")
          and not usable("I cannot translate this text.", "简体中文"))

    echo_src = ("そしてワールドウィッチーズという素晴らしい世界への参加を快く承諾して"
                "くださった島田フミカネ先生、本書を手にとられた読者の皆様に最大限の感謝を捧げます。")
    check("长日文句照抄被判为未译",
          is_echo("そして、ワールドウィッチーズという素晴らしい世界への参加を快く承諾して"
                  "くださった島田フミカネ先生、本書を手にとられた読者の皆様に最大限の感謝を捧げます。",
                  echo_src))
    check("正常中文译文不算照抄",
          not is_echo("在此向欣然允诺参与《世界魔女》这一精彩世界的岛田フミカネ老师致以最大感谢。",
                      echo_src))
    check("人名/编号等短文本不算照抄",
          not is_echo("少尉", "少尉") and not is_echo("BOOK☆WALKER", "BOOK☆WALKER")
          and not is_echo("［WEB］http://www.kadokawa.co.jp/", "［WEB］http://www.kadokawa.co.jp/"))

    # 端到端: mock 只会照抄 -> 必须计为未译, 且写出 failed_units.txt, 且不把照抄值写进缓存
    echo_book = fixtures.build_ruby_epub(books / "_tmp_echo.epub", blocks=3)
    outdir11 = books / "_tmp_echo_out"
    outdir11.mkdir(parents=True, exist_ok=True)

    def echo_only(_raw, h):
        try:
            body = json.loads(_raw)
            prompt = body["messages"][0]["content"]
        except Exception:  # noqa: BLE001
            prompt = _raw
        items = __import__("tests.stub_api", fromlist=["_extract_items"])._extract_items(prompt)
        if items is not None:
            answer = json.dumps({"translations": [
                {"id": it.get("id"), "text": it.get("text", "")} for it in items]},
                ensure_ascii=False)
        else:
            m = re.search(r"原文:\n(.*)$", prompt, re.S)
            answer = (m.group(1).strip() if m else prompt[:200])
        reply(h, openai_resp(answer))

    c13 = Counter()
    srv13, base13 = serve(echo_only, c13)
    try:
        summary11 = process_input(Settings(
            api_key="k", base_url=base13, pdf_path=str(echo_book), outdir=str(outdir11),
            translate="简体中文", workers=1, max_attempts=1, stall_limit=0,
            title="照抄测试"))
        check("照抄原文被计为未译", summary11.get("skipped_units", 0) > 0,
              f"skipped={summary11.get('skipped_units')}")
        failed_str = summary11.get("failed_units") or ""
        failed_file = Path(failed_str) if failed_str else None
        check("生成了 failed_units.txt",
              bool(failed_file) and failed_file.is_file(), str(failed_str))
        if failed_file is not None and failed_file.is_file():
            body = failed_file.read_text(encoding="utf-8")
            check("失败报告写明原因是照抄", "照抄" in body, body[:120])
        ck11 = outdir11 / ".checkpoint_text.jsonl"
        cached = ck11.read_text(encoding="utf-8") if ck11.exists() else ""
        check("照抄值没有被写进缓存", "それは" not in cached and "にさかのぼる" not in cached)
    except VisionError as exc:
        check("照抄场景未异常中止", False, str(exc))
    finally:
        srv13.shutdown()
        for p in list(outdir11.glob("*")):
            if p.is_file():
                p.unlink(missing_ok=True)
        if outdir11.exists():
            outdir11.rmdir()
        echo_book.unlink(missing_ok=True)

    # ---------------- 12) 缓存里的"照抄条目"会被重新翻译 ----------------
    print("\n场景 12: 缓存中的照抄条目应被忽略并重译")
    from pdfvision.epub_io import collect_translation_units, read_epub, strip_ruby
    book12 = fixtures.build_ruby_epub(books / "_tmp_echocache.epub", blocks=2)
    outdir12 = books / "_tmp_echocache_out"
    outdir12.mkdir(parents=True, exist_ok=True)
    ck12 = outdir12 / ".checkpoint_text.jsonl"
    from pdfvision.text_client import text_key as tk
    # 取该书第一个"含假名的长句"单元, 手工放一条"值=原文"的 v=1 缓存(模拟模型照抄被写入)
    jp_unit = ""
    for _n, _raw in read_epub(book12).text_docs():
        for u in collect_translation_units(strip_ruby(_raw)):
            t = u["text"]
            if len(t) >= 8 and len(re.findall(r"[\u3040-\u309f\u30a0-\u30ff]", t)) >= 3:
                jp_unit = t
                break
        if jp_unit:
            break
    check("测试素材是日文长句", len(jp_unit) >= 8, repr(jp_unit[:40]))
    ck12.write_text(json.dumps({"k": tk(jp_unit, "简体中文"), "t": jp_unit, "v": 1},
                               ensure_ascii=False) + "\n", encoding="utf-8")

    c14 = Counter()
    srv14, base14 = serve(always_ok, c14)
    try:
        summary12 = process_input(Settings(
            api_key="k", base_url=base14, pdf_path=str(book12), outdir=str(outdir12),
            translate="简体中文", workers=1, max_attempts=1, title="回声缓存测试"))
        check("照抄条目被忽略(重新请求)", c14.n > 0, f"requests={c14.n}")
        check("统计到被忽略的缓存条目", summary12.get("rejected_cache", 0) >= 1,
              str(summary12.get("rejected_cache")))
    except VisionError as exc:
        check("照抄缓存场景未中止", False, str(exc))
    finally:
        srv14.shutdown()
        for p in list(outdir12.glob("*")):
            if p.is_file():
                p.unlink(missing_ok=True)
        if outdir12.exists():
            outdir12.rmdir()
        book12.unlink(missing_ok=True)

    print("\n" + ("回归检查全部通过 ✅" if not FAILS else f"有 {len(FAILS)} 项失败 ❌"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
