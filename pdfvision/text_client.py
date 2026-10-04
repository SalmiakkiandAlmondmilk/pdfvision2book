"""纯文本模型客户端: EPUB / TXT 输入时"仅翻译"(默认 deepseek-v4-flash)。

安全设计(每一条都对应一次真实事故):
  1. 切片按"字符数"切, 句末标点/空白优先 —— 中日文没有空格, 旧版按空格切会导致
     整章(几万字)当成一条发送, 单次输出可达上万 token;
  2. 每次请求带 max_tokens, 输出长度有硬上限;
  3. 批量失败改为"二分重试"(深度受限), 不再对每条独立重试 —— 避免请求放大 8 倍;
  4. 输出被截断(finish_reason=length)视为需要缩小请求, 而不是失败重试;
  5. 三重预算护栏: --max-requests / --max-output-tokens / 无进展看门狗;
  6. 超时默认不重试(长输出重试会重复计费)。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .http_client import AuthError, ChatHTTP, TransportError, VisionError
from .prompts import TEXT_TRANSLATE_PROMPT, TEXT_TRANSLATE_RETRY_PROMPT
from .vision_client import parse_json_loose

log = logging.getLogger("pdfvision")

# 缓存格式/语义版本: 进入缓存键。改版后旧条目自动失效(不会被误用)
CACHE_VERSION = "t5"
CACHE_RECORD_VERSION = 1     # checkpoint 行里的 "v" 字段

# 译文里不该出现的东西(出现即视为"模型拒答")
# 注意: 绝不能把「抱歉/对不起/sorry」列进来 —— 对白里本来就有道歉的话
REFUSAL_MARKERS = ("无法翻译", "无法完成", "不能翻译", "无法为你翻译", "请提供",
                   "作为ai", "作为 ai", "我是一个ai", "我是一个人工智能",
                   "i cannot", "i can't", "i'm unable", "as an ai",
                   "unable to translate", "cannot translate")

# 用于"照抄原文"比对: 去掉空白与标点后再比
_ECHO_STRIP = re.compile(
    r"[\s\u3000、。，,．.！!？?…「」『』（）()\[\]【】〈〉《》\-—ー~〜\"'“”‘’·]")

BATCH_CHARS = 1800      # 单次请求的最大字符数(输入侧)
BATCH_ITEMS = 24        # 单次请求的最大条目数(短句合并后请求更少、更省)
MAX_ITEM_CHARS = 1200   # 单条上限: 超过就切片(中英日通用, 按字符)
MAX_OUTPUT_TOKENS = 8192   # 单次请求输出上限
BISECT_DEPTH = 4        # 二分重试的最大深度

_SENT_END = "。！？!?；;…．.\n）)】」』”\"'’、，,：:"

KANA_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _norm_cmp(s: str) -> str:
    return _ECHO_STRIP.sub("", s or "")


def is_echo(value: str, source: str) -> bool:
    """判断"译文"是否只是把原文照抄回来了。

    只对"较长且含假名"的日文句子判定 —— 人名、『第一章』、URL、版权行这类
    本来就该原样保留的短文本不能被误判为照抄。
    """
    v, s = _norm_cmp(value), _norm_cmp(source)
    if not v or not s:
        return False
    if len(s) < 8 or len(KANA_RE.findall(s)) < 3:
        return False                      # 短文本 / 无假名: 不判为照抄
    if v == s:
        return True
    if len(KANA_RE.findall(v)) == 0:
        return False                      # 译文已是汉字(正常翻译) -> 不是照抄
    if abs(len(v) - len(s)) > max(2, int(len(s) * 0.25)):
        return False
    from difflib import SequenceMatcher
    return SequenceMatcher(None, v, s).ratio() >= 0.9


def text_key(text: str, target: str = "", version: str = CACHE_VERSION) -> str:
    return hashlib.md5(f"{version}|{target}|{text}".encode("utf-8")).hexdigest()


def cache_value_usable(value: str, target: str = "") -> bool:
    """宽松校验(用于新拿到的译文): 只拒绝明显的拒答与原样返回。

    中文译文里保留日文专名(如「オラーシャ」)是正常现象, 不能因此判为失败;
    但"完全没有中文、只有假名"说明模型把原文原样退回了, 必须拒绝。
    """
    v = (value or "").strip()
    if not v:
        return False
    low = v.lower()
    if any(m in low for m in REFUSAL_MARKERS):
        return False
    tgt = (target or "").lower()
    if ("中文" in target) or ("chinese" in tgt) or (tgt == "zh"):
        kana = len(KANA_RE.findall(v))
        cjk = len(CJK_RE.findall(v))
        if kana and not cjk:
            return False          # 纯日文/假名 -> 原样返回
    return True


def cache_value_trustworthy(value: str, target: str = "") -> bool:
    """严格校验(用于判断历史缓存条目是否被"原文"污染)。

    假名占比超过 10% 即认为存的是日文原文(旧版把兜底原文写进了缓存)。
    """
    if not cache_value_usable(value, target):
        return False
    v = (value or "").strip()
    tgt = (target or "").lower()
    if ("中文" in target) or ("chinese" in tgt) or (tgt == "zh"):
        letters = max(len(KANA_RE.findall(v)) + len(CJK_RE.findall(v)), 1)
        if len(KANA_RE.findall(v)) / letters > 0.10:
            return False
    return True


class BudgetError(VisionError):
    """预算护栏触发(必须中止, 任何重试逻辑都不得吞掉)。"""


class TextSession(ChatHTTP):
    """纯文本翻译会话(自带预算护栏)。"""

    def __init__(self, *args, cache: Optional[Dict[str, str]] = None,
                 on_cache: Optional[Callable[[str, str], None]] = None,
                 on_batch: Optional[Callable[[int, int, str], None]] = None,
                 max_requests: int = 0, max_output_tokens: int = 0,
                 stall_limit: int = 25, thinking: str = "disabled",
                 workers: int = 1, transport_fail_limit: int = 5, **kwargs):
        kwargs.setdefault("retry_on_timeout", False)   # 超时重试会重复计费
        kwargs.setdefault("max_output_tokens_per_request", MAX_OUTPUT_TOKENS)
        super().__init__(*args, **kwargs)
        self.cache: Dict[str, str] = cache if cache is not None else {}
        self.on_cache = on_cache
        self.on_batch = on_batch
        self.thinking = thinking          # 默认关闭思考: 推理 token 既贵又慢
        self.workers = max(1, int(workers or 1))
        self.json_mode_failed = False   # 服务端不支持 JSON 输出时置位
        self._lock = threading.Lock()

        # ---- 护栏 ----
        self.max_requests = int(max_requests or 0)          # 0 = 不限制
        self.max_output_tokens = int(max_output_tokens or 0)  # 0 = 不限制
        self.stall_limit = int(stall_limit or 0)            # 连续无进展请求上限
        self.output_tokens = 0           # 累计补全 token
        self.translated_units = 0        # 成功翻译的条目/片数
        self.cached_units = 0            # 命中缓存(重复句)而无需请求的段数
        self.rejected_cache = 0          # 被判定不可信而忽略的缓存条目数
        self.total_pieces = 0            # 本次待处理的片段总数(skipped 的分母)
        self.failures: List[Dict[str, str]] = []   # 最终未译的单元(源文/答复/原因)
        self._rejected: Dict[Tuple[int, int], Tuple[str, str]] = {}
        self.skipped_units = 0           # 放弃翻译(保留原文)的片数
        self._stall = 0
        self.transport_failures = 0      # 连续传输失败次数
        self.transport_fail_limit = int(transport_fail_limit or 0)
        self.abort_reason = ""

    # ------------------------------------------------------------------ #
    # 预算护栏
    # ------------------------------------------------------------------ #
    def _guard(self) -> None:
        with self._lock:
            if self.max_requests and self.requests >= self.max_requests:
                self.abort_reason = (f"已达到 --max-requests 上限({self.max_requests} 次请求), "
                                     f"为保护余额主动中止。已完成的译文已保存。")
                raise BudgetError(self.abort_reason)
            if self.max_output_tokens and self.output_tokens >= self.max_output_tokens:
                self.abort_reason = (f"已达到 --max-output-tokens 上限"
                                     f"({self.max_output_tokens} tokens), 为保护余额主动中止。")
                raise BudgetError(self.abort_reason)
            if self.stall_limit and self._stall >= self.stall_limit:
                self.abort_reason = (f"连续 {self.stall_limit} 次请求没有产生任何新译文, "
                                     f"疑似服务异常或内容无法处理, 已主动中止以免持续计费。")
                raise BudgetError(self.abort_reason)

    def _call(self, prompt: str, response_format: Optional[dict],
              temperature: float = 0.3, thinking: Optional[str] = None) -> Tuple[str, str]:
        """发一次请求, 返回 (内容, finish_reason), 并维护计数器与进展统计。"""
        self._guard()
        raw = self.chat([{"role": "user", "content": prompt}],
                        temperature=temperature, response_format=response_format,
                        thinking=thinking if thinking is not None else self.thinking)
        usage = self.usage_of(raw)
        with self._lock:
            self.output_tokens += usage.get("completion_tokens", 0)
        return self.content_of(raw), self.finish_reason_of(raw)

    def _note_progress(self, filled: int) -> None:
        with self._lock:
            if filled > 0:
                self.translated_units += filled
                self._stall = 0
                self.transport_failures = 0
            else:
                self._stall += 1

    # ------------------------------------------------------------------ #
    def translate_segments(self, segments: Sequence[Tuple[str, str]],
                           target: str) -> List[str]:
        """segments: [(text, context)] -> [译文](与输入等长同序)。"""
        final: List[Optional[str]] = [None] * len(segments)
        pieces: List[Tuple[int, int, str, str]] = []
        piece_out: Dict[Tuple[int, int], str] = {}
        for i, (text, ctx) in enumerate(segments):
            if not text.strip():
                final[i] = text
                continue
            hit = self.cache.get(text_key(text, target))
            if hit is not None and (not cache_value_usable(hit, target)
                                    or is_echo(hit, text)):
                # 缓存里存的是"照抄原文/拒答": 视为没有缓存, 重新翻译
                with self._lock:
                    self.rejected_cache += 1
                hit = None
            if hit is not None:
                final[i] = hit
                with self._lock:
                    self.cached_units += 1
                    self.total_pieces += 1
                continue
            plist = split_long(text)
            with self._lock:
                self.total_pieces += len(plist)
            for k, piece in enumerate(plist):
                pieces.append((i, k, piece, ctx))

        batches = list(make_batches(pieces))
        total_batches = len(batches)
        self._rejected = {}          # 记录本轮被判定不可用的答复(供失败报告使用)
        self._run_batches(batches, target, piece_out)

        joins: Dict[int, List[str]] = {}
        fully_translated: Dict[int, bool] = {}
        for gid, k, text, _ctx in pieces:
            got = piece_out.get((gid, k))
            ok = got is not None
            if not ok:
                self.skipped_units += 1
                ans, why = self._rejected.get((gid, k), ("", "未译"))
                self._record_failure(text, ans, why)
                got = text                       # 兜底: 译不出就保留原文, 不丢内容
            joins.setdefault(gid, []).append(got)
            fully_translated[gid] = fully_translated.get(gid, True) and ok
        for i, parts in joins.items():
            joined = "\n".join(p.strip("\n") for p in parts) if len(parts) > 1 else parts[0]
            final[i] = joined
            # 只有"整段都由模型译出且不是照抄原文"才写缓存
            src = segments[i][0]
            if fully_translated.get(i) and cache_value_usable(joined, target) \
                    and not is_echo(joined, src):
                self._remember(src, joined, target)
        return [r if r is not None else "" for r in final]

    def _record_failure(self, source: str, answer: str, why: str) -> None:
        with self._lock:
            if len(self.failures) < 500:
                self.failures.append({"source": source[:300],
                                      "answer": (answer or "")[:300], "why": why})

    # ------------------------------------------------------------------ #
    def _run_batches(self, batches, target: str, out: Dict[Tuple[int, int], str]) -> None:
        """顺序或并发执行批次; 并发时护栏异常立即中止。"""
        total = len(batches)

        def notify(done: int) -> None:
            if self.on_batch:
                try:
                    self.on_batch(done, total,
                                  f"已译 {self.translated_units} 段 / 发 {self.requests} 次请求")
                except Exception:  # noqa: BLE001
                    pass

        if self.workers <= 1 or total <= 1:
            for idx, batch in enumerate(batches, start=1):
                self._translate_batch(batch, target, out)
                notify(idx)
            return

        pool = ThreadPoolExecutor(max_workers=self.workers)
        try:
            futures = [pool.submit(self._translate_batch, b, target, out)
                       for b in batches]
            done = 0
            for fut in as_completed(futures):
                done += 1
                exc = fut.exception()
                if exc is not None:
                    if isinstance(exc, BudgetError):
                        pool.shutdown(wait=False, cancel_futures=True)
                        raise exc
                    log.warning("批次处理异常(已跳过): %s", exc)
                notify(done)
        finally:
            pool.shutdown(wait=True)

    # ------------------------------------------------------------------ #
    def _translate_batch(self, batch: List[Tuple[int, int, str, str]], target: str,
                         out: Dict[Tuple[int, int], str], depth: int = 0) -> None:
        items = [{"id": f"{gid}:{k}", "text": text, "context": ctx[:400]}
                 for gid, k, text, ctx in batch]
        prompt = TEXT_TRANSLATE_PROMPT.format(
            target=target, payload=json.dumps({"items": items}, ensure_ascii=False))
        try:
            content, finish = self._call(
                prompt,
                None if self.json_mode_failed else {"type": "json_object"})
        except VisionError as exc:
            if isinstance(exc, BudgetError):
                raise                      # 护栏: 必须中止, 不得被任何重试逻辑吞掉
            if isinstance(exc, AuthError):
                # 密钥无效 / 余额不足: 继续跑下去只会把整本书留成原文, 立即中止
                self.abort_reason = (f"API 拒绝请求({exc})。可能是密钥无效或余额不足, "
                                     f"已立即中止。已完成部分已保存。")
                raise BudgetError(self.abort_reason) from exc
            if not self.json_mode_failed and _looks_like_json_unsupported(exc):
                log.warning("服务端不支持 JSON 输出, 改用普通模式重试")
                self.json_mode_failed = True
                return self._translate_batch(batch, target, out, depth)
            if isinstance(exc, TransportError):
                # 传输层失败(429/5xx/网络/超时): 拆小重试只会成倍放大请求, 直接放弃本批
                with self._lock:
                    self.transport_failures += 1
                    n = self.transport_failures
                log.warning("传输层失败(连续第 %d 次), 本批 %d 条保留原文: %s",
                            n, len(batch), exc)
                self.skipped_units += len(batch)
                if self.transport_fail_limit and n >= self.transport_fail_limit:
                    self.abort_reason = (
                        f"连续 {n} 次请求失败(最近一次: {exc})。可能是余额不足、限流或网络异常, "
                        f"已中止以免把整本书留成原文。已完成部分已保存。")
                    raise BudgetError(self.abort_reason) from exc
                return
            return self._retry_smaller(batch, target, out, depth, str(exc))

        if finish == "length":
            # 输出被截断 -> 请求太大, 需要缩小; 绝不当作失败反复重试
            log.warning("响应被 max_tokens 截断, 拆分后重试(条目数 %d)", len(batch))
            return self._retry_smaller(batch, target, out, depth, "finish_reason=length")

        by_key, ordered = _translation_mapping(parse_json_loose(content))
        missing: List[Tuple[int, int, str, str]] = []
        filled = 0
        for j, (gid, k, text, ctx) in enumerate(batch):
            got = by_key.get(f"{gid}:{k}")
            if got is None and j < len(ordered):     # 兜底: 按返回顺序对齐
                got = ordered[j]
            reason = ""
            if got is None or not str(got).strip():
                reason = "返回缺失或为空"
            elif not cache_value_usable(str(got), target):
                reason = "疑似模型拒答"
            elif is_echo(str(got), text):
                reason = "照抄原文未翻译"
            if reason:
                self._rejected[(gid, k)] = (str(got or ""), reason)
                missing.append((gid, k, text, ctx))
            elif (gid, k) not in out:
                out[(gid, k)] = str(got)
                filled += 1
        self._note_progress(filled)
        if missing:
            self._retry_smaller(missing, target, out, depth, "部分条目缺失/不可解析/疑似未译")

    def _retry_smaller(self, batch, target: str, out: Dict[Tuple[int, int], str],
                       depth: int, why: str) -> None:
        """失败/截断时的统一处理: 二分拆小重试; 到叶子仍失败则单条重试一次后放弃。"""
        if len(batch) > 1 and depth < BISECT_DEPTH:
            mid = len(batch) // 2
            log.warning("%s, 二分重试: %d 条 -> %d + %d", why, len(batch), mid,
                        len(batch) - mid)
            self._translate_batch(batch[:mid], target, out, depth + 1)
            self._translate_batch(batch[mid:], target, out, depth + 1)
            return
        if len(batch) > 1:
            # 已到最大二分深度: 逐条单次尝试(不再各自重试, 避免请求放大)
            for item in batch:
                self._translate_single(item, target, out)
            return
        if batch:
            self._translate_single(batch[0], target, out)

    def _translate_single(self, item, target: str,
                          out: Dict[Tuple[int, int], str]) -> None:
        """单条重试: 先按当前设置试一次; 若被判为拒答/照抄, 再用"打开思考模式"试一次。

        只对失败的少数条目追加这一次请求, 有界且便宜; 思考模式能让模型更愿意
        正经翻译"道歉对白""致谢长句"这类它容易照抄或敷衍的内容。
        """
        gid, k, text, _ctx = item
        filled = 0
        attempts = [None]
        if self.thinking != "enabled":
            attempts.append("enabled")
        for th in attempts:
            if (gid, k) in out:
                break
            try:
                prompt = TEXT_TRANSLATE_RETRY_PROMPT.format(target=target, text=text)
                content, finish = self._call(prompt, None, thinking=th)
                got = _extract_single(content) if finish != "length" else ""
                reason = ""
                if not got:
                    reason = "返回为空" if finish != "length" else "输出被截断"
                elif not cache_value_usable(got, target):
                    reason = "疑似模型拒答"
                elif is_echo(got, text):
                    reason = "照抄原文未翻译"
                if not reason:
                    out[(gid, k)] = got
                    filled = 1
                    break
                self._rejected[(gid, k)] = (got, reason)
                log.warning("单条翻译被判定为不可用(%s, 思考模式=%s): %s",
                            reason, th or self.thinking, text[:40])
            except TransportError as exc:
                log.warning("单条翻译传输失败(保留原文): %s", exc)
                break
            except BudgetError:
                raise                      # 护栏异常必须向上传播
            except VisionError as exc:
                log.warning("单条翻译失败(保留原文): %s", exc)
                break
        self._note_progress(filled)

    def _remember(self, text: str, translated: str, target: str) -> None:
        if not cache_value_usable(translated, target) or is_echo(translated, text):
            return                       # 兜底原文/拒答/照抄 绝不入缓存
        key = text_key(text, target)
        self.cache[key] = translated
        if self.on_cache:
            try:
                self.on_cache(key, translated)
            except Exception:  # noqa: BLE001  缓存写盘失败不影响主流程
                pass


# --------------------------------------------------------------------------- #
# 切片与分批
# --------------------------------------------------------------------------- #
def split_long(text: str, limit: int = MAX_ITEM_CHARS) -> List[str]:
    """把长文本切成 <= limit 字符的片。

    关键点: 中日文没有空格, 因此按**字符**切, 在窗口内尽量选句末标点/空白处断开;
    保证任何语言的输入都会被切片(旧版按空格切会让整章变成一条请求)。
    """
    if len(text) <= limit:
        return [text]
    pieces: List[str] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + limit, n)
        if end < n:
            floor = max(start + limit // 2, start + 1)
            for i in range(end, floor, -1):
                ch = text[i - 1]
                if ch in _SENT_END or ch.isspace():
                    end = i
                    break
        pieces.append(text[start:end])
        start = end
    return pieces


def make_batches(todo: List[Tuple[int, int, str, str]]):
    batch: List[Tuple[int, int, str, str]] = []
    chars = 0
    for item in todo:
        size = len(item[2])
        if batch and (chars + size > BATCH_CHARS or len(batch) >= BATCH_ITEMS):
            yield batch
            batch, chars = [], 0
        batch.append(item)
        chars += size
    if batch:
        yield batch


def _extract_single(content: str) -> str:
    """单条翻译结果清洗: 别把模型返回的 JSON 当译文写进书里。"""
    t = (content or "").strip()
    if not t:
        return ""
    if t.startswith("{") or t.startswith("["):
        obj = parse_json_loose(t)
        if isinstance(obj, dict):
            _by_key, ordered = _translation_mapping(obj)
            if ordered:
                return ordered[0].strip()
        return ""        # JSON 形状但没有可用译文 -> 视为失败(宁可保留原文)
    return t


def _translation_mapping(obj: Optional[dict]) -> Tuple[Dict[str, str], List[str]]:
    """解析 {"translations":[{"id":"0:1","text":"..."}]} 或 {"translations":["..."]}。

    返回 (按 id 索引, 按顺序列表)。
    """
    by_key: Dict[str, str] = {}
    ordered: List[str] = []
    if not isinstance(obj, dict):
        return by_key, ordered
    arr = obj.get("translations")
    if not isinstance(arr, list):
        return by_key, ordered
    for i, item in enumerate(arr):
        if isinstance(item, dict):
            key = str(item.get("id", i))
            val = item.get("text", item.get("translation", ""))
        else:
            key, val = str(i), item
        if isinstance(val, str):
            by_key[key] = val
            ordered.append(val)
    return by_key, ordered


def _looks_like_json_unsupported(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "response_format" in msg or ("json" in msg and "support" in msg)
