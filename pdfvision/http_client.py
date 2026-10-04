"""OpenAI 兼容 Chat Completions 的公共 HTTP 客户端(标准库实现, 带重试)。"""
from __future__ import annotations

import json
import logging
import socket
import time
import urllib.error
import urllib.request
from typing import Dict, Optional

log = logging.getLogger("pdfvision")


class VisionError(RuntimeError):
    """可向用户展示的 API 错误(视觉/文本客户端共用)。"""


class TransportError(VisionError):
    """传输层失败: HTTP 错误(重试已用尽)或网络错误。"""


class RequestTimeout(TransportError):
    """请求超时(长输出可能已计费, 默认不重试)。"""


class AuthError(TransportError):
    """密钥无效 / 余额不足 / 无权限 —— 继续跑没有意义, 应立即中止。"""


def http_detail(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", "replace")
        parsed = json.loads(raw)
        return (parsed.get("error") or {}).get("message") or raw
    except Exception:  # noqa: BLE001
        return str(exc.reason or "")


class ChatHTTP:
    """极简 chat/completions 客户端: 429/5xx/网络错误指数退避重试。"""

    def __init__(self, api_key: str, base_url: str, model: str,
                 timeout: int = 180, max_attempts: int = 6,
                 max_output_tokens_per_request: int = 0,
                 retry_on_timeout: bool = True,
                 thinking: str = "auto"):
        if not api_key:
            raise VisionError(
                "缺少 DeepSeek API Key: 请设置环境变量 DEEPSEEK_API_KEY, "
                "或在 .env 中填写, 或用 --api-key 传入。")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_attempts = max_attempts
        # 单次请求的输出上限: 防止模型生成失控导致巨额计费(0 = 不发送该参数)
        self.max_output_tokens_per_request = max_output_tokens_per_request
        # 超时后是否重试: 长文本翻译重试会重复计费, 默认由调用方决定
        self.retry_on_timeout = retry_on_timeout
        # 思考模式: "auto" 表示不发送参数(服务端默认开启且 effort=high)
        self.thinking = thinking or "auto"
        self.total_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        self.requests = 0          # 实际发出的逻辑请求数

    @property
    def endpoint(self) -> str:
        return self.base_url + "/chat/completions"

    # ------------------------------------------------------------------ #
    def accumulate(self, usage) -> None:
        if isinstance(usage, dict):
            for k, v in usage.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    self.total_usage[k] = self.total_usage.get(k, 0) + int(v)

    def chat(self, messages, temperature: float = 0.2,
             response_format: Optional[dict] = None,
             max_tokens: Optional[int] = None,
             thinking: Optional[str] = None) -> dict:
        """发一次请求, 返回原始响应 dict。

        thinking: "disabled"/"enabled" 时显式开关思考模式;
                  None/"auto" 表示不发送该参数(用服务端默认, 默认是开启且 effort=high)。
        """
        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature}
        if response_format:
            payload["response_format"] = response_format
        mode = thinking if thinking is not None else self.thinking
        if mode in ("disabled", "enabled"):
            payload["thinking"] = {"type": mode}
        limit = max_tokens or self.max_output_tokens_per_request
        if limit and limit > 0:
            payload["max_tokens"] = int(limit)
        self.requests += 1
        raw = self.post(payload)
        self.accumulate(raw.get("usage") or {})
        return raw

    def post(self, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint, data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            })
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = resp.read()
                obj = json.loads(data.decode("utf-8"))
                if isinstance(obj, dict) and "choices" not in obj:
                    raise VisionError(f"服务端返回异常响应: {str(obj)[:400]}")
                return obj
            except urllib.error.HTTPError as exc:
                detail = http_detail(exc)
                low = detail.lower()
                if exc.code in (401, 402, 403) or "insufficient" in low or "balance" in low \
                        or "unauthorized" in low or "invalid api key" in low:
                    raise AuthError(f"HTTP {exc.code}: {detail}") from exc
                if exc.code in (429, 500, 502, 503, 504):
                    last_err = TransportError(f"HTTP {exc.code}: {detail}")
                    if attempt >= self.max_attempts:
                        break
                    wait = min(2 ** attempt, 30)
                    log.warning("请求失败 HTTP %s (%s), %.1fs 后重试 (%d/%d)",
                                exc.code, detail, wait, attempt, self.max_attempts)
                    time.sleep(wait)
                    continue
                raise TransportError(f"HTTP {exc.code}: {detail}") from exc
            except (TimeoutError, socket.timeout) as exc:
                # 超时往往意味着"长输出已经开始生成", 盲目重试会重复计费
                last_err = RequestTimeout(
                    f"请求超时({self.timeout}s): {exc}")
                if not self.retry_on_timeout:
                    raise RequestTimeout(
                        f"请求超时({self.timeout}s), 已按策略不重试以避免重复计费") from exc
                if attempt >= self.max_attempts:
                    break
                wait = min(2 ** attempt, 30)
                log.warning("请求超时, %.1fs 后重试 (%d/%d)", wait, attempt, self.max_attempts)
                time.sleep(wait)
            except (urllib.error.URLError, ConnectionError) as exc:
                last_err = TransportError(f"网络错误: {exc}")
                if attempt >= self.max_attempts:
                    break
                wait = min(2 ** attempt, 30)
                log.warning("网络错误(%s), %.1fs 后重试 (%d/%d)", exc, wait, attempt,
                            self.max_attempts)
                time.sleep(wait)
            except json.JSONDecodeError as exc:
                raise VisionError(f"响应不是合法 JSON: {exc}") from exc
        raise TransportError(f"多次重试后仍然失败: {last_err}")

    @staticmethod
    def content_of(raw: dict) -> str:
        try:
            return (raw["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise VisionError(f"响应结构异常: {raw}") from exc

    @staticmethod
    def finish_reason_of(raw: dict) -> str:
        """'stop' / 'length' / ''(未知)。length 表示输出被上限截断。"""
        try:
            return str(raw["choices"][0].get("finish_reason") or "")
        except (KeyError, IndexError, TypeError, AttributeError):
            return ""

    @staticmethod
    def usage_of(raw: dict) -> Dict[str, int]:
        out: Dict[str, int] = {}
        usage = raw.get("usage") if isinstance(raw, dict) else None
        if isinstance(usage, dict):
            for k, v in usage.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[k] = int(v)
        return out
