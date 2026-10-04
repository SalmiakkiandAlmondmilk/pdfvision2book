# -*- coding: utf-8 -*-
"""本地 mock DeepSeek API(OpenAI 兼容), 同时模拟视觉模型与纯文本模型。

  * 视觉请求: content 为数组且含 image_url -> 返回"转写 + [[FIG:n]] 占位符 +
    <<<FIGURES>>> JSON"(插图数量取自提示词里的候选区域提示);
  * 文本请求: content 为字符串(或数组里没有图片) -> 若含 items 则按 JSON 返回译文,
    否则返回单段译文。

校验: 模型名、base64 图片、请求结构。
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_COUNT = {"vision": 0, "text": 0}
MODELS: list = []          # 所有请求用到的模型名
THINKING: list = []        # 每个请求携带的 thinking.type(未携带为 None)
CAND_RE = re.compile(r"检测到 (\d+) 个可能是插图的区域")
KANA_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")


def fake_translate(text: str) -> str:
    """模拟一条"中文译文": 保留标记与汉字、去掉假名(避免被判为"照抄原文")。"""
    return "【译】" + KANA_RE.sub("*", text)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 静默
        pass

    def _ok(self, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(length))
        model = req.get("model", "")
        MODELS.append(model)
        THINKING.append((req.get("thinking") or {}).get("type"))
        content = req["messages"][0]["content"]

        if isinstance(content, list):
            images = [c for c in content if c.get("type") == "image_url"]
            if images:
                return self._vision(req, model, content, images)
            prompt = "\n".join(c.get("text", "") for c in content)
        else:
            prompt = content
        return self._text(req, model, prompt)

    # ------------------------------------------------------------------ #
    def _vision(self, req, model, content, images):
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        prompt = "\n".join(texts)
        assert model == "deepseek-v4-flash-vision-exp", f"视觉模型名不对: {model}"
        assert len(images) == 1, f"应恰好 1 张图片, 实际 {len(images)}"
        url = images[0]["image_url"]["url"]
        assert url.startswith("data:image/"), f"image_url 不是 data URL: {url[:40]}"
        assert len(url) > 100, "base64 图片数据过短"

        _COUNT["vision"] += 1
        i = _COUNT["vision"]
        m = CAND_RE.search(prompt)
        cand = int(m.group(1)) if m else 0
        n_figs = min(cand, 2)
        translated = "翻译要求" in prompt
        combined = "【原文】" in prompt and "完成转写后" in prompt

        figs = [{"n": k, "pos": "", "type": "chart",
                 "caption": f"模拟插图 {k}",
                 "text": f"图内文字-{k}"}
                for k in range(1, n_figs + 1)]
        markers = "".join(f"\n\n[[FIG:{k}]]\n\n" for k in range(1, n_figs + 1))
        body_orig = (f"# Mock Heading {i}\n\n"
                     f"Transcribed paragraph {i} with numbers 12345 and @#$.{markers}")
        body_tr = f"第 {i} 段模拟译文, 含数字 12345。{markers}"

        if translated and combined:
            answer = f"【原文】\n{body_orig}\n【译文】\n{body_tr}\n"
            for f in figs:
                f["caption_t"] = f["caption"] + "(译)"
                f["text_t"] = f["text"] + "(译)"
        elif translated:
            answer = body_tr + "\n"
            for f in figs:
                f["caption_t"] = f["caption"] + "(译)"
                f["text_t"] = f["text"] + "(译)"
        else:
            answer = body_orig + "\n"
        answer += f"\n<<<FIGURES>>>\n{json.dumps({'figures': figs}, ensure_ascii=False)}"

        return self._ok(self._resp(model, answer, prompt_tokens=120 + i,
                                   completion_tokens=60 + i))

    # ------------------------------------------------------------------ #
    def _text(self, req, model, prompt):
        assert model == "deepseek-v4-flash", f"文本模型名不对: {model}"
        _COUNT["text"] += 1
        i = _COUNT["text"]
        items = _extract_items(prompt)
        if items is not None:
            payload = {"translations": [
                {"id": it.get("id"), "text": fake_translate(str(it.get("text", "")))}
                for it in items]}
            answer = json.dumps(payload, ensure_ascii=False)
        else:
            m = re.search(r"原文:\n(.*)$", prompt, re.S)
            src = m.group(1).strip() if m else prompt[:200]
            answer = fake_translate(src)
        return self._ok(self._resp(model, answer, prompt_tokens=50 + i,
                                   completion_tokens=30 + i))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resp(model, answer, prompt_tokens, completion_tokens):
        return {
            "id": f"chatcmpl-mock-{model}",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": answer},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens,
                      # 真实响应会包含嵌套 details 对象, 客户端必须能容忍
                      "prompt_tokens_details": {"cached_tokens": 5},
                      "completion_tokens_details": {"reasoning_tokens": 3}},
        }


def _extract_items(prompt: str):
    """从提示词里取出 items 数组。"""
    start = prompt.find('{"items"')
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(prompt)):
        if prompt[i] == "{":
            depth += 1
        elif prompt[i] == "}":
            depth -= 1
            if depth == 0:
                blob = prompt[start:i + 1]
                try:
                    return json.loads(blob).get("items", [])
                except (json.JSONDecodeError, TypeError, ValueError):
                    return None
    return None


def start(port: int = 8765):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv


def reset():
    _COUNT["vision"] = 0
    _COUNT["text"] = 0
    MODELS.clear()
    THINKING.clear()


if __name__ == "__main__":
    import sys
    srv = start(int(sys.argv[1]) if len(sys.argv) > 1 else 8765)
    print(f"mock api 运行于 http://127.0.0.1:{srv.server_address[1]} (Ctrl+C 退出)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
