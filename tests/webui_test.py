# -*- coding: utf-8 -*-
"""网页上传模式的端到端测试: multipart 上传 -> 轮询状态 -> 下载 TXT/EPUB/ZIP。

用法: python tests/webui_test.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdfvision.config import Settings  # noqa: E402
from pdfvision.webui import WebHandler, _Server  # noqa: E402
from tests import fixtures  # noqa: E402
from tests.stub_api import start as start_stub  # noqa: E402


def _wait(web: str, jid: str, timeout: int = 120) -> dict:
    """轮询作业状态直到完成/出错。"""
    deadline = time.time() + timeout
    status: dict = {}
    while time.time() < deadline:
        _st, resp = http_json(f"{web}/status?id={jid}")
        status = json.loads(resp)
        if status.get("status") in ("done", "error"):
            return status
        time.sleep(0.4)
    return status

FAILS: list = []


def check(name: str, cond: bool, detail: str = ""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  -> {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def _multipart(fields: dict, files: dict) -> tuple:
    boundary = "----pdfvisionWebTestBoundary7MA4YWxkTrZu0gW"
    parts: list = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n")
    for k, (fname, ctype, data) in files.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; "
                     f"filename=\"{fname}\"\r\nContent-Type: {ctype}\r\n\r\n".encode("utf-8"))
        parts.append(data)
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(p if isinstance(p, bytes) else p.encode("utf-8") for p in parts)
    ctype = f"multipart/form-data; boundary={boundary}"
    return body, ctype


def http_json(url: str, data: bytes = None, ctype: str = None, timeout: int = 120):
    headers = {}
    if data is not None:
        headers["Content-Type"] = ctype
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def main() -> int:
    if not (ROOT / "tests" / "sample.pdf").exists():
        from tests.make_sample_pdf import build as b
        b(ROOT / "tests" / "sample.pdf")
    stub = start_stub(8766)
    stub_url = f"http://127.0.0.1:{stub.server_address[1]}"

    base_settings = Settings(api_key="test-key", base_url=stub_url)
    srv = _Server(("127.0.0.1", 0), WebHandler, base_settings)
    import threading
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    web = f"http://127.0.0.1:{srv.server_address[1]}"
    print(f"webui 测试服务器: {web}")

    # 1) 首页
    st, html = http_json(web + "/")
    check("首页可访问", st == 200 and b"pdfvision2book" in html)

    pdf_bytes = (ROOT / "tests" / "sample.pdf").read_bytes()

    # 2) 缺少 API Key 时给出 400 提示(服务器不配置 key)
    srv2_settings = Settings(api_key="", base_url=stub_url)
    srv.settings = srv2_settings
    body, ctype = _multipart(
        {"mode": "page"},
        {"file": ("sample.pdf", "application/pdf", pdf_bytes)})
    st, resp = http_json(web + "/upload", body, ctype)
    check("无 Key 上传被拒绝并提示", st == 400 and b"API Key" in resp, resp[:200])
    srv.settings = base_settings

    # 3) 正常上传
    body, ctype = _multipart(
        {"mode": "page", "translate": "简体中文", "with_original": "on",
         "workers": "1", "title": "网页端到端测试"},
        {"file": ("sample.pdf", "application/pdf", pdf_bytes)})
    st, resp = http_json(web + "/upload", body, ctype, timeout=60)
    job = json.loads(resp)
    check("上传返回作业 id", st == 200 and "id" in job, resp[:200])
    jid = job["id"]

    deadline = time.time() + 120
    status = None
    while time.time() < deadline:
        st, resp = http_json(f"{web}/status?id={jid}")
        status = json.loads(resp)
        if status.get("status") in ("done", "error"):
            break
        time.sleep(0.5)
    check("作业最终完成", status and status.get("status") == "done",
          json.dumps(status, ensure_ascii=False)[:300])

    # 4) 下载 txt / epub / zip
    st, txt = http_json(f"{web}{status['result']['txt']}")
    txt_dec = txt.decode("utf-8", "replace")
    check("下载 TXT 成功", st == 200 and "网页端到端测试" in txt_dec)
    st, epub = http_json(f"{web}{status['result']['epub']}")
    check("下载 EPUB 成功", st == 200 and epub[:2] == b"PK")
    st, zdata = http_json(f"{web}/download-all?id={jid}")
    if st == 200:
        names = zipfile.ZipFile(__import__("io").BytesIO(zdata)).namelist()
        check("ZIP 含 txt/epub/meta", "网页端到端测试.txt" in names
              and "网页端到端测试.epub" in names and "meta.json" in names,
              str(names))
    else:
        check("下载 ZIP 成功", False, f"status={st}")

    # 5) 图片上传(视觉通道)
    png_path = Path(tempfile.gettempdir()) / "pdfvision_web_hello.png"
    fixtures.build_text_png(png_path, "WEB IMAGE 999")
    body, ctype = _multipart({"mode": "page", "workers": "1", "title": "网页图片测试"},
                             {"file": ("hello.png", "image/png", png_path.read_bytes())})
    st, resp = http_json(web + "/upload", body, ctype, timeout=60)
    jid_img = json.loads(resp).get("id")
    status_img = _wait(web, jid_img)
    check("图片上传处理完成", status_img.get("status") == "done",
          json.dumps(status_img, ensure_ascii=False)[:200])

    # 6) EPUB 上传(文本通道, 仅翻译, 图片原位保留)
    epub_path = Path(tempfile.gettempdir()) / "pdfvision_web_book.epub"
    fixtures.build_simple_epub(epub_path)
    body, ctype = _multipart({"translate": "简体中文", "workers": "1", "title": "网页EPUB测试"},
                             {"file": ("book.epub", "application/epub+zip",
                                       epub_path.read_bytes())})
    st, resp = http_json(web + "/upload", body, ctype, timeout=60)
    jid_epub = json.loads(resp).get("id")
    status_epub = _wait(web, jid_epub)
    check("EPUB 上传处理完成", status_epub.get("status") == "done",
          json.dumps(status_epub, ensure_ascii=False)[:200])
    if status_epub.get("status") == "done":
        st, edata = http_json(f"{web}{status_epub['result']['epub']}")
        with zipfile.ZipFile(__import__("io").BytesIO(edata)) as zf:
            names6 = zf.namelist()
            body6 = "".join(zf.read(n).decode("utf-8", "replace")
                            for n in names6 if n.endswith((".xhtml", ".xml")))
        check("EPUB 输出保留图片", "OEBPS/images/pic.png" in names6)
        check("EPUB 输出已翻译", "【译】" in body6)

    # 7) 动图拒绝
    gif_path = Path(tempfile.gettempdir()) / "pdfvision_web_anim.gif"
    fixtures.build_animated_gif(gif_path)
    body, ctype = _multipart({"workers": "1"},
                             {"file": ("anim.gif", "image/gif", gif_path.read_bytes())})
    st, resp = http_json(web + "/upload", body, ctype, timeout=60)
    jid_gif = json.loads(resp).get("id")
    status_gif = _wait(web, jid_gif)
    check("动图被拒绝并提示", status_gif.get("status") == "error"
          and "动图" in status_gif.get("error", ""),
          json.dumps(status_gif, ensure_ascii=False)[:200])

    srv.shutdown()
    stub.shutdown()
    print("\n" + ("网页模式检查全部通过 ✅" if not FAILS else f"有 {len(FAILS)} 项失败 ❌"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
