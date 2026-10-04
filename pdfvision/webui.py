"""本地网页上传界面(纯标准库 ThreadingHTTPServer)。

用法: python main.py --serve
打开 http://127.0.0.1:8000 上传 PDF / 图片 / EPUB / TXT,
处理完成后下载 TXT 与 EPUB。
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import tempfile
import threading
import time
import urllib.parse
import zipfile
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional

from .config import SUPPORTED_EXTS, Settings
from .pipeline import process_input
from .vision_client import VisionError

log = logging.getLogger("pdfvision.webui")

_JOBS: Dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_JOB_ID_RE = re.compile(r"^[a-z0-9]{8,20}$")
_MAX_JOBS = 60


@dataclass
class _State:
    id: str = ""
    status: str = "running"          # running | done | error
    progress: tuple = (0, 0, "")
    error: str = ""
    summary: dict = field(default_factory=dict)
    outdir: str = ""
    title: str = ""
    created: float = field(default_factory=time.time)


def _progress_cb(state: _State):
    def cb(done: int, total: int, msg: str):
        state.progress = (done, total, msg)
    return cb


def _launch(state: _State, settings: Settings):
    try:
        summary = process_input(settings, progress=_progress_cb(state))
        state.summary = summary
        state.outdir = summary["outdir"]
        state.status = "done"
    except Exception as exc:  # noqa: BLE001
        state.status = "error"
        state.error = str(exc)


class WebHandler(BaseHTTPRequestHandler):
    server_version = "pdfvision2book/1.0"

    # ------------------------------------------------------------------ #
    def log_message(self, fmt, *args):  # 静默访问日志
        log.debug(fmt % args)

    # ------------------------------------------------------------------ #
    def _send(self, code: int, body: bytes, ctype: str, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj: dict, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    # ------------------------------------------------------------------ #
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        if path in ("/", "/index.html"):
            return self._send(200, PAGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
        if path == "/status":
            jid = query.get("id", [""])[0]
            return self._send_json(self._status(jid))
        if path == "/download":
            return self._download(query, single=True)
        if path == "/download-all":
            return self._download(query, single=False)
        if path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        return self._send(404, b"not found", "text/plain")

    # ------------------------------------------------------------------ #
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/upload":
            return self._send(404, b"not found", "text/plain")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        body = self.rfile.read(length)
        fields, files = _parse_multipart(body, self.headers.get("Content-Type"))
        return self._accept_upload(fields, files)

    # ------------------------------------------------------------------ #
    def _accept_upload(self, fields: Dict[str, str], files: Dict[str, tuple]):
        if not files:
            return self._send_json({"error": "没有收到文件"}, 400)
        _field, (fname, data) = next(iter(files.items()))
        fname = fname or "upload.pdf"
        if not data:
            return self._send_json({"error": "文件内容为空"}, 400)

        api_key = (fields.get("api_key") or "").strip()
        base = self.server.settings  # type: ignore[attr-defined]
        if not api_key and not base.api_key:
            return self._send_json(
                {"error": "未配置 API Key: 请在本页填写, 或设置环境变量 "
                          "DEEPSEEK_API_KEY / .env"}, 400)

        workdir = Path(tempfile.mkdtemp(prefix="pdfvision_run_"))
        ext = (Path(fname).suffix or "").lower()
        if ext not in SUPPORTED_EXTS:
            return self._send_json(
                {"error": f"不支持的文件类型: {ext or fname}。"
                          "支持 PDF / 图片(jpg png webp bmp tif gif, 动图除外) / EPUB / TXT"}, 400)
        pdf_path = workdir / ("upload" + (ext or ".pdf"))
        pdf_path.write_bytes(data)

        title = (fields.get("title") or "").strip()
        mode = fields.get("mode") or "page"
        if mode not in ("page", "images"):
            mode = "page"
        translate = (fields.get("translate") or "").strip() or None
        with_original = fields.get("with_original") == "on" or bool(fields.get("with_original"))
        save_pages = fields.get("save_pages") == "on"
        keep_figures = fields.get("keep_figures", "on") == "on"
        try:
            max_requests = max(int(fields.get("max_requests") or "0"), 0)
            max_output_tokens = max(int(fields.get("max_output_tokens") or "0"), 0)
        except ValueError:
            max_requests, max_output_tokens = 0, 0
        fmt = (fields.get("format") or "both").strip()
        formats = {"txt": ["txt"], "epub": ["epub"],
                   "both": ["txt", "epub"]}.get(fmt, ["txt", "epub"])
        try:
            workers = min(int(fields.get("workers") or "2"), 8)
            dpi = min(int(fields.get("dpi") or "150"), 400)
        except ValueError:
            workers, dpi = 2, 150

        settings = Settings(
            api_key=api_key or base.api_key,
            base_url=base.base_url,
            vision_model=base.vision_model,
            text_model=base.text_model,
            timeout=base.timeout,
            pdf_path=str(pdf_path),
            outdir=str(workdir / "out"),
            mode=mode, dpi=dpi, save_pages=save_pages,
            translate=translate, with_original=with_original,
            workers=workers, title=title or Path(fname).stem,
            formats=formats, keep_figures=keep_figures,
            max_requests=max_requests, max_output_tokens=max_output_tokens,
        )
        jid = _new_id()
        state = _State(id=jid, outdir=settings.outdir, title=settings.document_title)
        with _JOBS_LOCK:
            if len(_JOBS) >= _MAX_JOBS:
                # 淘汰最早完成的作业
                for old in sorted(_JOBS.values(),
                                  key=lambda s: s.created)[:len(_JOBS) - _MAX_JOBS + 1]:
                    _JOBS.pop(old.id, None)
            _JOBS[jid] = state
        threading.Thread(target=_launch, args=(state, settings), daemon=True).start()
        return self._send_json({"id": jid})

    # ------------------------------------------------------------------ #
    def _status(self, jid: str) -> dict:
        state = _JOBS.get(jid)
        if not state:
            return {"error": "作业不存在或已过期"}
        d, t, msg = state.progress
        out = {"id": state.id, "status": state.status,
               "progress": {"done": d, "total": t, "msg": msg}}
        if state.error:
            out["error"] = state.error
        if state.status == "done":
            out["title"] = state.title
            result = {}
            for kind, label in (("txt", "下载 TXT"), ("epub", "下载 EPUB")):
                if state.summary.get(kind):
                    result[kind] = f"/download?id={jid}&kind={kind}"
            result["meta"] = f"/download?id={jid}&kind=meta"
            result["zip"] = f"/download-all?id={jid}"
            out["result"] = result
            out["usage"] = state.summary.get("usage", {})
            out["seconds"] = state.summary.get("seconds", 0)
        return out

    # ------------------------------------------------------------------ #
    def _download(self, query, single: bool):
        jid = query.get("id", [""])[0]
        state = _JOBS.get(jid)
        if not state or state.status != "done":
            return self._send_json({"error": "作业不存在或尚未完成"}, 404)
        outdir = Path(state.summary["outdir"])
        if single:
            kind = query.get("kind", [""])[0]
            fname = Path(state.summary.get(kind, "")).name
            if kind not in ("txt", "epub", "meta") or not fname:
                return self._send_json({"error": "未知文件"}, 400)
            p = outdir / fname
            if not p.exists():
                return self._send_json({"error": "文件不存在"}, 404)
            data = p.read_bytes()
            ctype = {"txt": "text/plain; charset=utf-8",
                     "epub": "application/epub+zip",
                     "meta": "application/json; charset=utf-8"}.get(kind, "application/octet-stream")
            return self._send(200, data, ctype,
                              {"Content-Disposition": _content_disposition(fname)})
        # zip
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for key in ("txt", "epub", "meta"):
                name = Path(state.summary.get(key, "") or "").name
                if not name:
                    continue
                p = outdir / name
                if p.exists() and p.is_file():
                    zf.writestr(name, p.read_bytes())
            pages_dir = outdir / "pages"
            if pages_dir.exists():
                for p in sorted(pages_dir.iterdir()):
                    if p.is_file():
                        zf.writestr(f"pages/{p.name}", p.read_bytes())
        data = buf.getvalue()
        fname = (re.sub(r'[\\/:*?"<>|]', "_", state.title) or "output") + ".zip"
        return self._send(200, data, "application/zip",
                          {"Content-Disposition": _content_disposition(fname)})


# --------------------------------------------------------------------------- #
# multipart 解析(email 标准库)
# --------------------------------------------------------------------------- #
def _parse_multipart(body: bytes, content_type: Optional[str]):
    fields: Dict[str, str] = {}
    files: Dict[str, tuple] = {}      # 字段名 -> (文件名, 字节)
    if not content_type or "multipart/form-data" not in content_type:
        return fields, files
    headers = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n"
    msg = BytesParser(policy=policy.default).parsebytes(
        headers.encode("latin-1") + body)
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        if filename:
            files[name] = (filename, payload)
        else:
            fields[name] = payload.decode("utf-8", "replace")
    return fields, files


def _content_disposition(fname: str) -> str:
    """兼容中文文件名的 Content-Disposition(ASCII 兜底 + RFC5987 filename*)。"""
    ascii_name = re.sub(r"[^\x20-\x7e]", "_", fname).strip() or "output"
    enc = urllib.parse.quote(fname)
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{enc}"


def _new_id() -> str:
    import hashlib
    import secrets
    return hashlib.sha1(
        (secrets.token_hex(8) + str(time.time_ns())).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 页面
# --------------------------------------------------------------------------- #
PAGE_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pdfvision2book · PDF 图片阅读转译</title>
<style>
:root{--c:#2563eb;--bg:#f6f8fb;}
*{box-sizing:border-box}
body{font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;
     background:var(--bg);margin:0;color:#1e293b}
.wrap{max-width:720px;margin:0 auto;padding:32px 16px 80px}
h1{font-size:26px;margin:8px 0 2px}
.sub{color:#64748b;font-size:14px;margin-bottom:22px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:14px;
      padding:20px 22px;margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.05)}
label{display:block;font-weight:600;font-size:13.5px;margin:14px 0 6px}
input[type=text],input[type=password],select{width:100%;padding:9px 11px;
      border:1px solid #cbd5e1;border-radius:8px;font-size:14px;background:#fff}
.row{display:flex;gap:14px}.row>div{flex:1}
.opt{display:flex;align-items:center;gap:6px;font-weight:400;font-size:14px;margin-top:8px}
.drop{border:2px dashed #94a3b8;border-radius:12px;padding:34px 16px;text-align:center;
      color:#475569;cursor:pointer;transition:.15s;background:#fafcff}
.drop.hover{border-color:var(--c);background:#eff6ff;color:var(--c)}
.drop b{color:var(--c)}
#fname{margin-top:10px;font-size:13px;color:#16a34a;word-break:break-all}
button{background:var(--c);border:0;color:#fff;font-size:15px;font-weight:600;
       padding:12px 18px;border-radius:9px;cursor:pointer;width:100%;margin-top:18px}
button:disabled{opacity:.55;cursor:not-allowed}
#status{display:none;margin-top:16px;font-size:14px;color:#334155}
#bar{height:8px;background:#e2e8f0;border-radius:6px;overflow:hidden;margin-top:8px}
#bar>i{display:block;height:100%;width:0;background:var(--c);transition:width .3s}
#result{display:none;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;
        padding:16px 18px;margin-top:16px}
#result a{margin-right:14px;font-weight:600;color:var(--c)}
#err{display:none;background:#fef2f2;border:1px solid #fecaca;color:#b91c1c;
     border-radius:12px;padding:12px 16px;margin-top:16px;white-space:pre-wrap}
.note{font-size:12.5px;color:#94a3b8;margin-top:8px}
a.foot{color:#64748b;font-size:12.5px}
</style></head><body><div class="wrap">
<h1>📖 pdfvision2book</h1>
<div class="sub">基于 <b>DeepSeek V4 Flash</b> 视觉 / 文本双模型：<br>
  PDF / 图片 → 识别文字（可选翻译）＋识别插图并裁切嵌入 EPUB；
  EPUB / TXT → 仅翻译，EPUB 图片原位保留。<br>
  输出 <b>TXT</b> 与 <b>EPUB</b>，全部处理在本机完成。</div>

<div class="card">
  <form id="form" enctype="multipart/form-data">
    <div class="drop" id="drop">
      点击选择或拖拽文件到此处<br>
      <span class="note" style="color:#94a3b8">支持 PDF、图片（jpg/png/webp/bmp/tif/gif，动图除外）、EPUB、TXT</span>
    </div>
    <input type="file" id="file" name="file"
           accept=".pdf,.png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff,.gif,.epub,.txt" hidden>
    <div id="fname"></div>

    <label>任务类型（仅 PDF / 图片输入时生效）</label>
    <select name="mode">
      <option value="page">整页渲染阅读（推荐，适合扫描件与图文混排）</option>
      <option value="images">仅阅读 PDF 内嵌图片</option>
    </select>

    <label>输出格式</label>
    <select name="format">
      <option value="both" selected>TXT + EPUB（默认）</option>
      <option value="epub">仅 EPUB</option>
      <option value="txt">仅 TXT</option>
    </select>

    <label>是否翻译（PDF/图片不填 = 仅转写原文；EPUB/TXT 不填 = 翻成简体中文）</label>
    <input type="text" name="translate" list="langs" placeholder="例如：简体中文 / English / 日本語 / 繁体中文 …">
    <datalist id="langs">
      <option value="简体中文"><option value="English"><option value="日本語">
      <option value="한국어"><option value="繁体中文"><option value="Français">
    </datalist>

    <div class="row">
      <div><label>书名（可选）</label><input type="text" name="title" placeholder="默认用 PDF 文件名"></div>
      <div><label>API Key（可选，留空用环境变量）</label>
           <input type="password" name="api_key" autocomplete="off"></div>
    </div>

    <label class="opt"><input type="checkbox" name="keep_figures" checked> 识别插图并在 EPUB 中原位嵌入裁切出的图片（默认开）</label>
    <label class="opt"><input type="checkbox" name="with_original"> 翻译时同时保留原文对照</label>

    <div class="row">
      <div><label>费用护栏：最多请求次数（0=不限）</label>
           <input type="text" name="max_requests" placeholder="例如 500"></div>
      <div><label>费用护栏：累计输出 token 上限（0=不限）</label>
           <input type="text" name="max_output_tokens" placeholder="例如 300000"></div>
    </div>
    <div class="note">已内置护栏：单次请求输出上限 8192 tokens；超时不重试；连续 25 次请求无新译文自动中止；完成部分会照常保存。</div>
    <label class="opt"><input type="checkbox" name="save_pages"> 保存送入模型的页面图片（便于核对）</label>
    <label class="opt" style="margin-bottom:6px">
      并发数 <select name="workers" style="width:80px;display:inline-block;margin:0 0 0 6px">
        <option>1</option><option selected>2</option><option>3</option><option>4</option>
      </select></label>

    <button id="go" type="submit">开始处理</button>
  </form>

  <div id="status">
    <div id="msg">准备中…</div>
    <div id="bar"><i></i></div>
  </div>
  <div id="err"></div>
  <div id="result"></div>
  <div class="note">提示：也可以直接用命令行 <code>python main.py 文件.pdf [--translate 简体中文]</code>，不依赖本页面。</div>
</div>

<p class="foot">处理期间请勿关闭页面；完成后会自动出现下载链接。</p>
</div>

<script>
const drop=document.getElementById('drop'),fileEl=document.getElementById('file'),
      fname=document.getElementById('fname');
drop.onclick=()=>fileEl.click();
drop.ondragover=e=>{e.preventDefault();drop.classList.add('hover');};
drop.ondragleave=()=>drop.classList.remove('hover');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('hover');
  if(e.dataTransfer.files.length)fileEl.files=e.dataTransfer.files;};
fileEl.onchange=()=>{fname.textContent=fileEl.files[0]?('已选择：'+fileEl.files[0].name):'';};

const form=document.getElementById('form'),go=document.getElementById('go'),
      statusBox=document.getElementById('status'),msgEl=document.getElementById('msg'),
      barEl=document.querySelector('#bar>i'),errEl=document.getElementById('err'),
      resEl=document.getElementById('result');
let timer=null;

form.onsubmit=async e=>{
  e.preventDefault();
  if(!fileEl.files[0]){err('请先选择 PDF 文件');return;}
  errEl.style.display='none';resEl.style.display='none';
  statusBox.style.display='block';go.disabled=true;barEl.style.width='0%';
  msgEl.textContent='上传中…';
  const fd=new FormData(form);
  try{
    const r=await fetch('/upload',{method:'POST',body:fd});
    const j=await r.json();
    if(j.error)throw new Error(j.error);
    poll(j.id);
  }catch(ex){fail(ex.message);}
};

function poll(id){
  clearTimeout(timer);
  fetch('/status?id='+id).then(r=>r.json()).then(j=>{
    if(j.error){fail(j.error);return;}
    const p=j.progress||{};
    msgEl.textContent=(p.done||0)+' / '+(p.total||'?')+'  '+ (p.msg||'');
    const w=p.total?Math.round(p.done/p.total*100):0;
    barEl.style.width=(w||3)+'%';
    if(j.status==='done'){done(j);return;}
    if(j.status==='error'){fail(j.error||'处理失败');return;}
    timer=setTimeout(()=>poll(id),1200);
  }).catch(()=>{timer=setTimeout(()=>poll(id),2000);});
}
function done(j){
  statusBox.style.display='none';go.disabled=false;
  const r=j.result;
  const labels={txt:'⬇ 下载 TXT',epub:'⬇ 下载 EPUB'};
  let links='';
  ['txt','epub'].forEach(k=>{if(r[k])links+='<a href="'+r[k]+'">'+labels[k]+'</a>';});
  links+='<a href="'+r.zip+'">⬇ 打包下载 ZIP</a>';
  resEl.style.display='block';
  resEl.innerHTML='<b style="font-size:15px">✔ 处理完成：'+esc(j.title||'文档')+'</b>'+
    '<div style="margin-top:10px">'+links+'</div>'+
    '<div class="note">耗时 '+j.seconds+' 秒 · Token：'+
      ((j.usage&&j.usage.total_tokens)||0)+'（提示 '+
      ((j.usage&&j.usage.prompt_tokens)||0)+' + 补全 '+
      ((j.usage&&j.usage.completion_tokens)||0)+'）</div>';
}
function fail(m){statusBox.style.display='none';go.disabled=false;
  errEl.style.display='block';errEl.textContent='出错了：'+m;}
function err(m){fail(m);}
function esc(s){return s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
</script></body></html>
"""


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #
class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler, settings: Settings):
        super().__init__(addr, handler)
        self.settings = settings


def serve(settings: Settings, host: str = "127.0.0.1", port: int = 8000) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        httpd = _Server((host, port), WebHandler, settings)
    except OSError as exc:
        print(f"[错误] 无法监听 {host}:{port} -> {exc}", file=__import__("sys").stderr)
        return 1
    url = f"http://{host}:{port}"
    print("=" * 60)
    print(" pdfvision2book 网页模式已启动")
    print(f"   请用浏览器打开: {url}")
    if settings.api_key:
        print("   API Key: 已从环境变量/.env 读取(也可在页面另行填写)")
    else:
        print("   API Key: 未配置, 请在页面上的输入框填写")
    print("   按 Ctrl+C 退出")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出。")
    finally:
        httpd.server_close()
    return 0
