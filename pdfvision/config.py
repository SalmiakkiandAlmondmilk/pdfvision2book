"""配置与 .env / 环境变量加载。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# 项目里所有默认值集中在此, 便于维护
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_VISION_MODEL = "deepseek-v4-flash-vision-exp"   # 视觉: 读图/转写/插图识别
DEFAULT_TEXT_MODEL = "deepseek-v4-flash"                # 纯文本: EPUB/TXT 仅翻译

# 支持的文件类型
VISUAL_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
               ".gif", ".jp2", ".j2k", ".pnm", ".pgm", ".ppm"}
TEXT_EXTS = {".epub", ".txt"}
SUPPORTED_EXTS = VISUAL_EXTS | TEXT_EXTS


def load_dotenv(dotenv_path: Optional[str] = None) -> None:
    """极简 .env 解析: KEY=VALUE, # 注释, 不覆盖已存在的环境变量。"""
    path = Path(dotenv_path) if dotenv_path else (Path.cwd() / ".env")
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class Settings:
    """一次运行的完整配置。"""

    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    vision_model: str = DEFAULT_VISION_MODEL   # 读图模型(PDF/图片输入)
    text_model: str = DEFAULT_TEXT_MODEL       # 纯文本模型(EPUB/TXT 输入, 仅翻译)
    timeout: int = 180            # 单次 HTTP 请求超时(秒)
    max_attempts: int = 6         # 失败重试次数上限(429/5xx/网络错误)

    pdf_path: Optional[str] = None   # 输入文件(PDF/图片/EPUB/TXT 均可)
    outdir: Optional[str] = None     # None => 在输入文件同目录生成 <stem>_vision 文件夹

    mode: str = "page"            # PDF: "page" 整页渲染 | "images" 仅提取嵌入图片
    dpi: int = 150                # 页面渲染分辨率
    max_pages: Optional[int] = None  # 只处理前 N 页(调试/试跑用)
    save_pages: bool = False      # 把送入模型的图片保存到 outdir/pages 便于核对

    keep_figures: bool = True     # 识别插图并在 EPUB 中原位嵌入裁切出的图片
    figure_dpi: int = 200         # 插图裁切分辨率
    figure_max_side: int = 1600   # 插图裁切后单边最大像素

    translate: Optional[str] = None  # 目标语言, 例如 "简体中文"/"English"; None=仅转写原文
    with_original: bool = False   # 翻译时是否保留原文对照
    default_text_target: str = "简体中文"  # EPUB/TXT 输入未指定 --translate 时的默认目标语言
    pages_per_chapter: int = 8    # 每章包含的页数(用于 EPUB 分章); 0 => 每页一章
    workers: int = 2              # 并发请求数(请留意账号速率限制)
    formats: List[str] = field(default_factory=lambda: ["txt", "epub"])  # 输出格式
    resume: bool = True           # 断点续跑: 复用 checkpoint 中已完成结果, 不重复扣费

    # ---- 预算护栏(防止异常情况下持续计费) ----
    max_requests: int = 0         # 本次运行最多请求次数; 0 = 不限制
    max_output_tokens: int = 0    # 累计补全 token 上限; 0 = 不限制
    stall_limit: int = 25         # 连续多少次请求没有任何新译文就中止
    transport_fail_limit: int = 5  # 连续多少次请求失败(网络/限流/余额)就中止, 避免默默留原文
    max_output_tokens_per_request: int = 8192   # 单次请求输出上限(防止单次生成失控)
    dry_run: bool = False         # 只预估不调用 API

    # ---- 思考模式(默认开启, 推理内容按输出 token 计费且拖慢速度) ----
    text_thinking: str = "disabled"   # 文本翻译: disabled / enabled / auto
    vision_thinking: str = "auto"     # 读图: auto(服务端默认) / disabled / enabled
    strip_ruby: bool = True           # 翻译时去掉日语注音 <rt>/<rp>(译文不需要假名)

    title: str = ""               # 输出书名; 空 => 用文件名

    extra: dict = field(default_factory=dict)

    # ---------- 派生属性 ----------
    @property
    def source_path(self) -> Optional[str]:
        return self.pdf_path

    @property
    def effective_outdir(self) -> Path:
        if self.outdir:
            base = Path(self.outdir)
        elif self.pdf_path:
            base = Path(self.pdf_path).resolve().parent / (Path(self.pdf_path).stem + "_vision")
        else:
            base = Path.cwd() / "vision_output"
        base.mkdir(parents=True, exist_ok=True)
        return base

    @property
    def document_title(self) -> str:
        if self.title:
            return self.title
        if self.pdf_path:
            return Path(self.pdf_path).stem
        return "转译文档"

    @property
    def target_language_tag(self) -> str:
        """EPUB dc:language 用 IETF 标签。"""
        if not self.translate:
            return "zh-CN"
        t = self.translate.lower()
        for zh in ("中文", "汉语", "简体中文", "中文简体", "zh", "zh-cn", "chinese"):
            if zh in t:
                return "zh-CN"
        if t.startswith("en") or "english" in t:
            return "en"
        if t.startswith("ja") or "japanese" in t:
            return "ja"
        if t.startswith("ko") or "korean" in t:
            return "ko"
        if t.startswith("fr") or "french" in t:
            return "fr"
        if t.startswith("de") or "german" in t:
            return "de"
        return "zh-CN"


def default_settings() -> Settings:
    load_dotenv()
    s = Settings()
    s.api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    s.base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    s.vision_model = (os.environ.get("DEEPSEEK_VISION_MODEL")
                      or os.environ.get("DEEPSEEK_MODEL")
                      or DEFAULT_VISION_MODEL).strip() or DEFAULT_VISION_MODEL
    s.text_model = (os.environ.get("DEEPSEEK_TEXT_MODEL")
                    or DEFAULT_TEXT_MODEL).strip() or DEFAULT_TEXT_MODEL
    try:
        s.timeout = int(os.environ.get("DEEPSEEK_TIMEOUT", s.timeout))
    except ValueError:
        pass
    return s
