# 📖 pdfvision2book

基于 **DeepSeek V4 Flash** 双模型（视觉 `deepseek-v4-flash-vision-exp` + 纯文本 `deepseek-v4-flash`）
的**转译程序** —— 独立程序，不是插件。

| 输入 | 走哪个模型 | 做什么 | 输出 |
|---|---|---|---|
| **PDF** | 视觉模型 | 逐页阅读：识别文字（可选翻译）＋识别插图并裁切嵌入 | TXT / EPUB3 |
| **图片**（jpg/png/webp/bmp/tif/gif 静态图，**动图除外**） | 视觉模型 | 识别图中文字（可选翻译） | TXT / EPUB3 |
| **EPUB** | 纯文本模型 | **仅翻译**；图片、行内结构**原位保留** | 翻译后的 EPUB / TXT |
| **TXT** | 纯文本模型 | **仅翻译** | TXT / EPUB3 |

> 与「从 PDF 抽文字层」的传统工具不同：本程序走**视觉理解**通道，
> 扫描件、图文混排、文字层不可用都没关系。
> 输出文件里**不含**任何「第 N 页」「第 N 页 · 图片 N」「第 1–8 页」之类的页/图标记，正文连排到底。

---

## 1. 特性

| 能力 | 说明 |
|---|---|
| 自动识别输入类型 | 按文件头 + 扩展名判断 PDF / 图片 / EPUB / TXT，分别走视觉或文本通道 |
| 插图识别与嵌入 | 视觉模型用 `[[FIG:n]]` 标注插图位置并给出九宫格方位与图中文字；程序按方位与几何版面区域（嵌入位图、矢量绘图簇）配对，裁切后在 EPUB 原位置嵌入 `<figure>` |
| 插图开关 | `--keep-figures`（默认开）/ `--no-figures`；整页扫描图不会被当作插图重复裁切 |
| 转写 + 翻译 | PDF/图片：默认仅转写原文，`--translate 简体中文` 出译文，`--with-original` 原文对照 |
| EPUB 图片原位保留 | EPUB 输入只替换文字节点，其余资源（图片/字体/CSS/OPF）原样拷贝，图片引用路径不变 |
| 双格式输出 | `.txt` + `.epub`（EPUB3，含目录；纯标准库打包） |
| 断点续跑 | 每个单元结果即时落盘 checkpoint，中断后重跑同一条命令只补未完成部分，不重复扣费 |
| 结果不丢失 | 输出目录不可写或被中途删除时自动重建/改存临时目录 |
| **费用护栏** | 单次输出上限、超时不重试、`--max-requests` / `--max-output-tokens` 硬上限、无进展看门狗、`--dry-run` 跑前预估 |
| 双入口 | 命令行 + 本地网页上传界面（纯标准库，无 Web 框架依赖） |

---

## 1.1 费用护栏（重要，建议先读）

长文本翻译曾出现一次严重事故：日文 EPUB 的**整章几万字没有空格**，旧版按空格切片导致整章作为一次请求发出；
失败后又对每条逐条重试（各 6 次退避），请求放大 8 倍，连续 5 小时、每小时 200+ 请求，直到余额耗尽才停止。
现在针对性做了六层防护：

| 防护 | 说明 |
|---|---|
| 按字符切片 | 中日文没有空格也一定切成 ≤1200 字的片（句末标点优先断开），绝不整章发送 |
| 单次输出上限 | 每次请求带 `max_tokens`（默认 8192），模型无法生成失控长文 |
| 截断二分重试 | 输出被截断（`finish_reason=length`）时把请求二分缩小，而不是反复重试同一个大请求 |
| 传输失败不放大 | 429/5xx/网络错误重试耗尽后直接放弃该批（保留原文），不再逐条重试 |
| 超时不重试 | 长输出超时往往已在计费，重试等于再花一次钱；默认不重试 |
| 硬预算 + 看门狗 | `--max-requests` / `--max-output-tokens` 达上限即中止；连续 25 次请求无新译文自动中止；**已完成的译文照常保存** |

**跑之前先预估（不花钱）**：

```bash
python main.py 日文书.epub --translate 简体中文 --dry-run
# 输出: 翻译单元/切片数/最长单元/预计请求数/输出 token 上限
```

确认规模后再正式跑，并按需加硬上限：

```bash
python main.py 日文书.epub --translate 简体中文 --max-requests 400 --max-output-tokens 200000
```

### 1.2 长文本翻译的三个隐性成本（实测）

以一本 2.4 MB 的日语竖排小说（24 个文档、`<ruby>` 注音）为例：

| 项目 | 修复前 | 修复后 |
|---|---|---|
| 翻译单元 | 14306 个碎片（汉字与注音被拆开，平均 7.6 字/片） | **3505 个完整句子**（同一 `<p>` 内合并） |
| 请求数 | 1803 次（失败时还会逐条重试放大） | **147 次** |
| 思考模式 | 默认开启（effort=high），推理 token 按输出计费、还拖慢速度 | **默认关闭**（`--text-thinking disabled`） |
| 并发 | 顺序发送，一次一个 | `--workers N` 并发（默认 2，建议 4） |

三点结论：

1. **句子被 `<ruby>`/`<span>` 切碎**会让请求数暴涨、译文质量下降 → 现在按块级元素合并成句，
   并把日语注音 `<rt>/<rp>` 去掉（`--keep-ruby` 可保留）；
2. **思考模式默认开启**：每个请求都会先生成一段推理链，`reasoning_content` 按输出 token 计费。
   翻译这种确定性任务不需要它，默认关闭即可省下大头（要质量优先可用 `--text-thinking enabled`）；
3. **顺序发送**在几百个请求时就是几十分钟 → 文本通道现在同样支持并发。

`tests/epub_report.py` 可以在跑之前看清一本书的结构，`tests/epub_e2e_mock.py` 可以用 mock API
把整本书离线跑一遍（不花钱），确认请求数与产出无误。

---

## 2. 安装与配置

```bash
cd pdfvision2book
pip install -r requirements.txt        # 只装一个: pymupdf
```

配置 API Key（任选其一）：

```bash
set DEEPSEEK_API_KEY=sk-xxxx           # Windows CMD
$env:DEEPSEEK_API_KEY="sk-xxxx"        # PowerShell
export DEEPSEEK_API_KEY=sk-xxxx        # Linux/macOS
# 或: 复制 .env.example 为 .env 并填写
```

可选覆盖（见 `.env.example`）：`DEEPSEEK_BASE_URL`、`DEEPSEEK_VISION_MODEL`、`DEEPSEEK_TEXT_MODEL`、`DEEPSEEK_TIMEOUT`。

---

## 3. 命令行用法

```bash
# —— PDF（视觉模型）——
python main.py 扫描版书.pdf                             # 转写原文 -> TXT + EPUB(含插图嵌入)
python main.py book.pdf --translate 简体中文             # 读出后翻成中文
python main.py book.pdf --translate 简体中文 --with-original   # 双语对照
python main.py book.pdf --no-figures                     # 不嵌入插图, 只保留图中文字
python main.py book.pdf --mode images                    # 只看 PDF 内嵌图片

# —— 图片（视觉模型）——
python main.py 插图页.png --translate 简体中文            # 识别图中文字并翻译
python main.py photo.jpg --format txt                    # 只输出 TXT

# —— EPUB / TXT（纯文本模型, 仅翻译）——
python main.py 英文书.epub --translate 简体中文           # 图片原位保留
python main.py 英文书.epub                               # 不填语言 -> 默认翻成简体中文
python main.py 笔记.txt --translate English              # 中文 -> 英文

# —— 通用 ——
python main.py book.pdf --format epub                    # 只输出 EPUB(txt/epub/both)
python main.py book.pdf --max-pages 3 --workers 3        # 试跑 3 页, 并发 3
python main.py book.pdf --no-resume                      # 不用断点, 全部重跑
```

常用参数（`python main.py --help` 查看全部）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--format txt\|epub\|both` | `both` | 输出格式 |
| `--translate 语言` | — | PDF/图片不填=仅转写；EPUB/TXT 不填=默认简体中文 |
| `--with-original` | 关 | 翻译时附原文对照（PDF/图片） |
| `--keep-figures` / `--no-figures` | 开 | 是否把识别到的插图裁切嵌入 EPUB |
| `--figure-dpi N` | 200 | 插图裁切分辨率 |
| `--mode page\|images` | `page` | PDF：整页渲染 / 仅内嵌图片 |
| `--dpi N` | 150 | 页面渲染分辨率（字小可调 200~300） |
| `--workers N` | 2 | 并发请求数 |
| `--pages-per-chapter N` | 8 | EPUB 内部分章粒度（目录只显示序号，正文无标记） |
| `--max-pages N` | 全本 | 只处理前 N 页 |
| `--no-resume` | 续跑开 | 已完成的页面也重新识别 |
| `--dry-run` | 关 | 只预估处理量/请求数/token 上限, 不调用 API |
| `--max-requests N` | 0 | 最多请求次数, 达到即中止(保护余额) |
| `--max-output-tokens N` | 0 | 累计输出 token 上限, 达到即中止 |
| `--stall-limit N` | 25 | 连续多少次请求无新译文就中止 |
| `--max-output-tokens-per-request N` | 8192 | 单次请求输出上限 |
| `--text-thinking` | `disabled` | 文本翻译思考模式（推理 token 既贵又慢，默认关闭） |
| `--vision-thinking` | `auto` | 读图请求思考模式（默认用服务端默认） |
| `--keep-ruby` | 关 | 保留日语注音 `<rt>`（默认翻译时去掉） |
| `--vision-model` / `--text-model` | 见上 | 分别指定读图模型与文本模型 |

输出默认在输入文件旁生成 `<文件名>_vision/`：

```
我的扫描书_vision/
├─ 我的扫描书.txt
├─ 我的扫描书.epub          # 插图以 <figure> 内嵌(OEBPS/figures/*.png)
├─ meta.json                # 参数与 token 用量
├─ .checkpoint.jsonl        # 断点续跑用(可删)
└─ pages/                   # 仅 --save-pages 时: 送入模型的页面图片
```

---

## 4. 本地网页上传模式

```bash
python main.py --serve            # 浏览器打开 http://127.0.0.1:8000
```

- 拖拽/选择文件：**PDF / 图片 / EPUB / TXT** 均可，类型自动识别；
- 选项：任务类型、输出格式（TXT / EPUB / 两者）、是否翻译、**是否嵌入插图**、并发数、API Key；
- 轮询进度，完成后下载 TXT / EPUB / 打包 ZIP；动图会明确报错拒绝；
- 局域网访问：`python main.py --serve --host 0.0.0.0 --port 任一可用端口`（注意安全）。

---

## 5. 工作原理

```
PDF / 图片 ──► [PyMuPDF] 页面渲染 / 内嵌图提取 / 动图检测 / 插图几何候选
    │                    (嵌入位图 + 矢量绘图簇, 过滤整页背景与过小区域)
    ▼
视觉模型 deepseek-v4-flash-vision-exp  (每张图 1 次请求, base64 内联)
    · 逐字转写(原语言, 保留标题/列表/表格)
    · 插图处插入 [[FIG:n]] 占位符
    · 末尾 <<<FIGURES>>> + JSON: 序号/九宫格方位/类型/描述/图中文字(/译文字段)
    ▼
按方位(九宫格) + 阅读顺序把模型插图与几何区域配对 ──► 裁切 PNG
    ▼
装配: 去标记正文 + <figure><img/> + 图中文字 ──► TXT / EPUB3

EPUB ──► 标签级分词, 只在文字片段上翻译(保留标签/图片引用/行内结构)
TXT  ──► 按段翻译(JSON 批量对齐, 逐条降级重试)
    ▼
纯文本模型 deepseek-v4-flash ──► 重打包 EPUB(图片原位) / TXT
```

要点：

- **插图配对**：模型给出九宫格方位（`top-left`…`bottom-right`），程序与几何候选区域按方位优先配对，
  剩余按阅读顺序配对；配不上的插图只保留图中文字（不硬塞错误的图）。
- **整页扫描图**不裁切（面积 > 72% 视为整页背景），避免把整页图片重复嵌进书里。
- **译文缓存可信性**：`.checkpoint_text.jsonl` 条目带版本标记，旧版写入的条目一律忽略
  （旧版缺陷：翻译失败时把原文当译文写入缓存，重跑时会"不调用 API 却输出日文"）；
  载入时还会丢弃纯假名、模型拒答之类的值。检查/清理工具：
  `python tests/check_cache.py <缓存文件> 简体中文 [--clean]`。
- **判定规则（两次真实事故后修正）**：
  - 对白里的「抱歉/对不起/sorry」是**正常译文**，不再被当成"模型拒答"误杀
    （旧版把这三个词写进了拒答黑名单，导致道歉对白永远译不出来）；
  - **照抄原文 = 未翻译**：译文与源文高度相同（长日文句 + 假名）时判为失败并重试，
    不再被静默当成"已翻译"留在书里；人名、『第一章』、URL、版权行等短文本不受影响；
  - 被判失败的条目会再用"打开思考模式"重试一次（只针对这些少数条目，成本有界）；
  - 仍未译出的片段写入输出目录的 **`failed_units.txt`**（原文 + 模型原始答复 + 原因），
    日志与 GUI 弹窗都会指向该文件。
- **失败必须可见**：请求连续失败（默认 5 次）或遇到密钥/余额错误会立即中止；
  结束时若有未译段落会给出告警与比例，不再静默产出"半本原文"。
- **断点续跑**：checkpoint 的键包含提示词版本、模式、DPI、目标语言、模型、插图开关与图片指纹；
  任一设置变化会自动重新识别，不会误用旧结果；`--no-resume` 可忽略全部缓存重新翻译。
- **结果保护**：写盘前重建输出目录；仍失败则改存系统临时目录并在输出里注明。

---

## 6. 自测（mock，离线、不花钱）

```bash
python tests/make_sample_pdf.py tests/sample.pdf   # 可选: 生成 3 页样例 PDF
python tests/smoke.py          # 11 个场景: PDF/图片/EPUB/TXT、插图嵌入、断点续跑、动图拒绝、
                               #           双模型分派、无页/图标记断言
python tests/regression_test.py # 事故回归: 中日文切片、429 不放大、截断二分、三重护栏、超时不重试、
                               #           dry-run、中止仍保存、日语注音合并/去注音/关思考
python tests/webui_test.py     # 网页上传全流程(含图片、EPUB 上传与动图拒绝)
python tests/cli_test.py       # CLI 参数映射(四种输入 × 格式/翻译/插图开关)
python tests/epub_report.py 某书.epub    # 诊断: 逐文档统计翻译单元/最长单元/预计请求数
python tests/epub_e2e_mock.py 某书.epub  # 用 mock 把整本书离线跑一遍(不花钱), 校验产出
```

### 插图内文字位置测验（真实模型，需 Key）

按九宫格 9 个位置分别生成"插图 + 图中文字"页面，检验模型能否读出框内文字、
是否输出 [[FIG:n]] 占位符、以及几何区域配对是否正确：

```bash
python tests/live_figure_positions.py                      # 9 个位置全测(9 次请求)
python tests/live_figure_positions.py --positions TL,MC,BR --min-accuracy 0.6
```

未配置 Key 时自动跳过（退出码 0）；识别率低于 `--min-accuracy`（默认 0.7）时退出码非 0。

---

## 7. 常见问题

- **EPUB/TXT 没写 `--translate` 会怎样？** 这两类输入只做翻译，默认目标语言为「简体中文」
  （可用 `--translate` 覆盖）。
- **EPUB 翻译很久/请求特别多怎么办？** 先用 `python main.py 书.epub --dry-run` 或
  `python tests/epub_report.py 书.epub` 看结构：如果某个文档有几千个"翻译单元"，说明它是
  逐句/带注音的排版；现在已按块合并并去掉 `<rt>`，请求数会大幅下降。可加 `--workers 4`
  并发，并用 `--max-requests` 设上限。
- **EPUB 翻译后图片还在吗？** 在。程序只替换文字节点，图片文件与引用路径原样保留，仅更新 `dc:language`。
- **插图没被嵌入？** 只有"模型判定为插图 + 几何上找到对应区域"才会裁切嵌入；
  表格不算插图；整页扫描图不裁切。可用 `--no-figures` 完全关闭。
- **动图报错**：GIF/WebP/APNG 多帧图片不支持，请另存为静态 PNG/JPG。
- **中断/误删输出目录？** 直接重跑同一条命令即可续跑；想全部重来加 `--no-resume`。
- **识别精度不够？** 提高 `--dpi`（200~300）；插图可提高 `--figure-dpi`。
- **EPUB 用什么打开？** 微信读书 / Apple Books / Calibre / Edge 等均支持 EPUB3。
- **安全提示**：文件内容会发送到 DeepSeek API，请勿处理敏感资料。

## 8. 目录结构

```
pdfvision2book/
├─ main.py                  # 命令行/网页入口
├─ requirements.txt         # 运行时依赖(仅 pymupdf)
├─ .env.example
├─ pdfvision/
│  ├─ config.py             # 配置、.env 加载、默认模型
│  ├─ sources.py            # 输入类型探测与分派
│  ├─ pdf_ingest.py         # 页面渲染/内嵌图/动图检测/插图候选与裁切(PyMuPDF)
│  ├─ prompts.py            # 视觉与文本模型提示词
│  ├─ http_client.py        # OpenAI 兼容客户端 + 重试
│  ├─ vision_client.py      # 视觉模型: 转写/插图 JSON 解析
│  ├─ text_client.py        # 纯文本模型: 批量翻译(JSON 对齐)
│  ├─ epub_io.py            # EPUB 读取/标签级翻译/重打包(图片原位)
│  ├─ bookbuilder.py        # Markdown/插图 -> TXT / EPUB3
│  ├─ pipeline.py           # 双通道编排、插图装配、断点续跑、结果保护
│  └─ webui.py              # 本地网页上传界面
└─ tests/                   # mock API + 离线冒烟 + 真实模型位置测验
```

> 免责声明：程序按 DeepSeek 官方 OpenAI 兼容接口实现（`/chat/completions`，
> base64 `image_url`；文本翻译走普通 messages + JSON 输出）。用量与计费以你的账号账单为准。
