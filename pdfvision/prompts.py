"""给视觉模型 / 纯文本模型的提示词(中文指令, 便于用户读懂与修改)。"""

MARKER_ORIGINAL = "原文"
MARKER_TRANSLATION = "译文"
FIGURES_SEP = "<<<FIGURES>>>"
PROMPT_VERSION = "v3-figures"      # 提示词版本(参与断点续跑的键, 改动后旧断点不复用)

_VISION_BASE = """你是一个专业的文档阅读助手。下面给你一张来自 PDF / 图片的页面图像。

任务一 · 转写: 逐字、完整地转写页面上出现的全部文字, 原文是什么语言就输出什么语言,
不省略、不总结、不增删内容; 保持阅读顺序(多栏版面先左栏后右栏), 按段落自然分段;
能辨认的标题、列表、表格用 Markdown 表达(标题用 # / ##, 列表用 - , 表格用 Markdown 表格)。

任务二 · 插图标记: 页面中出现的插图(照片、图表、示意图、地图、漫画、装饰画等非正文图形),
请在其**出现的位置**单独占一行插入占位符 [[FIG:n]], n 从 1 开始,
按自上而下、先左后右的顺序编号。纯图片页/整页照片也要标记。
注意: 表格不算插图, 用 Markdown 表格表达即可, 不要标记 [[FIG:]]。

任务三 · 插图信息: 全部转写内容写完后, 另起一行输出 {sep},
紧跟着输出一个合法 JSON 对象(不要用代码块包裹、不要有多余文字):
{{"figures":[{{"n":1,"type":"photo|chart|diagram|map|comic|other","caption":"一句话客观描述","text":"图中出现的全部文字, 逐字转写, 没有则为空字符串"{t_fields}}}]}}
页面没有插图时输出 {sep} 后跟 {{"figures":[]}}。

其他要求: 不要编造图中不存在的文字; 空白页只输出 [空白页/无内容];
不要输出任何解释性前言。"""

_T_FIELDS = ""","caption_t":"caption 的译文","text_t":"text 的译文"""

_TRANSLATE_ONLY_TAIL = """
翻译要求: 请阅读整页内容, 把其中全部正文文字忠实翻译成「{target}」。
- 翻译准确通顺, 保留标题层级、列表、表格结构; 专有名词首次出现可附原文。
- [[FIG:n]] 占位符必须原样保留在译文的对应位置。
- 只输出译文正文(含占位符), 不要输出【原文】小节。
- 上面 {sep} 之后的 JSON 仍需输出, 并在其中给出 caption_t / text_t 译文字段。"""

_TRANSLATE_BOTH_TAIL = """
翻译要求: 完成转写后, 把转写出的全部正文文字忠实翻译成「{target}」, 规则同上。
输出必须严格按下面顺序:
【原文】
(完整逐字转写内容, 含 [[FIG:n]] 占位符)
【译文】
(完整译文, 含 [[FIG:n]] 占位符)
{sep}
(插图 JSON, 同时包含原文字段与 caption_t / text_t 译文字段)"""


def build_vision_prompt(translate: bool, with_original: bool, target: str = "",
                        candidate_count: int = 0) -> str:
    """构造视觉模型提示词。translate=False => 仅转写原文。"""
    t_fields = _T_FIELDS if translate else ""
    prompt = _VISION_BASE.format(sep=FIGURES_SEP, t_fields=t_fields)
    if candidate_count > 0:
        prompt += (f"\n\n提示: 版面分析在该页检测到 {candidate_count} 个可能是插图的区域, "
                   f"请重点核对, 并按实际内容标记 (最终以你的判断为准)。")
    if not translate:
        return prompt
    target = target or "目标语言"
    tail = _TRANSLATE_BOTH_TAIL if with_original else _TRANSLATE_ONLY_TAIL
    return prompt + "\n" + tail.format(target=target, sep=FIGURES_SEP)


def candidate_hint_count(prompt: str) -> int:
    """测试/调试用: 从提示词里读回候选插图数量。"""
    import re
    m = re.search(r"检测到 (\d+) 个可能是插图的区域", prompt)
    return int(m.group(1)) if m else 0


# --------------------------------------------------------------------------- #
# 纯文本模型(EPUB / TXT 输入, 仅翻译)
# --------------------------------------------------------------------------- #
TEXT_TRANSLATE_PROMPT = """你是专业的图书翻译。下面是一个 JSON 对象, items 里每条包含:
  id      —— 条目编号
  text    —— 需要翻译的原文
  context —— 该条目所在段落的完整上下文(仅供理解, 不要翻译它)

请把每条 text 翻译成「{target}」, 并遵守:
1. 逐条翻译, 输出条目数量、顺序、id 必须与输入完全一致, 不得合并或拆分条目。
2. 保持原文的 Markdown 标记、换行、以及 [[FIG:n]] 之类的占位符原样不变。
3. 人名、数字、单位、URL、代码、专有名词要准确; 术语前后一致。
4. 若某条 text 本身已是目标语言, 原样返回。
5. 只输出 JSON: {{"translations":[{{"id":0,"text":"译文"}}, ...]}}, 不要任何解释或代码块。

输入:
{payload}"""

TEXT_TRANSLATE_RETRY_PROMPT = """把下面这段文字翻译成「{target}」。
只输出译文本身, 不要任何说明; 保持 Markdown 标记与 [[FIG:n]] 占位符不变。

原文:
{text}"""
