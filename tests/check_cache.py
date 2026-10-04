# -*- coding: utf-8 -*-
"""诊断/清理: 检查翻译缓存(.checkpoint_text.jsonl)是否被"原文/拒答"污染。

用法:
    python tests/check_cache.py <checkpoint_text.jsonl> [目标语言]            # 只检查
    python tests/check_cache.py <checkpoint_text.jsonl> [目标语言] --clean    # 清理并重写

判定: 目标语言是中文时, 值里假名占比高、只有假名、或含拒答话术 => 视为污染条目。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdfvision.text_client import (CACHE_RECORD_VERSION, KANA_RE,  # noqa: E402
                                   cache_value_trustworthy)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_clean = "--clean" in sys.argv
    if not args:
        print("用法: python tests/check_cache.py <checkpoint_text.jsonl> [目标语言] [--clean]")
        return 2
    path = Path(args[0])
    target = args[1] if len(args) > 1 else "简体中文"
    raw_lines = [l for l in path.read_text(encoding="utf-8", errors="replace").splitlines()
                 if l.strip()]

    total = good = legacy = bad = 0
    bad_samples, good_samples = [], []
    keep_lines = []
    for line in raw_lines:
        try:
            obj = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        t = obj.get("t")
        if not isinstance(t, str):
            continue
        total += 1
        if obj.get("v") != CACHE_RECORD_VERSION:
            legacy += 1
            continue                      # 旧格式: 不保留(新版也不会使用)
        if cache_value_trustworthy(t, target):
            good += 1
            keep_lines.append(line)
            if len(good_samples) < 3:
                good_samples.append(t[:60])
        else:
            bad += 1
            if len(bad_samples) < 6:
                bad_samples.append((obj.get("k", "")[:8], t[:60]))

    print(f"文件: {path}")
    print(f"大小: {path.stat().st_size / 1024:.0f} KB · 目标语言: {target} · 格式版本 v={CACHE_RECORD_VERSION}")
    print(f"条目总数: {total}")
    print(f"  可信译文  : {good}")
    print(f"  污染/不可信: {bad}  (原文、纯假名、模型拒答)")
    print(f"  旧格式忽略 : {legacy}  (无版本标记, 新版不再使用)")
    if bad_samples:
        print("\n污染样本(这些值其实是原文/拒答):")
        for k, t in bad_samples:
            print(f"  key={k}… value={t!r}")
    if good_samples:
        print("\n正常样本:")
        for t in good_samples:
            print(f"  {t!r}")

    if do_clean:
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
        path.write_text("\n".join(keep_lines) + ("\n" if keep_lines else ""), encoding="utf-8")
        print(f"\n已清理: 保留 {len(keep_lines)} 条, 原文件备份为 {backup.name}")
    elif bad or legacy:
        print(f"\n提示: 新版程序已自动忽略这些条目(缓存键带版本号 + 载入时校验), "
              f"可加 --clean 直接清理文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
