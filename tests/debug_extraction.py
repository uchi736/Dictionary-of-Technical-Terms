#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import re
from collections import Counter

text_sample = """
アンモニア燃料エンジンの開発。
アンモニア濃度を測定。
GHG削減目標。
アンモニ内部構造。
エンジン周囲の温度。
"""

patterns = [
    r'[ァ-ヶー]+',  # カタカナ
    r'[一-龯]{2,}',  # 漢字（2文字以上）
    r'[A-Z][A-Za-z0-9]*',  # 英数字
    r'[ァ-ヶー]+[一-龯]+|[一-龯]+[ァ-ヶー]+',  # カタカナ+漢字
]

print("=" * 50)
print("正規表現パターンテスト")
print("=" * 50)

candidates = Counter()

for i, pattern in enumerate(patterns, 1):
    print(f"\nパターン {i}: {pattern}")
    matches = list(re.finditer(pattern, text_sample))
    for match in matches:
        term = match.group()
        if 2 <= len(term) <= 15:
            print(f"  - {term} (位置: {match.start()}-{match.end()})")
            candidates[term] += 1

print("\n" + "=" * 50)
print("抽出された全候補語:")
print("=" * 50)
for term, count in candidates.most_common():
    print(f"  {term}: {count}回")