"""数据清洗与特征构造模块。

处理步骤：
1. 数值字段类型转换与缺失剔除
2. 重复挂牌去重
3. 楼层、户型、梯户比、朝向等字段解析
4. 单价/面积/总价的极端值过滤
5. 构造分析所需的派生特征并落盘
"""

import re

import pandas as pd

from .config import PROCESSED_DATA_PATH

FLOOR_RE = re.compile(r"(低|中|高)楼层.*?共(\d+)层")
SIZE_RE = re.compile(r"(\d+)室(\d+)厅(\d+)厨(\d+)卫")
LADDER_RE = re.compile(r"([一二两三四五六七八九十]+)梯([一二两三四五六七八九十]+)户")

_CN_DIGITS = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}

AREA_BINS = [0, 60, 90, 120, 150, float("inf")]
AREA_LABELS = ["60㎡以下", "60-90㎡", "90-120㎡", "120-150㎡", "150㎡以上"]

PRICE_BINS = [0, 50, 80, 120, 200, float("inf")]
PRICE_LABELS = ["50万以下", "50-80万", "80-120万", "120-200万", "200万以上"]


def cn_to_int(text: str):
    """中文数字转整数，如 十 -> 10、二十二 -> 22。"""
    if text is None or text == "":
        return None
    if text.isdigit():
        return int(text)
    if "十" in text:
        left, _, right = text.partition("十")
        tens = _CN_DIGITS.get(left, 1) if left else 1
        ones = _CN_DIGITS.get(right, 0) if right else 0
        return tens * 10 + ones
    return _CN_DIGITS.get(text)


def _direction_type(value: str) -> str:
    """将朝向组合标准化为 南北通透 / 朝南 / 朝北 / 其他。"""
    value = str(value)
    has_south = "南" in value
    has_north = "北" in value
    if has_south and has_north:
        return "南北通透"
    if has_south:
        return "朝南"
    if has_north:
        return "朝北"
    return "其他"


def _filter_quantile(df: pd.DataFrame, column: str, lower: float = 0.002, upper: float = 0.998):
    """按分位数过滤极端值，返回过滤后的数据和删除行数。"""
    low, high = df[column].quantile([lower, upper])
    mask = df[column].between(low, high)
    return df[mask].copy(), int((~mask).sum())


def clean_data(raw: pd.DataFrame):
    """执行完整清洗流程，返回 (清洗后数据, 清洗报告)。"""
    df = raw.copy()
    steps = [("原始记录数", len(df))]

    # 1. 核心数值字段转换
    for col in ["total_price", "unit_price", "square"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["total_price", "unit_price", "square"])
    steps.append(("数值缺失剔除", before - len(df)))

    # 2. 同一房源重复挂牌去重
    before = len(df)
    df = df.drop_duplicates(subset=["title", "community", "total_price", "square"])
    steps.append(("重复挂牌剔除", before - len(df)))

    # 3. 楼层解析
    floor_parts = df["floor"].astype(str).str.extract(FLOOR_RE)
    df["floor_level"] = floor_parts[0]
    df["total_floors"] = pd.to_numeric(floor_parts[1], errors="coerce")

    # 4. 户型解析
    size_parts = df["size"].astype(str).str.extract(SIZE_RE)
    for idx, name in enumerate(["rooms", "halls", "bathrooms"], start=1):
        df[name] = pd.to_numeric(size_parts[idx], errors="coerce")

    # 5. 梯户比解析
    ladder_parts = df["elevatorNum"].astype(str).str.extract(LADDER_RE)
    df["elevators"] = ladder_parts[0].map(cn_to_int)
    df["households"] = ladder_parts[1].map(cn_to_int)
    df["households_per_elevator"] = df["households"] / df["elevators"].replace(0, pd.NA)

    # 6. 类别字段标准化
    df["direction_type"] = df["direction"].map(_direction_type)
    df["elevator_flag"] = df["elevator"].map({"有": 1, "无": 0})
    df["decoration"] = df["decoration"].fillna("其他")
    df["type"] = df["type"].fillna("其他")

    # 7. 极端值过滤
    for col in ["unit_price", "square", "total_price"]:
        df, removed = _filter_quantile(df, col)
        steps.append((f"{col} 极端值剔除", removed))

    # 8. 派生分组字段
    df["area_bin"] = pd.cut(df["square"], bins=AREA_BINS, labels=AREA_LABELS, right=False)
    df["price_bin"] = pd.cut(df["total_price"], bins=PRICE_BINS, labels=PRICE_LABELS, right=False)
    df["layout"] = df["rooms"].astype("Int64").astype(str) + "室" + df["halls"].astype("Int64").astype(str) + "厅"

    steps.append(("清洗后记录数", len(df)))

    cleaned = df.reset_index(drop=True)
    cleaned.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8-sig")

    report = {
        "steps": steps,
        "rows": len(cleaned),
        "missing": int(cleaned[["floor_level", "rooms", "halls", "elevators", "households"]].isna().sum().sum()),
        "districts": int(cleaned["district"].nunique()),
        "communities": int(cleaned["community"].nunique()),
    }
    return cleaned, report
