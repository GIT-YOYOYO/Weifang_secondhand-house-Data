"""数据清洗与特征构造模块。

功能：
    将原始挂牌数据整理为可直接分析的建模数据，并记录每一步清洗结果。

实现方式：
    1. 数值字段强转为数值类型，剔除核心字段缺失的记录；
    2. 按标题/小区/总价/面积识别同一房源的重复挂牌并去重；
    3. 用正则表达式从 floor、size、elevatorNum 中解析楼层、户型、梯户比，
       同时将朝向、装修、电梯等类别字段标准化；
    4. 对单价、面积、总价按 0.2%～99.8% 分位数过滤极端值；
    5. 构造面积段、总价段、户型标签等派生字段，输出 processed_data.csv。
    清洗步骤及行数变化通过 report 字典返回，用于报告的数据概览板块。
"""

import re

import pandas as pd

from .config import PROCESSED_DATA_PATH

# 楼层格式如“中楼层 (共11层)”，提取楼层位置和总层数
FLOOR_RE = re.compile(r"(低|中|高)楼层.*?共(\d+)层")
# 户型格式如“3室2厅1厨1卫”，提取室、厅、厨、卫数量
SIZE_RE = re.compile(r"(\d+)室(\d+)厅(\d+)厨(\d+)卫")
# 梯户比格式如“一梯两户”“四梯三十五户”，提取电梯数与每层户数
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
    """中文数字转整数。

    支持 1～99 的中文数字：不含“十”时按位映射（一 -> 1）；
    含“十”时拆分为十位和个位（十 -> 10、二十二 -> 22、三十五 -> 35）。
    空值返回 None，供后续缺失值统计。
    """
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
    """将朝向组合标准化为 南北通透 / 朝南 / 朝北 / 其他。

    同时包含“南”“北”视为南北通透；仅含其中一个时归为对应朝向；
    其余组合（如东西向）统一归为“其他”。
    """
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
    """按分位数过滤极端值，返回过滤后的数据和删除行数。

    默认保留 0.2%～99.8% 区间：既能剔除明显录入错误或异常挂牌，
    又避免误删高价大平层等真实高端样本。
    """
    low, high = df[column].quantile([lower, upper])
    mask = df[column].between(low, high)
    return df[mask].copy(), int((~mask).sum())


def clean_data(raw: pd.DataFrame):
    """执行完整清洗流程。

    按模块开头的 1～8 步依次处理，逐步记录每一步删除的行数，
    最终将结果写入 data/processed_data.csv。

    Args:
        raw: 原始房源数据。

    Returns:
        (cleaned, report) 二元组。cleaned 为清洗后的 DataFrame；
        report 包含 steps（各步骤及数量）、rows（最终行数）、
        missing（解析字段缺失数）、districts 和 communities。
    """
    df = raw.copy()
    steps = [("原始记录数", len(df))]

    # 1. 核心数值字段转换与缺失剔除
    #    挂牌数据中的价格、面积可能混入空串或非数值文本，先统一转为数值，
    #    无法解析的值记为 NaN。这里只删除三个核心字段缺失的记录，
    #    非核心字段的缺失留到分析或建模阶段处理，避免过度删除样本。
    for col in ["total_price", "unit_price", "square"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["total_price", "unit_price", "square"])
    steps.append(("数值缺失剔除", before - len(df)))

    # 2. 同一房源重复挂牌去重
    #    同一套房源可能被中介重复发布，其标题、小区、总价和面积通常一致。
    #    用这四个字段组成业务主键去重，保留首次出现的记录；
    #    只按标题去重会误删不同房源，只用 id 则识别不出重复挂牌。
    before = len(df)
    df = df.drop_duplicates(subset=["title", "community", "total_price", "square"])
    steps.append(("重复挂牌剔除", before - len(df)))

    # 3. 楼层解析
    #    原始格式如“中楼层 (共11层)”，用正则提取楼层位置（低/中/高）
    #    和总层数。格式不匹配时解析结果为 NaN，不在此处删除样本。
    floor_parts = df["floor"].astype(str).str.extract(FLOOR_RE)
    df["floor_level"] = floor_parts[0]
    df["total_floors"] = pd.to_numeric(floor_parts[1], errors="coerce")

    # 4. 户型解析
    #    原始格式如“3室2厅1厨1卫”。SIZE_RE 的捕获组依次为
    #    室、厅、厨、卫（第 3 组“厨”本次不使用），因此 rooms/halls
    #    分别取第 0、1 组，bathrooms 取第 3 组，避免列错位。
    size_parts = df["size"].astype(str).str.extract(SIZE_RE)
    df["rooms"] = pd.to_numeric(size_parts[0], errors="coerce")
    df["halls"] = pd.to_numeric(size_parts[1], errors="coerce")
    df["bathrooms"] = pd.to_numeric(size_parts[3], errors="coerce")

    # 5. 梯户比解析
    #    原始格式如“一梯两户”“四梯三十五户”，先分别提取电梯数和每层户数，
    #    再用 cn_to_int 将中文数字转为整数；户/梯由两者相除得到，
    #    电梯数为 0 时置为缺失，避免除零。
    ladder_parts = df["elevatorNum"].astype(str).str.extract(LADDER_RE)
    df["elevators"] = ladder_parts[0].map(cn_to_int)
    df["households"] = ladder_parts[1].map(cn_to_int)
    df["households_per_elevator"] = df["households"] / df["elevators"].replace(0, pd.NA)

    # 6. 类别字段标准化
    #    朝向组合较多，归并为南北通透/朝南/朝北/其他四类，便于分组统计；
    #    电梯“有/无”转为 1/0；装修和建筑类型缺失统一填“其他”，
    #    保证后续 groupby 和图表不会因空值丢失样本。
    df["direction_type"] = df["direction"].map(_direction_type)
    df["elevator_flag"] = df["elevator"].map({"有": 1, "无": 0})
    df["decoration"] = df["decoration"].fillna("其他")
    df["type"] = df["type"].fillna("其他")

    # 7. 极端值过滤
    #    对单价、面积、总价依次按 0.2%～99.8% 分位数裁剪，剔除明显录入
    #    错误或异常挂牌，同时保留高价大平层等真实高端样本；每一步单独
    #    记录删除行数，便于在报告中还原清洗漏斗。
    for col in ["unit_price", "square", "total_price"]:
        df, removed = _filter_quantile(df, col)
        steps.append((f"{col} 极端值剔除", removed))

    # 8. 派生分组字段
    #    按固定区间生成面积段和总价段，保证图表中的类别顺序稳定；
    #    layout 统一为“N室M厅”，使用 Int64 避免出现“3.0室”的格式。
    df["area_bin"] = pd.cut(df["square"], bins=AREA_BINS, labels=AREA_LABELS, right=False)
    df["price_bin"] = pd.cut(df["total_price"], bins=PRICE_BINS, labels=PRICE_LABELS, right=False)
    df["layout"] = df["rooms"].astype("Int64").astype(str) + "室" + df["halls"].astype("Int64").astype(str) + "厅"

    steps.append(("清洗后记录数", len(df)))

    # 落盘供后续分析复用，使用 utf-8-sig 保证 Excel 打开中文不乱码
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
