"""数据读取模块。

功能：
    统一封装原始数据与清洗后数据的读取逻辑，隔离文件路径和编码细节。

实现方式：
    load_raw 读取 data/sh_data.csv，load_processed 读取 preprocessing
    落盘的 data/processed_data.csv；两者均使用 utf-8-sig 编码，兼容
    带 BOM 的中文 CSV。默认路径来自 config，也支持传入自定义路径。
"""

import pandas as pd

from .config import PROCESSED_DATA_PATH, RAW_DATA_PATH


def load_raw(path=RAW_DATA_PATH) -> pd.DataFrame:
    """读取原始二手房 CSV 数据。

    Args:
        path: 原始数据文件路径，默认为 data/sh_data.csv。

    Returns:
        原始房源数据 DataFrame，字段保持 CSV 中的原始格式。
    """
    return pd.read_csv(path, encoding="utf-8-sig")


def load_processed(path=PROCESSED_DATA_PATH) -> pd.DataFrame:
    """读取清洗并构造特征后的数据。

    Args:
        path: 清洗后数据文件路径，默认为 data/processed_data.csv。

    Returns:
        已包含解析字段和派生字段的 DataFrame。
    """
    return pd.read_csv(path, encoding="utf-8-sig")
