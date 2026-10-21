"""数据读取模块。"""

import pandas as pd

from .config import PROCESSED_DATA_PATH, RAW_DATA_PATH


def load_raw(path=RAW_DATA_PATH) -> pd.DataFrame:
    """读取原始二手房数据。"""
    return pd.read_csv(path, encoding="utf-8-sig")


def load_processed(path=PROCESSED_DATA_PATH) -> pd.DataFrame:
    """读取清洗后的数据。"""
    return pd.read_csv(path, encoding="utf-8-sig")
