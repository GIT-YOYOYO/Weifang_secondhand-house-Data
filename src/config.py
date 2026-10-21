"""项目配置：路径与全局常量。"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

for _dir in (DATA_DIR, OUTPUT_DIR, FIGURE_DIR, TABLE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

RAW_DATA_PATH = DATA_DIR / "sh_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.csv"
REPORT_PATH = OUTPUT_DIR / "report.html"

RANDOM_STATE = 42
TOP_N = 15
MIN_COMMUNITY_SAMPLES = 30

COLOR_SEQUENCE = [
    "#2563EB", "#F59E0B", "#10B981", "#EF4444",
    "#8B5CF6", "#06B6D4", "#EC4899", "#84CC16",
]
