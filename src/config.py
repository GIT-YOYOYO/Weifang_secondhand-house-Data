"""项目配置：路径与全局常量。"""

from pathlib import Path

# 项目根目录及各级输出目录
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"   # 独立图表 HTML
TABLE_DIR = OUTPUT_DIR / "tables"     # 分析结果 CSV

for _dir in (DATA_DIR, OUTPUT_DIR, FIGURE_DIR, TABLE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

RAW_DATA_PATH = DATA_DIR / "sh_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.csv"
REPORT_PATH = OUTPUT_DIR / "潍坊市二手房市场分析报告.html"

RANDOM_STATE = 42          # 统一随机种子，保证抽样与建模结果可复现
TOP_N = 15                 # 排行榜展示条数
MIN_COMMUNITY_SAMPLES = 30  # 小区参与排名的最大样本门槛

COLOR_SEQUENCE = [
    "#2563EB", "#F59E0B", "#10B981", "#EF4444",
    "#8B5CF6", "#06B6D4", "#EC4899", "#84CC16",
]
