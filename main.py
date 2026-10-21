"""项目入口：运行完整数据分析流水线。

功能：
    一条命令完成数据读取、清洗、四类分析、图表导出和 HTML 报告生成。

实现方式：
    直接调用 src.pipeline.run_pipeline()，具体业务逻辑全部封装在 src/ 包中，
    本文件仅作为可执行入口，便于在 PyCharm 或命令行中运行。
"""

from src.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
