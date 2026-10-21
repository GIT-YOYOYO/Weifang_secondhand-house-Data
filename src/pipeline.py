"""分析流水线：数据读取 → 清洗 → 四类分析 → 图表与 HTML 报告。"""

import time

from .analysis import community, factors, market, modeling
from .data_loader import load_raw
from .preprocessing import clean_data
from .report import build_report, export_figures

ANALYSIS_MODULES = [
    ("market", market),
    ("community", community),
    ("factors", factors),
    ("modeling", modeling),
]


def run_pipeline(export_figure_files: bool = True) -> str:
    """运行完整分析流水线，返回 HTML 报告路径。"""
    started = time.time()
    total_steps = len(ANALYSIS_MODULES) + 3
    step = 0

    print("=" * 62)
    print("  潍坊市二手房样本数据项目 · 分析流水线")
    print("=" * 62)

    step += 1
    print(f"\n[{step}/{total_steps}] 读取原始数据 ...")
    raw = load_raw()
    print(f"        原始记录 {len(raw):,} 条，字段 {raw.shape[1]} 个")

    step += 1
    print(f"[{step}/{total_steps}] 数据清洗与特征构造 ...")
    df, clean_report = clean_data(raw)
    for name, count in clean_report["steps"]:
        print(f"        {name}: {count:,}")
    print(f"        结果已保存至 data/processed_data.csv")

    results = {}
    for key, module in ANALYSIS_MODULES:
        step += 1
        print(f"[{step}/{total_steps}] 分析：{module.__name__.split('.')[-1]} ...")
        results[key] = module.analyze(df)
        print(f"        完成 {len(results[key]['figures'])} 张图表、"
              f"{len(results[key]['tables'])} 张数据表")

    step += 1
    print(f"[{step}/{total_steps}] 生成 HTML 报告与图表文件 ...")
    report_path = build_report(results, clean_report)
    if export_figure_files:
        paths = export_figures(results)
        print(f"        已导出 {len(paths)} 个独立图表至 output/figures/")

    elapsed = time.time() - started
    print(f"\n完成，用时 {elapsed:.1f} 秒")
    print(f"报告：{report_path}")
    return report_path
