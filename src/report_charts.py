"""报告静态图表生成模块（Matplotlib）。

功能：
    为《数据质量分析报告》和《数据分析报告》生成 Word 文档所需的
    静态 JPG 插图，与 Plotly 交互式图表并存。

实现方式：
    - 所有图表使用 Matplotlib 的 Agg 后端绘制，不依赖浏览器；
    - 统一设置 Microsoft YaHei 中文字体、蓝/橙/绿配色和简洁网格；
    - 数据质量图（dq_*）基于原始数据与清洗后数据计算；
    - 数据分析图（da_*）基于清洗后数据，其中模型图重新训练随机森林
      并使用与 modeling 模块相同的特征和参数，保证数字一致；
    - 输出目录为 output/report_assets/，由 generate_report_charts 统一调度。
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_STATE, REPORT_ASSET_DIR, TABLE_DIR
from .preprocessing import FLOOR_RE, LADDER_RE, SIZE_RE, _filter_quantile

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2563EB"
ORANGE = "#F59E0B"
GREEN = "#10B981"
RED = "#EF4444"
PALETTE = ["#2563EB", "#F59E0B", "#10B981", "#EF4444",
           "#8B5CF6", "#06B6D4", "#EC4899", "#84CC16"]

NUMERIC_FEATURES = [
    "square", "rooms", "halls", "bathrooms", "total_floors",
    "households_per_elevator", "elevator_flag",
]
CATEGORICAL_FEATURES = ["district", "type", "decoration", "floor_level", "direction_type"]


def _style_ax(ax, grid_axis="y"):
    """统一坐标轴样式：去掉上/右边框，添加浅色网格。"""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, name: str, output_dir: Path) -> Path:
    """保存图表为 JPG（180 DPI、质量 92），返回文件路径。"""
    path = output_dir / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white",
                format="jpg", pil_kwargs={"quality": 92})
    plt.close(fig)
    return path


def _figure_pipeline_flow(output_dir: Path) -> Path:
    """图 1-1：数据分层流转与质量控制节点。"""
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        (0.2, 2.4, 1.8, 1.3, "原始层\nsh_data.csv\n18,339 行 × 16 字段", "#DBEAFE"),
        (2.35, 2.4, 1.8, 1.3, "清洗与校验\n去重 · 解析 · 过滤", "#FEF3C7"),
        (4.5, 2.4, 1.8, 1.3, "明细层\nprocessed_data.csv\n18,115 行", "#D1FAE5"),
        (6.65, 2.4, 1.8, 1.3, "应用层\n四个分析模块\n指标 / 图表 / 表格", "#EDE9FE"),
        (8.35, 2.4, 1.45, 1.3, "交付物\nHTML 报告\n20 图 · 6 表", "#FCE7F3"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                                    facecolor=color, edgecolor="#94A3B8", linewidth=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)
    for x0 in (2.0, 4.15, 6.3, 8.15):
        ax.add_patch(FancyArrowPatch((x0 + 0.15, 3.05), (x0 + 0.33, 3.05), arrowstyle="-|>",
                                     mutation_scale=14, color="#475569", linewidth=1.5))
    ax.add_patch(FancyBboxPatch((0.2, 0.65), 9.6, 0.8, boxstyle="round,pad=0.04,rounding_size=0.08",
                                facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=1))
    ax.text(5.0, 1.05,
            "贯穿全流程：main.py 流水线编排 · 清洗报告与运行日志 · 可复现随机种子（random_state=42）",
            ha="center", va="center", fontsize=9.5, color="#334155")
    for x0 in (1.1, 3.25, 5.4, 7.55, 9.0):
        ax.annotate("", xy=(x0, 2.4), xytext=(x0, 1.45),
                    arrowprops=dict(arrowstyle="-", color="#CBD5E1", linewidth=1, linestyle="--"))
    return _save(fig, "dq_01_流程.jpg", output_dir)


def _figures_quality(raw: pd.DataFrame, cleaned: pd.DataFrame, output_dir: Path) -> list[Path]:
    """生成数据质量报告所需的 7 张图（dq_01～dq_07）。"""
    paths = [_figure_pipeline_flow(output_dir)]

    # 图 2-1：检查结果汇总（16 项：通过 10、警告 5、不通过 1）
    dims = ["完整性", "规范性", "唯一性", "一致性", "准确性", "时效性", "代表性"]
    passed = [2, 5, 1, 2, 0, 0, 0]
    warned = [0, 0, 1, 0, 2, 0, 2]
    failed = [0, 0, 0, 0, 0, 1, 0]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    y = np.arange(len(dims))
    ax.barh(y, passed, color=GREEN, label="通过", height=0.56)
    ax.barh(y, warned, left=passed, color=ORANGE, label="警告", height=0.56)
    ax.barh(y, failed, left=np.array(passed) + np.array(warned), color=RED, label="不通过", height=0.56)
    ax.set_yticks(y, dims)
    ax.set_xlabel("检查项数量（项）")
    ax.set_title("各质量维度检查结果：共 16 项，通过 10 项、警告 5 项、不通过 1 项", fontsize=11)
    ax.legend(loc="lower right", frameon=False)
    _style_ax(ax, grid_axis="x")
    paths.append(_save(fig, "dq_02_检查汇总.jpg", output_dir))

    # 图 3-1：各区县房源数量与中位单价
    district = cleaned.groupby("district").agg(n=("id", "size"), price=("unit_price", "median")).sort_values("n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.barh(district.index, district["n"], color=BLUE)
    for i, v in enumerate(district["n"]):
        ax.text(v + 20, i, f"{v:,}", va="center", fontsize=8.5)
    ax.set_title("各区县房源数量（套）", fontsize=11)
    _style_ax(ax, grid_axis="x")
    ax = axes[1]
    ax.barh(district.index, district["price"], color=ORANGE)
    for i, v in enumerate(district["price"]):
        ax.text(v + 60, i, f"{v:,.0f}", va="center", fontsize=8.5)
    ax.set_title("各区县中位单价（元/㎡）", fontsize=11)
    _style_ax(ax, grid_axis="x")
    fig.tight_layout()
    paths.append(_save(fig, "dq_03_区县覆盖.jpg", output_dir))

    # 图 4-1：字段缺失率与关键字段解析成功率
    fields = list(raw.columns)
    missing = [raw[c].isna().mean() * 100 for c in fields]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    ax.barh(fields, missing, color=GREEN)
    ax.set_xlim(0, 105)
    ax.set_title("16 个字段缺失率（分母 18,339 行）：全部为 0%", fontsize=10.5)
    ax.set_xlabel("缺失率（%）")
    _style_ax(ax, grid_axis="x")
    parse_names = ["楼层解析", "户型解析", "梯户比解析", "朝向字段"]
    parse_values = [
        raw["floor"].astype(str).str.match(FLOOR_RE).mean() * 100,
        raw["size"].astype(str).str.match(SIZE_RE).mean() * 100,
        raw["elevatorNum"].astype(str).str.match(LADDER_RE).mean() * 100,
        raw["direction"].notna().mean() * 100,
    ]
    ax = axes[1]
    ax.barh(parse_names, parse_values, color=BLUE)
    ax.set_xlim(0, 110)
    for i, v in enumerate(parse_values):
        ax.text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_title("关键字段解析成功率", fontsize=10.5)
    ax.set_xlabel("成功率（%）")
    _style_ax(ax, grid_axis="x")
    fig.tight_layout()
    paths.append(_save(fig, "dq_04_缺失与解析.jpg", output_dir))

    # 图 5-1：清洗剔除明细（按清洗顺序复现每一步剔除数，保证与流水线一致）
    dup_business = int(raw.duplicated(subset=["title", "community", "total_price", "square"]).sum())
    working = raw.drop_duplicates(subset=["title", "community", "total_price", "square"])
    removed_by_filter = []
    for column in ["unit_price", "square", "total_price"]:
        working, removed = _filter_quantile(working, column)
        removed_by_filter.append(removed)
    removed = len(raw) - len(cleaned)
    steps = ["原始数据", "业务去重", "单价极端值", "面积极端值", "总价极端值", "清洗后"]
    values = [len(raw), -dup_business, -removed_by_filter[0], -removed_by_filter[1], -removed_by_filter[2], len(cleaned)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    running = 0
    for i, (name, val) in enumerate(zip(steps, values)):
        if i == 0:
            ax.bar(i, val, color=BLUE, width=0.6)
            ax.text(i, val + 220, f"{val:,}", ha="center", fontsize=9.5, fontweight="bold")
            running = val
        elif i == len(values) - 1:
            ax.bar(i, val, color=GREEN, width=0.6)
            ax.text(i, val + 220, f"{val:,}", ha="center", fontsize=9.5, fontweight="bold")
        else:
            base = running + val
            ax.bar(i, -val, bottom=base, color=ORANGE, width=0.6)
            ax.text(i, running + 220, f"−{-val}", ha="center", fontsize=9, color="#B45309")
            running = base
    ax.set_xticks(range(len(steps)), steps, fontsize=9)
    ax.set_ylabel("记录数（行）")
    ax.set_title(f"清洗剔除明细：{len(raw):,} → {len(cleaned):,} 行，共剔除 {removed} 行"
                 f"（{removed / len(raw) * 100:.2f}%）", fontsize=11)
    ax.set_ylim(0, len(raw) * 1.1)
    _style_ax(ax)
    paths.append(_save(fig, "dq_05_清洗.jpg", output_dir))

    # 图 6-1：挂牌单价与计算单价的一致性
    calc_price = raw["total_price"] * 10000 / raw["square"]
    diff = (raw["unit_price"] - calc_price).abs()
    sample_idx = np.random.default_rng(RANDOM_STATE).choice(len(raw), size=min(3000, len(raw)), replace=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(calc_price.iloc[sample_idx], raw["unit_price"].iloc[sample_idx],
               s=8, alpha=0.35, color=BLUE, edgecolors="none")
    lim = [0, 20000]
    ax.plot(lim, lim, color=RED, linestyle="--", linewidth=1.5, label="完全一致线 y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("由总价与面积计算的单价（元/㎡）")
    ax.set_ylabel("挂牌单价（元/㎡）")
    ax.set_title(f"挂牌单价与计算单价的偏差：99% 记录 ≤{diff.quantile(0.99):.0f} 元，"
                 f"最大 {diff.max():.0f} 元（n={len(raw):,}）", fontsize=10.5)
    ax.legend(frameon=False, loc="upper left")
    _style_ax(ax, grid_axis="both")
    paths.append(_save(fig, "dq_06_一致性.jpg", output_dir))

    # 图 7-1：区域代表性
    counts = raw["district"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.barh(counts.index, counts.values, color=BLUE)
    for i, (v, pct) in enumerate(zip(counts.values, counts.values / len(raw) * 100)):
        ax.text(v + 20, i, f"{v:,}（{pct:.1f}%）", va="center", fontsize=8.5)
    ax.set_xlim(0, counts.max() * 1.25)
    ax.set_xlabel("房源数量（套，分母 18,339）")
    ax.set_title(f"样本空间分布：8 个区县均有一定覆盖（单区县最高占 {counts.max() / len(raw) * 100:.1f}%）",
                 fontsize=10.5)
    _style_ax(ax, grid_axis="x")
    paths.append(_save(fig, "dq_07_代表性.jpg", output_dir))
    return paths


def _figures_analysis(cleaned: pd.DataFrame, output_dir: Path) -> list[Path]:
    """生成数据分析报告所需的 5 张图（da_01～da_05）。"""
    paths = []

    # 图 3-1：区县中位单价
    district = cleaned.groupby("district").agg(price=("unit_price", "median"), n=("id", "size")).sort_values("price")
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.barh(district.index, district["price"], color=BLUE)
    for i, (v, n) in enumerate(zip(district["price"], district["n"])):
        ax.text(v + 80, i, f"{v:,.0f} 元/㎡（{n:,} 套）", va="center", fontsize=8.5)
    ax.set_xlim(0, district["price"].max() * 1.32)
    ax.set_xlabel("中位单价（元/㎡）")
    ax.set_title("高新技术产业开发区中位单价最高，寒亭区最低", fontsize=11)
    _style_ax(ax, grid_axis="x")
    paths.append(_save(fig, "da_01_区县.jpg", output_dir))

    # 图 4-1：小区单价 TOP15
    grouped = cleaned.groupby("community").agg(n=("id", "size"), price=("unit_price", "median"))
    grouped = grouped[grouped["n"] >= 30].sort_values("price", ascending=False).head(15).sort_values("price")
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.barh(grouped.index, grouped["price"], color=ORANGE)
    for i, v in enumerate(grouped["price"]):
        ax.text(v + 100, i, f"{v:,.0f}", va="center", fontsize=8.5)
    ax.set_xlim(0, grouped["price"].max() * 1.18)
    ax.set_xlabel("中位单价（元/㎡）")
    ax.set_title("样本量 ≥30 套的小区中，歌尔绿城单价最高（13,272 元/㎡）", fontsize=10.5)
    _style_ax(ax, grid_axis="x")
    paths.append(_save(fig, "da_02_小区.jpg", output_dir))

    # 图 5-1：装修、电梯、楼层与建筑类型的单价差异
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    work = cleaned.copy()
    work["elevator_label"] = work["elevator_flag"].map({1: "有电梯", 0: "无电梯"})
    specs = [
        ("decoration", ["精装", "简装", "毛坯", "其他"], "装修情况"),
        ("elevator_label", ["无电梯", "有电梯"], "有无电梯"),
        ("floor_level", ["低", "中", "高"], "楼层位置"),
        ("type", ["板楼", "板塔结合", "塔楼"], "建筑类型"),
    ]
    for ax, (col, order, title) in zip(axes.flat, specs):
        med = work.groupby(col)["unit_price"].median().reindex(
            [o for o in order if o in work[col].unique()]
        )
        ax.bar(med.index, med.values, color=PALETTE[: len(med)])
        for i, v in enumerate(med.values):
            ax.text(i, v + 80, f"{v:,.0f}", ha="center", fontsize=8.5)
        ax.set_title(f"{title}（中位单价，元/㎡）", fontsize=10.5)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_ylim(0, med.max() * 1.18)
        _style_ax(ax)
    fig.suptitle("装修、电梯、楼层与建筑类型的单价差异", fontsize=12)
    fig.tight_layout()
    paths.append(_save(fig, "da_03_因素.jpg", output_dir))

    # 图 6-1：随机森林预测效果
    features = cleaned[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = cleaned["unit_price"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE
    )
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         CATEGORICAL_FEATURES),
    ])
    model = Pipeline([
        ("pre", pre),
        ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                        n_jobs=-1, random_state=RANDOM_STATE)),
    ])
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    idx = np.random.default_rng(RANDOM_STATE).choice(len(y_test), size=min(3000, len(y_test)), replace=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(np.asarray(y_test)[idx], pred[idx], s=8, alpha=0.35, color=BLUE, edgecolors="none")
    ax.plot([0, 20000], [0, 20000], color=RED, linestyle="--", linewidth=1.5, label="理想预测 y = x")
    ax.set_xlim(0, 20000)
    ax.set_ylim(0, 20000)
    ax.set_xlabel("实际单价（元/㎡）")
    ax.set_ylabel("预测单价（元/㎡）")
    ax.set_title(f"随机森林测试集预测：R²={r2:.3f}，MAE={mae:,.0f} 元/㎡", fontsize=10.5)
    ax.legend(frameon=False, loc="upper left")
    _style_ax(ax, grid_axis="both")
    paths.append(_save(fig, "da_04_模型.jpg", output_dir))

    # 图 6-2：低价候选 TOP10
    bargain_path = TABLE_DIR / "bargain_candidates.csv"
    if bargain_path.exists():
        bargain = pd.read_csv(bargain_path, encoding="utf-8-sig").head(10).sort_values("折价率(%)")
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        labels = [str(c)[:12] for c in bargain["小区"]]
        ax.barh(labels, bargain["折价率(%)"], color=ORANGE)
        for i, v in enumerate(bargain["折价率(%)"]):
            ax.text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=8.5)
        ax.set_xlabel("相对模型预测单价的折价率（%）")
        ax.set_title("测试集中折价率最高的 10 个小区房源（阈值 30%）", fontsize=10.5)
        _style_ax(ax, grid_axis="x")
        paths.append(_save(fig, "da_05_低价.jpg", output_dir))
    return paths


def generate_report_charts(raw: pd.DataFrame, cleaned: pd.DataFrame) -> list[Path]:
    """生成两份 Word 报告所需的全部 Matplotlib 插图。

    Args:
        raw: 原始房源数据，用于数据质量检查图。
        cleaned: 清洗后的房源数据，用于分析图。

    Returns:
        生成的 JPG 文件路径列表。
    """
    REPORT_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    paths = _figures_quality(raw, cleaned, REPORT_ASSET_DIR)
    paths += _figures_analysis(cleaned, REPORT_ASSET_DIR)
    return paths
