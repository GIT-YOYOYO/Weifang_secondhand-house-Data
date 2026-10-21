"""价格影响因素分析。

功能：
    量化装修、电梯、楼层、建筑类型、朝向和面积对二手房单价的影响。

实现方式：
    - _group_summary 对每个维度分组统计中位单价、房源数和相对整体溢价；
    - 使用 Kruskal-Wallis 检验判断组间单价差异是否显著（样本不足的类别
      不参与检验）；
    - _box / _fig_impact 输出箱线图和溢价对比图，_fig_area_scatter 通过
      抽样散点与线性趋势线展示面积与单价的关系；
    - analyze 汇总指标、图表、分组统计表和自动生成的文字结论。
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import kruskal

from ..charts import apply_theme
from ..config import TABLE_DIR

# 分析维度：显示名 -> 数据列
FACTORS = {
    "装修": "decoration",
    "电梯": "elevator_label",
    "楼层": "floor_level",
    "建筑类型": "type",
    "朝向": "direction_type",
}

FLOOR_ORDER = ["低", "中", "高"]
TYPE_ORDER = ["板楼", "板塔结合", "塔楼"]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """将 0/1 电梯标记转换为可读类别，便于分组与图表展示。"""
    data = df.copy()
    data["elevator_label"] = data["elevator_flag"].map({1: "有电梯", 0: "无电梯"})
    return data


def _group_summary(data: pd.DataFrame, column: str, min_samples: int = 30) -> tuple[pd.DataFrame, float]:
    """计算某因素各类别的价格统计与 Kruskal-Wallis 检验 p 值。

    溢价以该因素所有房源的单价中位数为基准，避免不同因素之间
    因样本结构不同而无法横向比较。样本量不足 min_samples 的类别
    不参与显著性检验，防止小样本噪声影响结论。
    """
    overall = data["unit_price"].median()
    stats = (
        data.groupby(column, as_index=False)
        .agg(房源数=("id", "size"), 中位单价=("unit_price", "median"), 中位总价=("total_price", "median"))
    )
    # 相对整体中位数的溢价百分比，正值表示高于整体水平
    stats["相对整体溢价"] = (stats["中位单价"] / overall - 1) * 100
    groups = [g["unit_price"].to_numpy() for _, g in data.groupby(column) if len(g) >= min_samples]
    p_value = kruskal(*groups).pvalue if len(groups) >= 2 else np.nan
    return stats.round(1), p_value


def _box(data: pd.DataFrame, column: str, title: str, order=None, height: int = 420) -> go.Figure:
    """绘制某因素各类别的单价箱线图。

    箱体展示四分位区间与中位数，须线展示分布范围；
    order 用于固定类别顺序，points=False 避免大样本下散点过密。
    """
    fig = px.box(
        data, x=column, y="unit_price", color=column,
        category_orders={column: order} if order else None,
        points=False,
    )
    fig.update_xaxes(title="")
    fig.update_yaxes(title="元/㎡")
    fig.update_layout(showlegend=False)
    return apply_theme(fig, title, height=height)


def _fig_impact(impacts: pd.DataFrame) -> go.Figure:
    """汇总各因素类别的溢价幅度，用统一坐标系横向对比。"""
    data = impacts.sort_values("相对整体溢价")
    data["维度"] = data["因素"] + " · " + data["类别"]
    fig = px.bar(
        data, x="相对整体溢价", y="维度", orientation="h",
        color="因素", text=data["相对整体溢价"].round(1),
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside", cliponaxis=False)
    fig.add_vline(x=0, line_color="#94A3B8", line_width=1)
    fig.update_xaxes(title="相对整体中位单价溢价（%）")
    fig.update_yaxes(title="")
    return apply_theme(fig, "各因素类别的单价溢价对比", height=560)


def _fig_area_scatter(df: pd.DataFrame) -> go.Figure:
    """绘制面积-单价散点图并叠加线性趋势线。

    为控制渲染性能，最多随机抽样 5000 条（固定随机种子保证可复现）。
    """
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="square", y="unit_price", opacity=0.35,
        color_discrete_sequence=["#2563EB"],
    )
    slope, intercept = np.polyfit(sample["square"], sample["unit_price"], 1)
    xs = np.linspace(sample["square"].min(), sample["square"].max(), 60)
    fig.add_trace(
        go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="趋势线",
                   line=dict(color="#EF4444", width=2, dash="dash"))
    )
    fig.update_xaxes(title="建筑面积（㎡）")
    fig.update_yaxes(title="单价（元/㎡）")
    fig.update_layout(showlegend=False)
    return apply_theme(fig, "建筑面积与单价的关系", height=450)


def analyze(df: pd.DataFrame) -> dict:
    """执行价格影响因素分析。

    Args:
        df: 清洗后的房源数据。

    Returns:
        包含指标卡、箱线图/溢价对比图、分组统计表和文字结论的字典。
    """
    data = _prepare(df)
    overall_median = data["unit_price"].median()

    # 逐个因素计算分组统计和显著性 p 值，并统一列名为“因素/类别”
    summaries = {}
    p_values = {}
    for label, column in FACTORS.items():
        stats, p_value = _group_summary(data, column)
        stats.insert(0, "因素", label)
        stats.rename(columns={column: "类别"}, inplace=True)
        summaries[label] = stats
        p_values[label] = p_value

    combined = pd.concat(summaries.values(), ignore_index=True)
    combined.to_csv(TABLE_DIR / "factors_summary.csv", index=False, encoding="utf-8-sig")

    decoration_stats = summaries["装修"].set_index("类别")["中位单价"]
    elevator_stats = summaries["电梯"].set_index("类别")["中位单价"]
    type_stats = summaries["建筑类型"].set_index("类别")["中位单价"]
    floor_stats = summaries["楼层"].set_index("类别")["中位单价"]

    # 汇总样本量达标的类别，供溢价对比图使用
    impact_rows = []
    for label, stats in summaries.items():
        for _, row in stats[stats["房源数"] >= 30].iterrows():
            impact_rows.append({
                "因素": label,
                "类别": row["类别"],
                "相对整体溢价": row["相对整体溢价"],
            })
    impacts = pd.DataFrame(impact_rows)

    top_types = data["type"].value_counts()
    valid_types = [t for t in TYPE_ORDER if t in top_types.index]
    valid_floor = [f for f in FLOOR_ORDER if f in data["floor_level"].unique()]
    valid_decoration = [d for d in ["精装", "简装", "毛坯", "其他"] if d in data["decoration"].unique()]

    figures = {
        "decoration": _box(data, "decoration", "装修情况对单价的影响", valid_decoration),
        "elevator": _box(data, "elevator_label", "有无电梯对单价的影响", ["无电梯", "有电梯"]),
        "floor": _box(data, "floor_level", "楼层位置对单价的影响", valid_floor),
        "building": _box(data, "type", "建筑类型对单价的影响", valid_types),
        "impact": _fig_impact(impacts),
        "area_scatter": _fig_area_scatter(data),
    }

    corr = data["square"].corr(data["unit_price"], method="spearman")
    corr_total = data["square"].corr(data["total_price"], method="spearman")
    top_type = data["type"].value_counts().idxmax()
    main_floor = data["floor_level"].value_counts().idxmax()
    significant = sum(1 for p in p_values.values() if p < 0.05)

    def _premium_text(left: str, right: str, stats: pd.Series) -> str:
        """生成“A 较 B 中位单价高/低 X%”的对比结论文本。"""
        if left not in stats.index or right not in stats.index:
            return "样本不足"
        diff = (stats[left] / stats[right] - 1) * 100
        return f"{left}较{right}中位单价{'高' if diff >= 0 else '低'} {abs(diff):.1f}%"

    metrics = [
        {"label": "整体中位单价", "value": f"{overall_median:,.0f}", "unit": "元/㎡"},
        {"label": "面积-单价相关", "value": f"{corr:.2f}", "unit": "Spearman"},
        {"label": "面积-总价相关", "value": f"{corr_total:.2f}", "unit": "Spearman"},
        {"label": "显著影响因素", "value": str(sum(1 for p in p_values.values() if p < 0.05)), "unit": f"/ {len(p_values)} 个"},
    ]

    corr_text = (
        "面积越大单价越低，小户型房源单位面积价格更高"
        if corr < 0
        else "面积越大单价越高，大户型房源单位面积价格更高"
    )

    insights = [
        _premium_text("精装", "毛坯", decoration_stats) + "，装修状态与挂牌单价存在明显关联。",
        _premium_text("有电梯", "无电梯", elevator_stats) + "。",
        f"楼层方面，低楼层中位单价 {floor_stats.get('低', np.nan):.0f} 元/㎡，"
        f"中楼层 {floor_stats.get('中', np.nan):.0f} 元/㎡，高楼层 {floor_stats.get('高', np.nan):.0f} 元/㎡；"
        f"当前样本以{main_floor}楼层为主。",
        f"建筑类型以{top_type}为主（{int(data['type'].value_counts().iloc[0])} 套），"
        f"其中位单价 {type_stats.get(top_type, np.nan):.0f} 元/㎡。",
        f"面积与单价呈{'负' if corr < 0 else '正'}相关（Spearman {corr:.2f}），{corr_text}；"
        f"面积与总价相关系数 {corr_total:.2f}。",
        f"{len(p_values)} 个因素中有 {significant} 个通过 Kruskal-Wallis 检验（p < 0.05），"
        "表明这些属性与单价差异具有统计显著关联。",
    ]

    display = combined.copy()
    display.columns = ["因素", "类别", "房源数", "中位单价", "中位总价", "相对整体溢价(%)"]

    return {
        "title": "价格影响因素",
        "metrics": metrics,
        "figures": figures,
        "insights": insights,
        "tables": [{"title": "各因素分组价格统计", "df": display}],
    }
