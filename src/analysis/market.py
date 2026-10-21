"""市场概况分析。

从整体价格水平、区县对比、总价与面积结构、装修和户型构成等角度
刻画潍坊市二手房市场的整体面貌。
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..charts import apply_theme
from ..config import TABLE_DIR
from ..preprocessing import AREA_LABELS, PRICE_LABELS


def _district_stats(df: pd.DataFrame) -> pd.DataFrame:
    """按区县聚合市场指标。

    中位数用于价格指标，避免高价豪宅等极端样本拉高均值；
    结果按中位单价降序排列，用于后续排名与结论生成。
    """
    stats = (
        df.groupby("district", as_index=False)
        .agg(
            房源数=("id", "size"),
            中位单价=("unit_price", "median"),
            平均单价=("unit_price", "mean"),
            中位总价=("total_price", "median"),
            中位面积=("square", "median"),
        )
        .sort_values("中位单价", ascending=False)
        .reset_index(drop=True)
    )
    return stats.round(1)


def _fig_district_price(stats: pd.DataFrame) -> go.Figure:
    """绘制区县中位单价横向条形图（升序排列，最高价在顶部）。"""
    data = stats.sort_values("中位单价")
    fig = px.bar(
        data, x="中位单价", y="district", orientation="h",
        text=data["中位单价"].round(0).astype(int),
        color="中位单价", color_continuous_scale=["#BFDBFE", "#2563EB"],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(title="元/㎡")
    fig.update_yaxes(title="")
    return apply_theme(fig, "各区县二手房中位单价", height=430)


def _fig_district_count(stats: pd.DataFrame) -> go.Figure:
    """绘制各区县在售房源数量柱状图，反映市场供应分布。"""
    data = stats.sort_values("房源数", ascending=False)
    fig = px.bar(
        data, x="district", y="房源数",
        text="房源数", color="district",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(title="")
    fig.update_yaxes(title="套")
    fig.update_layout(showlegend=False)
    return apply_theme(fig, "各区县在售房源数量", height=430)


def _fig_price_hist(df: pd.DataFrame) -> go.Figure:
    """绘制总价分布直方图，并用虚线标注中位数位置。"""
    median = df["total_price"].median()
    fig = px.histogram(df, x="total_price", nbins=50, color_discrete_sequence=["#2563EB"])
    fig.add_vline(
        x=median, line_dash="dash", line_color="#EF4444",
        annotation_text=f"中位数 {median:.0f} 万", annotation_position="top",
    )
    fig.update_xaxes(title="总价（万元）")
    fig.update_yaxes(title="房源数")
    return apply_theme(fig, "二手房总价分布", height=430)


def _fig_area_bin(df: pd.DataFrame) -> go.Figure:
    """按预设面积段统计房源数量，保持 60㎡以下到 150㎡以上的固定顺序。"""
    counts = df["area_bin"].value_counts().reindex(AREA_LABELS).reset_index()
    counts.columns = ["面积段", "房源数"]
    fig = px.bar(counts, x="面积段", y="房源数", text="房源数", color="面积段")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(title="")
    fig.update_yaxes(title="套")
    fig.update_layout(showlegend=False)
    return apply_theme(fig, "房源面积段分布", height=430)


def _fig_decoration(df: pd.DataFrame) -> go.Figure:
    """绘制装修情况构成环形图，展示精装/简装/毛坯/其他的占比。"""
    counts = df["decoration"].value_counts().reset_index()
    counts.columns = ["装修", "房源数"]
    fig = px.pie(
        counts, names="装修", values="房源数", hole=0.45,
        color_discrete_sequence=["#2563EB", "#F59E0B", "#10B981", "#94A3B8"],
    )
    fig.update_traces(textinfo="label+percent", textposition="outside")
    return apply_theme(fig, "房源装修构成", height=430)


def _fig_layout(df: pd.DataFrame) -> go.Figure:
    """绘制房源数量最多的 10 种户型（按“N室M厅”聚合）。"""
    counts = df["layout"].value_counts().head(10).reset_index()
    counts.columns = ["户型", "房源数"]
    fig = px.bar(
        counts.sort_values("房源数"), x="房源数", y="户型",
        orientation="h", text="房源数", color_discrete_sequence=["#8B5CF6"],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(title="套")
    fig.update_yaxes(title="")
    return apply_theme(fig, "主流户型 TOP10", height=430)


def analyze(df: pd.DataFrame) -> dict:
    """执行市场概况分析。

    Args:
        df: 清洗后的房源数据。

    Returns:
        包含 title、metrics（指标卡）、figures（Plotly 图表）、
        insights（文字结论）和 tables（数据表）的字典。
    """
    stats = _district_stats(df)
    stats.to_csv(TABLE_DIR / "market_district_stats.csv", index=False, encoding="utf-8-sig")

    # 计算总价段、面积段分布以及装修占比，用于生成结论
    price_bin_counts = df["price_bin"].value_counts().reindex(PRICE_LABELS)
    area_counts = df["area_bin"].value_counts().reindex(AREA_LABELS)
    decor_share = df["decoration"].value_counts(normalize=True).mul(100).round(1)
    top_price_district = stats.iloc[0]
    low_price_district = stats.iloc[-1]

    metrics = [
        {"label": "在售房源", "value": f"{len(df):,}", "unit": "套"},
        {"label": "覆盖小区", "value": f"{df['community'].nunique():,}", "unit": "个"},
        {"label": "整体中位单价", "value": f"{df['unit_price'].median():,.0f}", "unit": "元/㎡"},
        {"label": "整体中位总价", "value": f"{df['total_price'].median():,.0f}", "unit": "万元"},
        {"label": "平均建筑面积", "value": f"{df['square'].mean():.1f}", "unit": "㎡"},
        {"label": "覆盖区县", "value": str(df["district"].nunique()), "unit": "个"},
    ]

    figures = {
        "district_price": _fig_district_price(stats),
        "district_count": _fig_district_count(stats),
        "price_hist": _fig_price_hist(df),
        "area_bin": _fig_area_bin(df),
        "decoration": _fig_decoration(df),
        "layout": _fig_layout(df),
    }

    # 主力总价段/面积段：房源数量最多的区间及其占比
    main_price_bin = price_bin_counts.idxmax()
    main_price_share = price_bin_counts.max() / len(df) * 100
    main_area = area_counts.idxmax()
    main_area_share = area_counts.max() / len(df) * 100

    insights = [
        f"单价最高的区县为{top_price_district['district']}（中位 {top_price_district['中位单价']:.0f} 元/㎡），"
        f"最低为{low_price_district['district']}（{low_price_district['中位单价']:.0f} 元/㎡），"
        f"两者相差约 {top_price_district['中位单价'] / low_price_district['中位单价']:.1f} 倍。",
        f"主力总价段为 {main_price_bin}，占全部房源的 {main_price_share:.1f}%；"
        f"总价中位数为 {df['total_price'].median():.0f} 万元。",
        f"主力面积段为 {main_area}，占比 {main_area_share:.1f}%，市场以改善型需求为主。",
        f"精装房源占比 {decor_share.get('精装', 0):.1f}%，简装占 {decor_share.get('简装', 0):.1f}%，"
        f"毛坯占 {decor_share.get('毛坯', 0):.1f}%。",
        f"房源供应最多的区县为{stats.sort_values('房源数', ascending=False).iloc[0]['district']}，"
        f"共 {int(stats['房源数'].max())} 套。",
    ]

    table = stats.copy()
    for col in ["中位单价", "平均单价", "中位总价", "中位面积"]:
        table[col] = table[col].round(0).astype(int)

    return {
        "title": "市场概况",
        "metrics": metrics,
        "figures": figures,
        "insights": insights,
        "tables": [{"title": "各区县市场指标", "df": table}],
    }
