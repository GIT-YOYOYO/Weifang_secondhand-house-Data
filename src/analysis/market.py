"""市场概况分析。"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..charts import apply_theme
from ..config import TABLE_DIR
from ..preprocessing import AREA_LABELS, PRICE_LABELS


def _district_stats(df: pd.DataFrame) -> pd.DataFrame:
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
    counts = df["area_bin"].value_counts().reindex(AREA_LABELS).reset_index()
    counts.columns = ["面积段", "房源数"]
    fig = px.bar(counts, x="面积段", y="房源数", text="房源数", color="面积段")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(title="")
    fig.update_yaxes(title="套")
    fig.update_layout(showlegend=False)
    return apply_theme(fig, "房源面积段分布", height=430)


def _fig_decoration(df: pd.DataFrame) -> go.Figure:
    counts = df["decoration"].value_counts().reset_index()
    counts.columns = ["装修", "房源数"]
    fig = px.pie(
        counts, names="装修", values="房源数", hole=0.45,
        color_discrete_sequence=["#2563EB", "#F59E0B", "#10B981", "#94A3B8"],
    )
    fig.update_traces(textinfo="label+percent", textposition="outside")
    return apply_theme(fig, "房源装修构成", height=430)


def _fig_layout(df: pd.DataFrame) -> go.Figure:
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
    """执行市场概况分析，返回指标、图表、表格与结论。"""
    stats = _district_stats(df)
    stats.to_csv(TABLE_DIR / "market_district_stats.csv", index=False, encoding="utf-8-sig")

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
