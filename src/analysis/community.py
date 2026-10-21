"""小区维度分析。

功能：
    以小区为单位聚合房源，识别价格高地、供应大户，以及与所在区县
    平均水平相比明显折价的小区。

实现方式：
    _community_stats 按小区聚合价格、面积和精装占比，并计算相对区县
    中位单价的折价率；分析时只保留样本量达标的小区，避免小样本失真；
    各 _fig_* 函数生成排行图与气泡图；analyze 汇总指标、结论和 TOP20 表。
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..charts import apply_theme
from ..config import MIN_COMMUNITY_SAMPLES, TABLE_DIR, TOP_N


def _community_stats(df: pd.DataFrame) -> pd.DataFrame:
    """按小区聚合价格、面积、装修占比等指标。

    区县取该小区房源中出现次数最多的区县；折价率定义为
    (区县中位单价 - 小区中位单价) / 区县中位单价，正值表示低于区县水平。
    """
    grouped = (
        df.groupby("community", as_index=False)
        .agg(
            district=("district", lambda s: s.mode().iat[0] if not s.mode().empty else ""),
            房源数=("id", "size"),
            中位单价=("unit_price", "median"),
            中位总价=("total_price", "median"),
            中位面积=("square", "median"),
            精装占比=("decoration", lambda s: (s == "精装").mean() * 100),
        )
    )
    # 用所在区县的中位单价作为基准，计算小区相对折价率
    district_median = df.groupby("district")["unit_price"].median()
    grouped["区县中位单价"] = grouped["district"].map(district_median)
    grouped["相对区县折价率"] = (1 - grouped["中位单价"] / grouped["区县中位单价"]) * 100
    return grouped


def _fig_top_price(qualified: pd.DataFrame) -> go.Figure:
    """绘制中位单价最高的 N 个小区，颜色区分所属区县。"""
    data = qualified.nlargest(TOP_N, "中位单价").sort_values("中位单价")
    fig = px.bar(
        data, x="中位单价", y="community", orientation="h",
        text=data["中位单价"].round(0).astype(int),
        color="district", hover_data=["房源数", "中位总价"],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(title="元/㎡")
    fig.update_yaxes(title="")
    return apply_theme(fig, f"中位单价 TOP{TOP_N} 小区（样本≥{MIN_COMMUNITY_SAMPLES}套）", height=520)


def _fig_top_supply(qualified: pd.DataFrame) -> go.Figure:
    """绘制在售房源数量最多的 N 个小区。"""
    data = qualified.nlargest(TOP_N, "房源数").sort_values("房源数")
    fig = px.bar(
        data, x="房源数", y="community", orientation="h",
        text="房源数", color="district",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(title="套")
    fig.update_yaxes(title="")
    return apply_theme(fig, f"在售房源量 TOP{TOP_N} 小区", height=520)


def _fig_scatter(qualified: pd.DataFrame) -> go.Figure:
    """绘制小区“房源规模-中位单价”气泡图。

    气泡大小代表中位总价，横轴使用对数刻度以缓解少数超大社区的拉伸。
    """
    fig = px.scatter(
        qualified, x="房源数", y="中位单价", size="中位总价",
        color="district", hover_name="community",
        hover_data={"中位总价": ":.0f", "中位面积": ":.0f"},
        size_max=28, opacity=0.75,
    )
    fig.update_xaxes(title="在售房源数（套）", type="log")
    fig.update_yaxes(title="中位单价（元/㎡）")
    return apply_theme(fig, "小区房源规模与单价分布", height=480)


def analyze(df: pd.DataFrame) -> dict:
    """执行小区维度分析。

    样本量低于 MIN_COMMUNITY_SAMPLES 的小区不参与排名，
    避免个别挂牌造成中位价失真。

    Args:
        df: 清洗后的房源数据。

    Returns:
        包含指标卡、图表、结论与排行表的字典。
    """
    grouped = _community_stats(df)
    # 仅保留样本量达标的小区，保证排名稳定
    qualified = grouped[grouped["房源数"] >= MIN_COMMUNITY_SAMPLES].copy()
    qualified = qualified.sort_values("中位单价", ascending=False).reset_index(drop=True)

    export = qualified.copy()
    for col in ["中位单价", "中位总价", "中位面积", "区县中位单价", "精装占比", "相对区县折价率"]:
        export[col] = export[col].round(1)
    export.to_csv(TABLE_DIR / "community_ranking.csv", index=False, encoding="utf-8-sig")

    top_price = qualified.iloc[0]
    top_supply = qualified.nlargest(1, "房源数").iloc[0]
    best_value = qualified.nlargest(1, "相对区县折价率").iloc[0]

    metrics = [
        {"label": "纳入排名小区", "value": f"{len(qualified):,}", "unit": "个"},
        {"label": "小区样本门槛", "value": str(MIN_COMMUNITY_SAMPLES), "unit": "套"},
        {"label": "最高中位单价", "value": f"{top_price['中位单价']:,.0f}", "unit": "元/㎡"},
        {"label": "最大在售规模", "value": f"{int(top_supply['房源数']):,}", "unit": "套"},
    ]

    figures = {
        "top_price": _fig_top_price(qualified),
        "top_supply": _fig_top_supply(qualified),
        "scatter": _fig_scatter(qualified),
    }

    insights = [
        f"{top_price['community']}（{top_price['district']}）中位单价最高，"
        f"达 {top_price['中位单价']:.0f} 元/㎡，样本 {int(top_price['房源数'])} 套。",
        f"{top_supply['community']}在售房源最多，共 {int(top_supply['房源数'])} 套，"
        f"中位单价 {top_supply['中位单价']:.0f} 元/㎡。",
        f"相对所在区县折价最明显的小区为{best_value['community']}，"
        f"中位单价比区县中位低 {best_value['相对区县折价率']:.1f}%。",
        f"共有 {len(qualified)} 个小区在售房源达到 {MIN_COMMUNITY_SAMPLES} 套以上，"
        f"覆盖全市在售房源的 {qualified['房源数'].sum() / len(df) * 100:.1f}%。",
    ]

    display = qualified.head(20)[
        ["community", "district", "房源数", "中位单价", "中位总价", "中位面积", "精装占比", "相对区县折价率"]
    ].copy()
    display.columns = ["小区", "区县", "房源数", "中位单价", "中位总价", "中位面积", "精装占比(%)", "较区县折价(%)"]
    for col in ["中位单价", "中位总价", "中位面积", "精装占比(%)", "较区县折价(%)"]:
        display[col] = display[col].round(1)

    return {
        "title": "小区维度",
        "metrics": metrics,
        "figures": figures,
        "insights": insights,
        "tables": [{"title": "小区单价排行 TOP20", "df": display}],
    }
