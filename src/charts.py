"""Plotly 图表统一样式。

功能：
    为各分析模块生成的图表提供一致的主题、字体、配色和布局。

实现方式：
    apply_theme 集中设置 plotly_white 模板、中文字体、颜色序列、边距、
    图例位置和坐标轴网格样式；各分析模块创建图表后调用一次即可统一外观。
"""

import plotly.graph_objects as go

from .config import COLOR_SEQUENCE

FONT_FAMILY = "Microsoft YaHei, PingFang SC, SimHei, sans-serif"


def apply_theme(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    """为图表应用统一主题。

    Args:
        fig: 待美化的 Plotly Figure 对象。
        title: 图表标题。
        height: 图表高度（像素），默认 420。

    Returns:
        修改后的 Figure，便于链式调用。
    """
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#0F172A")),
        template="plotly_white",
        colorway=COLOR_SEQUENCE,
        font=dict(family=FONT_FAMILY, size=12, color="#475569"),
        margin=dict(l=50, r=30, t=60, b=50),
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel=dict(font_family=FONT_FAMILY, font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#E2E8F0", zeroline=False)
    return fig
