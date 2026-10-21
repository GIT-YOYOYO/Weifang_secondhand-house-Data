"""交互式 HTML 分析报告生成。

功能：
    把四个分析模块的结果拼接为单文件、可离线打开的 HTML 报告，
    并支持将每张图表导出为独立 HTML 文件。

实现方式：
    - 使用 plotly.io.to_html 转换图表，仅第一张图内嵌 plotly.js，
      后续图表复用该脚本，在离线可用与控制体积之间取得平衡；
    - 通过字符串模板和 CSS 渲染页头、指标卡、关键发现、图表网格与数据表；
    - 所有动态文本均经 html.escape 转义，避免 HTML 注入；
    - export_figures 依据 FIGURE_NAME_MAP 将单图以中文文件名写入 output/figures/。
"""

import html
from datetime import datetime

import pandas as pd
import plotly.io as pio

from .config import FIGURE_DIR, REPORT_PATH

SECTION_META = {
    "market": ("市场概况", "从价格水平、区县分布、面积与总价结构刻画当前市场。"),
    "community": ("小区维度", "以小区为单位聚合，识别价格高地、供应大户与折价小区。"),
    "factors": ("价格影响因素", "量化装修、电梯、楼层、建筑类型与面积对单价的影响。"),
    "modeling": ("建模与应用", "训练价格预测模型，评估拟合效果并识别潜在低价房源。"),
}

# 独立图表文件的英文键 -> 中文文件名
FIGURE_NAME_MAP = {
    "market_district_price": "市场概况_区县中位单价",
    "market_district_count": "市场概况_区县房源数量",
    "market_price_hist": "市场概况_总价分布",
    "market_area_bin": "市场概况_面积段分布",
    "market_decoration": "市场概况_装修构成",
    "market_layout": "市场概况_主流户型",
    "community_top_price": "小区维度_单价排行",
    "community_top_supply": "小区维度_供应排行",
    "community_scatter": "小区维度_规模与单价",
    "factors_decoration": "影响因素_装修",
    "factors_elevator": "影响因素_电梯",
    "factors_floor": "影响因素_楼层",
    "factors_building": "影响因素_建筑类型",
    "factors_impact": "影响因素_溢价对比",
    "factors_area_scatter": "影响因素_面积与单价",
    "modeling_model_compare": "建模应用_模型对比",
    "modeling_pred_scatter": "建模应用_预测效果",
    "modeling_importance": "建模应用_特征重要性",
    "modeling_residuals": "建模应用_残差分布",
    "modeling_district_error": "建模应用_区县误差",
}

CSS = """
:root {
  --bg: #f1f5f9;
  --card: #ffffff;
  --text: #0f172a;
  --muted: #64748b;
  --line: #e2e8f0;
  --primary: #2563eb;
  --primary-dark: #1e40af;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", "Segoe UI", sans-serif;
  line-height: 1.65;
}
.container { width: min(1280px, calc(100% - 48px)); margin: 0 auto; }
.hero {
  background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #2563eb 100%);
  color: #fff;
  padding: 56px 0 32px;
  margin-bottom: 32px;
}
.hero .eyebrow {
  letter-spacing: .22em;
  font-size: 12px;
  color: #93c5fd;
  margin: 0 0 10px;
}
.hero h1 { font-size: 36px; margin: 0 0 12px; font-weight: 700; }
.hero .subtitle { color: #cbd5e1; margin: 0 0 28px; font-size: 14px; }
.nav { display: flex; flex-wrap: wrap; gap: 8px; }
.nav a {
  color: #dbeafe;
  text-decoration: none;
  font-size: 14px;
  padding: 8px 16px;
  border-radius: 999px;
  background: rgba(255, 255, 255, .1);
  border: 1px solid rgba(255, 255, 255, .16);
  transition: all .2s ease;
}
.nav a:hover { background: rgba(255, 255, 255, .22); color: #fff; }
.section { margin-bottom: 48px; }
.section-head { display: flex; align-items: flex-start; gap: 16px; margin-bottom: 20px; }
.section-index {
  flex: 0 0 auto;
  width: 46px; height: 46px;
  border-radius: 12px;
  background: var(--primary);
  color: #fff;
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 18px;
  box-shadow: 0 6px 16px rgba(37, 99, 235, .25);
}
.section-head h2 { margin: 0; font-size: 25px; }
.section-head p { margin: 2px 0 0; color: var(--muted); font-size: 14px; }
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 14px;
  margin-bottom: 18px;
}
.metric {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 18px 20px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
}
.metric .label { color: var(--muted); font-size: 13px; }
.metric .value { font-size: 28px; font-weight: 700; color: var(--primary-dark); line-height: 1.3; }
.metric .unit { color: var(--muted); font-size: 13px; margin-left: 4px; }
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 22px 26px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
}
.insights { margin-bottom: 18px; }
.insights h3 { margin: 0 0 10px; font-size: 16px; }
.insights ul { margin: 0; padding-left: 0; list-style: none; }
.insights li {
  position: relative;
  padding-left: 22px;
  margin: 8px 0;
  color: #334155;
  font-size: 14px;
}
.insights li::before {
  content: "";
  position: absolute;
  left: 4px; top: 10px;
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--primary);
}
.figure-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
  margin-bottom: 18px;
}
.figure-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 10px;
  overflow: hidden;
  box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
}
.figure-card.wide { grid-column: 1 / -1; }
.table-block { margin-top: 6px; }
.table-block h3 { font-size: 16px; margin: 22px 0 10px; }
.table-wrap {
  overflow-x: auto;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
}
table.data-table { width: 100%; border-collapse: collapse; font-size: 13px; white-space: nowrap; }
table.data-table th {
  background: #f8fafc;
  color: #475569;
  text-align: left;
  padding: 10px 14px;
  border-bottom: 1px solid var(--line);
  font-weight: 600;
}
table.data-table td { padding: 9px 14px; border-bottom: 1px solid #f1f5f9; }
table.data-table tr:last-child td { border-bottom: none; }
table.data-table tr:hover td { background: #f8fafc; }
.clean-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }
.clean-item {
  background: #f8fafc;
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 14px 16px;
  display: flex; justify-content: space-between; align-items: center;
}
.clean-item span { color: var(--muted); font-size: 13px; }
.clean-item strong { font-size: 18px; color: var(--primary-dark); }
footer {
  color: var(--muted);
  text-align: center;
  font-size: 13px;
  padding: 28px 0 48px;
  border-top: 1px solid var(--line);
  margin-top: 24px;
}
@media (max-width: 900px) {
  .figure-grid { grid-template-columns: 1fr; }
  .hero h1 { font-size: 27px; }
  .container { width: calc(100% - 28px); }
}
"""


def _figure_html(fig, first: bool) -> str:
    """将 Plotly 图表转为可嵌入的 HTML。

    第一张图内嵌完整 plotly.js，后续图表复用该脚本，
    使整个报告成为可离线打开的单文件。
    """
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=True if first else False,
        config={"responsive": True, "displaylogo": False},
    )


def _table_html(df: pd.DataFrame) -> str:
    """将 DataFrame 转为带样式的 HTML 表格。"""
    return df.to_html(index=False, classes="data-table", border=0, escape=True)


def _metrics_html(metrics: list[dict]) -> str:
    """渲染顶部指标卡网格（标签 + 数值 + 单位）。"""
    cards = []
    for item in metrics:
        unit = f'<span class="unit">{html.escape(item["unit"])}</span>' if item.get("unit") else ""
        cards.append(
            f'<div class="metric"><div class="label">{html.escape(item["label"])}</div>'
            f'<div class="value">{html.escape(str(item["value"]))}{unit}</div></div>'
        )
    return f'<div class="metrics-grid">{"".join(cards)}</div>'


def _insights_html(insights: list[str]) -> str:
    """渲染“关键发现”列表，所有文本做 HTML 转义防止注入。"""
    items = "".join(f"<li>{html.escape(text)}</li>" for text in insights)
    return f'<div class="card insights"><h3>关键发现</h3><ul>{items}</ul></div>'


def _tables_html(tables: list[dict]) -> str:
    """渲染分析模块附带的数据表区块。"""
    blocks = []
    for table in tables:
        blocks.append(
            f'<div class="table-block"><h3>{html.escape(table["title"])}</h3>'
            f'<div class="table-wrap">{_table_html(table["df"])}</div></div>'
        )
    return "".join(blocks)


def _render_section(section_id: str, index: str, result: dict, first: list[bool]) -> str:
    """渲染单个分析板块：标题、指标卡、结论、图表网格和数据表。

    first 使用单元素列表以便在循环中记录是否已嵌入 plotly.js。
    """
    figures = []
    for fig in result["figures"].values():
        html_fig = _figure_html(fig, first[0])
        first[0] = False
        figures.append(f'<div class="figure-card">{html_fig}</div>')
    return f"""
    <section class="section" id="{section_id}">
      <div class="section-head">
        <div class="section-index">{index}</div>
        <div><h2>{html.escape(result["title"])}</h2><p>{SECTION_META[section_id][1]}</p></div>
      </div>
      {_metrics_html(result["metrics"])}
      {_insights_html(result["insights"])}
      <div class="figure-grid">{"".join(figures)}</div>
      {_tables_html(result["tables"])}
    </section>
    """


def _render_overview(clean_report: dict, total_rows: int) -> str:
    """渲染数据概览板块，展示清洗漏斗和过滤比例。"""
    steps = clean_report["steps"]
    items = "".join(
        f'<div class="clean-item"><span>{html.escape(name)}</span><strong>{count:,}</strong></div>'
        for name, count in steps
    )
    removed = steps[0][1] - clean_report["rows"]
    return f"""
    <section class="section" id="overview">
      <div class="section-head">
        <div class="section-index">00</div>
        <div><h2>数据概览</h2><p>原始数据经过去重、字段解析与极端值过滤后的总体情况。</p></div>
      </div>
      {_metrics_html([
          {"label": "原始记录", "value": f"{total_rows:,}", "unit": "条"},
          {"label": "清洗后记录", "value": f"{clean_report['rows']:,}", "unit": "条"},
          {"label": "过滤比例", "value": f"{removed / total_rows * 100:.1f}", "unit": "%"},
          {"label": "覆盖区县", "value": str(clean_report["districts"]), "unit": "个"},
          {"label": "覆盖小区", "value": f"{clean_report['communities']:,}", "unit": "个"},
      ])}
      <div class="card">
        <h3 style="margin:0 0 14px;font-size:16px;">清洗流水线</h3>
        <div class="clean-grid">{items}</div>
      </div>
    </section>
    """


def build_report(results: dict, clean_report: dict) -> str:
    """将各分析模块结果渲染为单文件 HTML 报告。

    依次渲染数据概览和四个分析板块，拼装页头、导航与页脚后写入
    REPORT_PATH。第一张图表会内嵌 plotly.js，其余图表复用。

    Args:
        results: {模块键: analyze() 返回值} 的有序字典。
        clean_report: preprocessing.clean_data 返回的清洗报告。

    Returns:
        生成的报告文件路径。
    """
    first = [True]
    sections = [_render_overview(clean_report, clean_report["steps"][0][1] if clean_report["steps"] else clean_report["rows"])]
    for index, (key, result) in enumerate(results.items(), start=1):
        sections.append(_render_section(key, f"{index:02d}", result, first))

    nav = ['<a href="#overview">数据概览</a>']
    nav += [f'<a href="#{key}">{SECTION_META[key][0]}</a>' for key in results]

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    document = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>潍坊市二手房市场分析报告</title>
<style>{CSS}</style>
</head>
<body>
<header class="hero">
  <div class="container">
    <p class="eyebrow">WEIFANG SECONDHAND HOUSING</p>
    <h1>潍坊市二手房市场分析报告</h1>
    <p class="subtitle">样本量 {clean_report['rows']:,} 套 · 覆盖 {clean_report['communities']:,} 个小区 · 生成于 {generated}</p>
    <nav class="nav">{"".join(nav)}</nav>
  </div>
</header>
<main class="container">
  {"".join(sections)}
</main>
<footer>
  <div class="container">数据来源：潍坊市二手房挂牌样本 · 分析结果仅供研究参考，不构成投资建议</div>
</footer>
</body>
</html>"""

    REPORT_PATH.write_text(document, encoding="utf-8")
    return str(REPORT_PATH)


def export_figures(results: dict) -> list[str]:
    """将每张图表另存为独立 HTML 文件（文件名使用中文）。"""
    paths = []
    for key, result in results.items():
        for name, fig in result["figures"].items():
            stem = FIGURE_NAME_MAP.get(f"{key}_{name}", f"{key}_{name}")
            path = FIGURE_DIR / f"{stem}.html"
            fig.write_html(path, include_plotlyjs="cdn", full_html=True)
            paths.append(str(path))
    return paths
