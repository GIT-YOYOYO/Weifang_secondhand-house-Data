# 潍坊市二手房样本数据项目

基于潍坊市 18,000+ 条二手房挂牌样本的数据清洗、市场分析与价格建模项目，输出交互式 HTML 分析报告，并使用 Matplotlib 生成 Word 报告所需的静态插图。

## 功能模块

| 模块 | 内容 |
| --- | --- |
| 数据清洗 | 数值校验、重复挂牌去重、楼层/户型/梯户比解析、极端值过滤、派生特征构造 |
| 市场概况 | 区县价格对比、总价与面积分布、装修构成、主流户型 |
| 小区维度 | 小区价格排行、供应规模、相对区县折价率、规模与单价分布 |
| 价格影响因素 | 装修/电梯/楼层/建筑类型/朝向分组对比、Kruskal-Wallis 显著性检验、面积相关性 |
| 建模与应用 | 随机森林与岭回归价格预测、特征重要性、残差分析、潜在低价房源识别 |
| 报告静态图表 | 使用 Matplotlib 生成数据质量报告与数据分析报告的 12 张 JPG 插图 |

## 项目目录

```
潍坊市二手房样本数据项目/
├── data/
│   ├── sh_data.csv          # 原始二手房样本数据
│   └── processed_data.csv   # 清洗后数据（运行后生成）
├── src/
│   ├── config.py            # 路径与全局配置
│   ├── data_loader.py       # 数据读取
│   ├── preprocessing.py     # 数据清洗与特征构造
│   ├── charts.py            # Plotly 图表统一样式
│   ├── report_charts.py     # Matplotlib 报告插图生成
│   ├── report.py            # HTML 报告生成
│   ├── pipeline.py          # 分析流水线编排
│   └── analysis/
│       ├── market.py        # 市场概况
│       ├── community.py     # 小区维度
│       ├── factors.py       # 价格影响因素
│       └── modeling.py      # 建模与应用
├── output/
│   ├── 潍坊市二手房市场分析报告.html  # 交互式分析报告
│   ├── figures/             # 独立图表 HTML（中文文件名）
│   ├── report_assets/       # 报告静态插图 JPG（Matplotlib）
│   └── tables/              # 分析结果 CSV
├── main.py                  # 项目入口
├── requirements.txt         # 依赖清单
└── README.md
```

## 快速开始

```bash
pip install -r requirements.txt
python main.py
```

运行后用浏览器打开 `output/潍坊市二手房市场分析报告.html` 查看完整分析报告。

## 输出说明

- `output/潍坊市二手房市场分析报告.html`：单文件交互式报告（已内嵌 Plotly，可离线打开）
- `output/figures/*.html`：20 张独立图表，中文命名（如 `市场概况_区县中位单价.html`，Plotly 脚本通过 CDN 加载）
- `output/report_assets/*.jpg`：12 张报告静态插图（7 张数据质量 + 5 张数据分析，Matplotlib 生成）
- `output/tables/*.csv`：区县统计、小区排行、因素汇总、模型指标、低价候选等数据表
- `data/processed_data.csv`：清洗并构造特征后的建模数据

## 数据字段

`data/sh_data.csv` 共 18,339 条记录，包含 16 个字段：

| 字段 | 含义 |
| --- | --- |
| id | 记录编号 |
| title | 房源标题 |
| total_price | 总价（万元） |
| unit_price | 单价（元/㎡） |
| square | 建筑面积（㎡） |
| size | 户型（如 2室2厅1厨1卫） |
| floor | 楼层信息（如 中楼层 (共11层)） |
| direction | 朝向 |
| type | 建筑类型（如 板楼） |
| district | 所属区县 |
| nearby | 周边区域 |
| community | 小区名称 |
| decoration | 装修情况 |
| elevator | 是否有电梯 |
| elevatorNum | 梯户比（如 一梯三户） |
| ownership | 产权类型 |
