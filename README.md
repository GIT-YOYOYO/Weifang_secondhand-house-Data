# 潍坊市二手房样本数据项目

基于潍坊市二手房样本数据的采集与分析项目。

## 项目目录

```
潍坊市二手房样本数据项目/
├── data/
│   └── sh_data.csv      # 二手房样本数据（18,340 条）
├── src/                 # 数据清洗与分析脚本
├── output/              # 分析结果、图表输出
├── main.py              # 项目入口脚本
└── README.md            # 项目说明
```

## 数据说明

`data/sh_data.csv` 共 18,340 条二手房记录，字段如下：

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
| ownership | 产权类型（如 商品房） |

## 使用方式

```bash
python main.py
```

分析脚本放置在 `src/`，结果统一输出到 `output/`。
