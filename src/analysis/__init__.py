"""分析模块集合。

包含四个互相独立、接口一致的分析模块：
    - market：市场概况
    - community：小区维度
    - factors：价格影响因素
    - modeling：建模与应用

每个模块都暴露 ``analyze(df) -> dict``，返回指标卡、Plotly 图表、
文字结论与数据表，供流水线和报告模块统一消费。
"""
