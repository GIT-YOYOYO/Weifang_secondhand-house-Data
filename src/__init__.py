"""潍坊市二手房数据分析包。

功能：
    提供从原始 CSV 到市场分析报告的一整套可复用模块。

实现方式：
    按职责拆分为配置（config）、数据读取（data_loader）、数据清洗
    （preprocessing）、图表样式（charts）、分析模块（analysis）、
    报告生成（report）和流程编排（pipeline），由 main.py 统一调用。
"""
