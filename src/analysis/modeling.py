"""价格预测建模与低价房源识别。

以房源属性（面积、户型、楼层、区县、装修等）为特征，分别训练
随机森林和岭回归模型预测单价，比较两种模型的拟合效果；再基于
最优模型在测试集上的残差，识别挂牌价明显低于预测值的潜在房源。
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..charts import apply_theme
from ..config import RANDOM_STATE, TABLE_DIR

# 数值特征：面积、户型结构、楼层与梯户比、电梯标记
NUMERIC_FEATURES = [
    "square", "rooms", "halls", "bathrooms", "total_floors",
    "households_per_elevator", "elevator_flag",
]
# 类别特征：需要独热编码后参与建模
CATEGORICAL_FEATURES = ["district", "type", "decoration", "floor_level", "direction_type"]

# 特征英文名 -> 中文显示名，用于特征重要性图
FEATURE_LABELS = {
    "square": "建筑面积",
    "rooms": "卧室数",
    "halls": "客厅数",
    "bathrooms": "卫生间数",
    "total_floors": "总楼层",
    "households_per_elevator": "户/梯",
    "elevator_flag": "电梯",
    "district": "区县",
    "type": "建筑类型",
    "decoration": "装修",
    "floor_level": "楼层位置",
    "direction_type": "朝向",
}

BARGAIN_THRESHOLD = 30.0  # 挂牌单价低于预测值 30% 以上视为潜在低价候选
BARGAIN_TOP_N = 15


def _build_pipeline(model) -> Pipeline:
    """构建“预处理 + 模型”的 sklearn 流水线。

    数值特征用中位数填补缺失并标准化；类别特征用众数填补后
    做独热编码（handle_unknown="ignore" 保证测试集出现新类别时不报错）。
    每个模型都使用独立的预处理分支，避免拟合状态相互影响。
    """
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def _pretty_feature(name: str) -> str:
    """将 sklearn 输出特征名（如 cat__district_寒亭区）转为中文可读名称。"""
    name = name.split("__", 1)[-1]
    for column, label in FEATURE_LABELS.items():
        if name == column:
            return label
        if name.startswith(column + "_"):
            return f"{label}={name[len(column) + 1:]}"
    return name


def _fig_model_compare(metrics: pd.DataFrame) -> go.Figure:
    """对比各模型在测试集上的 R² 得分。"""
    fig = px.bar(
        metrics, x="模型", y="R²", text=metrics["R²"].round(3),
        color="模型", color_discrete_sequence=["#2563EB", "#F59E0B"],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(title="R²（测试集）", range=[0, 1])
    fig.update_xaxes(title="")
    fig.update_layout(showlegend=False)
    return apply_theme(fig, "模型拟合效果对比", height=380)


def _fig_pred_scatter(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> go.Figure:
    """绘制测试集“实际单价-预测单价”散点图，虚线为理想预测线。

    点越贴近对角线说明预测越准确；最多抽样 3000 个点控制渲染开销。
    """
    sample_idx = np.random.default_rng(RANDOM_STATE).choice(
        len(y_true), size=min(3000, len(y_true)), replace=False
    )
    fig = px.scatter(
        x=y_true[sample_idx], y=y_pred[sample_idx], opacity=0.35,
        color_discrete_sequence=["#2563EB"],
    )
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))
    fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="理想预测",
                             line=dict(color="#EF4444", width=2, dash="dash")))
    fig.update_xaxes(title="实际单价（元/㎡）")
    fig.update_yaxes(title="预测单价（元/㎡）")
    fig.update_layout(showlegend=False)
    return apply_theme(fig, f"测试集预测效果（{label}）", height=460)


def _fig_importance(importance: pd.DataFrame) -> go.Figure:
    """绘制随机森林特征重要性 TOP12 条形图。"""
    data = importance.sort_values("重要性")
    fig = px.bar(
        data, x="重要性", y="特征", orientation="h",
        color="重要性", color_continuous_scale=["#BFDBFE", "#2563EB"],
    )
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(title="特征重要性")
    fig.update_yaxes(title="")
    return apply_theme(fig, "随机森林特征重要性 TOP12", height=480)


def _fig_residuals(residuals: np.ndarray) -> go.Figure:
    """绘制残差（实际 - 预测）分布，中位数偏离 0 说明模型存在系统偏差。"""
    fig = px.histogram(
        x=residuals, nbins=60, color_discrete_sequence=["#10B981"],
        labels={"x": "残差（实际 - 预测，元/㎡）"},
    )
    fig.add_vline(x=float(np.median(residuals)), line_dash="dash", line_color="#EF4444",
                  annotation_text=f"中位差 {np.median(residuals):.0f}", annotation_position="top")
    fig.update_yaxes(title="样本数")
    return apply_theme(fig, "测试集预测残差分布", height=420)


def _fig_district_error(test: pd.DataFrame) -> go.Figure:
    """按区县统计测试集平均绝对误差，识别模型误差较大的区域。"""
    error = (
        test.assign(绝对误差=test["残差"].abs())
        .groupby("district", as_index=False)["绝对误差"].mean()
        .sort_values("绝对误差")
    )
    fig = px.bar(
        error, x="绝对误差", y="district", orientation="h",
        text=error["绝对误差"].round(0).astype(int),
        color="绝对误差", color_continuous_scale=["#A7F3D0", "#10B981"],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(title="平均绝对误差（元/㎡）")
    fig.update_yaxes(title="")
    return apply_theme(fig, "各区县模型平均预测误差", height=420)


def analyze(df: pd.DataFrame) -> dict:
    """训练价格预测模型并识别潜在低价房源。

    流程：
    1. 划分 80% 训练集 / 20% 测试集，保留测试集原始索引用于回查房源信息；
    2. 分别训练随机森林和岭回归，用 R²、MAE、RMSE 评估；
    3. 选择 R² 最高的模型，输出特征重要性、残差和分区域误差；
    4. 在测试集中筛选“挂牌单价低于预测值 30% 以上”的房源作为低价候选，
       只使用测试集可避免模型对训练样本过拟合造成的高估。

    Args:
        df: 清洗后的房源数据。

    Returns:
        包含指标卡、图表、评估表和低价候选表的字典。
    """
    features = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = df["unit_price"]

    # 同时切分特征、目标与原始索引，便于后续把预测结果映射回房源
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        features, target, df.index,
        test_size=0.2, random_state=RANDOM_STATE,
    )

    algorithms = {
        "随机森林": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "岭回归": Ridge(alpha=1.0),
    }

    # 训练每个模型并在同一测试集上评估，保留预测结果与流水线供后续使用
    results = {}
    predictions = {}
    for name, algorithm in algorithms.items():
        pipeline = _build_pipeline(algorithm)
        pipeline.fit(x_train, y_train)
        pred = pipeline.predict(x_test)
        results[name] = {
            "R²": r2_score(y_test, pred),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        }
        predictions[name] = pred
        results[name]["pipeline"] = pipeline

    metrics = pd.DataFrame([
        {"模型": name, "R²": round(score["R²"], 4), "MAE": round(score["MAE"], 1),
         "RMSE": round(score["RMSE"], 1)}
        for name, score in results.items()
    ])
    metrics.to_csv(TABLE_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")

    best_name = metrics.sort_values("R²", ascending=False).iloc[0]["模型"]
    best_pipeline = results[best_name]["pipeline"]
    best_pred = predictions[best_name]

    # 特征重要性（优先使用随机森林）
    rf_pipeline = results["随机森林"]["pipeline"]
    feature_names = rf_pipeline.named_steps["pre"].get_feature_names_out()
    importance = pd.DataFrame({
        "特征": [_pretty_feature(name) for name in feature_names],
        "重要性": rf_pipeline.named_steps["model"].feature_importances_,
    }).nlargest(12, "重要性")
    importance.to_csv(TABLE_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")

    # 测试集预测明细与低价候选
    test = df.loc[idx_test, ["title", "community", "district", "square", "total_price", "unit_price"]].copy()
    test["预测单价"] = best_pred.round(1)
    test["残差"] = test["unit_price"] - test["预测单价"]
    # 折价率 = 1 - 挂牌价 / 预测价，正值表示挂牌价低于模型估值
    test["折价率"] = (1 - test["unit_price"] / test["预测单价"]) * 100

    # 折价达到阈值的房源作为候选池，再取折价最大的若干条展示
    bargain_pool = test[test["折价率"] >= BARGAIN_THRESHOLD]
    bargain_count = len(bargain_pool)
    bargains = bargain_pool.sort_values("折价率", ascending=False).head(BARGAIN_TOP_N).copy()
    bargain_display = bargains[
        ["title", "community", "district", "square", "total_price", "unit_price", "预测单价", "折价率"]
    ].copy()
    bargain_display["title"] = bargain_display["title"].str.slice(0, 28)
    bargain_display.columns = ["房源标题", "小区", "区县", "面积(㎡)", "总价(万)", "挂牌单价", "预测单价", "折价率(%)"]
    for col in ["面积(㎡)", "挂牌单价", "预测单价", "折价率(%)"]:
        bargain_display[col] = bargain_display[col].round(1)

    bargain_export = bargains[
        ["title", "community", "district", "square", "total_price", "unit_price", "预测单价", "折价率"]
    ].rename(columns={
        "title": "房源标题", "community": "小区", "district": "区县",
        "square": "面积", "total_price": "总价", "unit_price": "挂牌单价",
        "预测单价": "预测单价", "折价率": "折价率(%)",
    })
    bargain_export.to_csv(TABLE_DIR / "bargain_candidates.csv", index=False, encoding="utf-8-sig")

    figures = {
        "model_compare": _fig_model_compare(metrics),
        "pred_scatter": _fig_pred_scatter(y_test.to_numpy(), best_pred, best_name),
        "importance": _fig_importance(importance),
        "residuals": _fig_residuals(test["残差"].to_numpy()),
        "district_error": _fig_district_error(test),
    }

    mae_ratio = results[best_name]["MAE"] / df["unit_price"].median() * 100
    top_features = "、".join(importance.head(3)["特征"].tolist())

    metrics_cards = [
        {"label": "最佳模型", "value": best_name, "unit": ""},
        {"label": "测试集 R²", "value": f"{results[best_name]['R²']:.3f}", "unit": ""},
        {"label": "平均绝对误差", "value": f"{results[best_name]['MAE']:,.0f}", "unit": "元/㎡"},
        {"label": "误差占中位价", "value": f"{mae_ratio:.1f}", "unit": "%"},
        {"label": "低价候选", "value": f"{bargain_count}", "unit": f"套（折价≥{BARGAIN_THRESHOLD:.0f}%）"},
    ]

    insights = [
        f"{best_name}在测试集上表现最好，R² 为 {results[best_name]['R²']:.3f}，"
        f"平均绝对误差 {results[best_name]['MAE']:.0f} 元/㎡（约为中位单价的 {mae_ratio:.1f}%）。",
        f"对单价影响最大的特征依次为：{top_features}，"
        f"房屋面积与户型结构是定价的核心因素。",
        f"在未参与训练的测试集中，发现 {bargain_count} 套挂牌单价比模型预测低 "
        f"{BARGAIN_THRESHOLD:.0f}% 以上，可作为进一步核实的候选房源。",
        "低价候选仅基于挂牌数据与模型预测，未考虑房龄、朝向细节、楼层具体位置与议价空间，"
        "实际交易前需线下核实。",
    ]

    return {
        "title": "建模与应用",
        "metrics": metrics_cards,
        "figures": figures,
        "insights": insights,
        "tables": [
            {"title": "模型评估指标", "df": metrics},
            {"title": f"潜在低价房源 TOP{BARGAIN_TOP_N}", "df": bargain_display},
        ],
    }
