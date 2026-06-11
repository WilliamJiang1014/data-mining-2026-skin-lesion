from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skin_lesion_risk.demo.data_loader import (  # noqa: E402
    figure_paths,
    load_fairness,
    load_graph_ablation,
    load_all_sample_models,
    load_m5_summary,
    load_main_results,
    load_pad_adaptation,
    load_split_stats,
    load_table4,
)

MODEL_DISPLAY_NAMES = {
    "m0_constant": "M0 常数基线",
    "m0_lightgbm": "M0 LightGBM",
    "m1_cnn_baseline": "M1 CNN",
    "m2_monet_feature_baseline": "M2 MONET",
    "m3_transformer_baseline": "M3 Transformer",
    "m4_multimodal_fusion": "M4 多模态融合",
    "m5_lgke_gnn": "M5 LGKE-GNN",
}

FAIRNESS_METRIC_LABELS = {
    "auroc": "AUROC",
    "sensitivity": "灵敏度",
    "fnr": "漏诊率 (FNR)",
}

GRAPH_VARIANT_LABELS = {
    "lesion_only": "仅病灶图",
    "attributes": "属性节点",
    "patient": "患者节点",
    "visual_knn": "视觉 KNN 边",
    "prototypes": "原型节点",
}

GRAPH_METRIC_LABELS = {
    "pauc_pct": "pAUC (%)",
    "auprc_pct": "AUPRC (%)",
    "fnr_pct": "FNR (%)",
}

PAD_METRIC_LABELS = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "sensitivity": "灵敏度",
    "specificity": "特异度",
    "fnr": "漏诊率 (FNR)",
    "ece": "ECE",
}
SAMPLE_MODEL_ORDER = ["m5_lgke_gnn", "m0_lightgbm", "m4_multimodal_fusion"]
COMPARE_MODEL_ORDER = ["m0_lightgbm", "m4_multimodal_fusion", "m5_lgke_gnn"]


def _model_label(name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(name, name)


def _ordered_sample_model_keys(sample_models: dict[str, dict[str, object]]) -> list[str]:
    preferred = [key for key in SAMPLE_MODEL_ORDER if key in sample_models]
    remainder = sorted([key for key in sample_models if key not in SAMPLE_MODEL_ORDER], key=_model_label)
    return preferred + remainder


def _thresholds_from_metrics(metrics: dict[str, object]) -> tuple[float, float]:
    thresholds = metrics.get("thresholds", {}) if isinstance(metrics, dict) else {}
    th_hs = float(thresholds.get("sensitivity_at_least_0_90", 0.0))
    th_youden = float(thresholds.get("youden", 0.5))
    return th_hs, th_youden


def _decision(score: float, threshold: float) -> str:
    return "阳性" if score >= threshold else "阴性"


def _pct_delta(new: float | None, baseline: float | None) -> str:
    if new is None or baseline is None or pd.isna(new) or pd.isna(baseline):
        return "N/A"
    return f"{(new - baseline) * 100:.2f}pp"


def _to_pct(value: float | None, *, ratio: bool = True) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%" if ratio else f"{value:.2f}%"


def _format_subgroup_label(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text
    return text


def _render_ml_pipeline(figures: dict[str, Path]) -> None:
    if "ml_pipeline_png" not in figures:
        return
    st.markdown("### 实验流程图")
    left, center, right = st.columns([1, 6, 1])
    with center:
        st.image(str(figures["ml_pipeline_png"]), caption="实验流程图", use_container_width=True)


@st.cache_data(show_spinner=False)
def _load_all_data() -> dict[str, object]:
    return {
        "table4": load_table4(ROOT),
        "main_results": load_main_results(ROOT),
        "split_stats": load_split_stats(ROOT),
        "m5_summary": load_m5_summary(ROOT),
        "sample_models": load_all_sample_models(ROOT),
        "fairness": load_fairness(ROOT),
        "graph_ablation": load_graph_ablation(ROOT),
        "pad_adaptation": load_pad_adaptation(ROOT),
        "figures": figure_paths(ROOT),
    }


def _render_key_story_cards(data: dict[str, object]) -> None:
    table4 = data["table4"]
    m5 = data["m5_summary"]
    pad = data["pad_adaptation"]

    baseline_pauc = None
    best_name = "N/A"
    best_pauc = None
    if isinstance(table4, pd.DataFrame) and not table4.empty:
        candidates = table4[["Model", "pAUC_mean"]].dropna().copy()
        if not candidates.empty:
            idx = candidates["pAUC_mean"].idxmax()
            best_name = str(candidates.loc[idx, "Model"])
            best_pauc = float(candidates.loc[idx, "pAUC_mean"])
        m0_row = table4[table4["Model"].astype(str).str.contains("LightGBM", case=False, na=False)]
        if not m0_row.empty:
            baseline_pauc = float(m0_row.iloc[0]["pAUC_mean"])

    m5_pauc = float(m5.get("pauc_tpr80_pct", 0.0)) / 100 if isinstance(m5, dict) and m5 else None
    pad_auroc = None
    if isinstance(pad, pd.DataFrame) and not pad.empty:
        auroc_rows = pad[pad["metric"] == "auroc"]
        if not auroc_rows.empty:
            pad_auroc = float(auroc_rows.iloc[0]["mean"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("M0–M4 最佳 pAUC 模型", best_name)
    c2.metric("M0–M4 最佳 pAUC", _to_pct(best_pauc))
    c3.metric("M5 相对 M0 pAUC 提升", _pct_delta(m5_pauc, baseline_pauc))
    c4.metric("PAD 外部验证 AUROC", _to_pct(pad_auroc))


def _render_split_summary(split_stats: pd.DataFrame) -> None:
    st.markdown("### 数据与划分摘要（患者级）")
    if not isinstance(split_stats, pd.DataFrame) or split_stats.empty:
        st.warning("暂缺 split_stats 数据。")
        return

    isic_test = split_stats[split_stats["split"].astype(str).str.contains("ISIC fold") & split_stats["split"].astype(str).str.contains("test")]
    pad_external = split_stats[split_stats["split"].astype(str).str.contains("PAD external")]
    if not isic_test.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("ISIC 每折测试样本（均值）", f"{int(isic_test['n_samples'].mean()):,}")
        c2.metric("ISIC 每折测试阳性率（均值）", f"{isic_test['positive_rate'].mean() * 100:.3f}%")
        c3.metric("ISIC 每折测试患者数（均值）", f"{int(isic_test['n_patients'].mean()):,}")
    if not pad_external.empty:
        pad_row = pad_external.iloc[0]
        st.caption(
            f"PAD 外部集：样本 {int(pad_row['n_samples']):,}，阳性率 {float(pad_row['positive_rate']) * 100:.2f}%，"
            f"患者数 {int(pad_row['n_patients']):,}。"
        )
    st.info("所有结果采用患者级划分，避免同一患者样本泄漏到训练/测试。")


def _render_m5_summary(m5: dict[str, object]) -> None:
    st.markdown("### M5 LGKE-GNN（五折汇总）")
    rows = [
        ("pAUC (TPR≥0.8)", m5.get("pauc_tpr80_pct"), m5.get("pauc_tpr80_std_pct")),
        ("AUPRC", m5.get("auprc_pct"), m5.get("auprc_std_pct")),
        ("AUROC", m5.get("auroc_pct"), m5.get("auroc_std_pct")),
        ("Sensitivity", m5.get("sensitivity_pct"), None),
        ("Specificity", m5.get("specificity_pct"), None),
    ]
    summary = pd.DataFrame(
        [
            {
                "指标": label,
                "均值 (%)": f"{float(mean):.2f}" if mean is not None else "N/A",
                "标准差 (%)": f"{float(std):.2f}" if std is not None else "—",
            }
            for label, mean, std in rows
        ]
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)
    folds = m5.get("folds")
    if folds is not None:
        st.caption(f"汇总折数：{folds}")


def _render_overview(data: dict[str, object]) -> None:
    m5 = data["m5_summary"]
    figures = data["figures"]

    st.subheader("项目概览")
    st.markdown(
        "本页面展示课程项目的预计算实验结果，无需 GPU 即可浏览。"
        "内容覆盖模型对比、样本分数、公平性与图结构解释。"
    )
    st.info("仅用于课程研究演示，不可用于临床诊断。")

    if isinstance(m5, dict) and m5:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("M5 pAUC", _to_pct(m5.get("pauc_tpr80_pct"), ratio=False))
        c2.metric("M5 AUPRC", _to_pct(m5.get("auprc_pct"), ratio=False))
        c3.metric("M5 AUROC", _to_pct(m5.get("auroc_pct"), ratio=False))
        c4.metric("M5 Sens.", _to_pct(m5.get("sensitivity_pct"), ratio=False))

    _render_key_story_cards(data)
    _render_ml_pipeline(figures)


def _render_model_results(data: dict[str, object]) -> None:
    table4 = data["table4"]
    main_results = data["main_results"]
    split_stats = data["split_stats"]
    m5 = data["m5_summary"]
    figures = data["figures"]

    st.subheader("模型结果总览")

    if isinstance(table4, pd.DataFrame) and not table4.empty:
        st.markdown("### 主结果表（M0–M4）")
        st.dataframe(table4[["Model", "pAUC", "AUPRC", "AUROC", "Sens.", "Spec."]], use_container_width=True)
    else:
        st.warning("暂缺 M0–M4 模型对比表数据。")

    if isinstance(m5, dict) and m5:
        _render_m5_summary(m5)

    _render_split_summary(split_stats)

    chart_rows: list[dict[str, float | str]] = []
    if isinstance(table4, pd.DataFrame) and not table4.empty:
        for _, row in table4.iterrows():
            chart_rows.append(
                {
                    "模型": row["Model"],
                    "pAUC": float(row["pAUC_mean"]) * 100,
                    "AUPRC": float(row["AUPRC_mean"]) * 100,
                    "AUROC": float(row["AUROC_mean"]) * 100,
                }
            )
    if isinstance(m5, dict) and m5:
        chart_rows.append(
            {
                "模型": "M5 LGKE-GNN",
                "pAUC": float(m5.get("pauc_tpr80_pct", 0.0)),
                "AUPRC": float(m5.get("auprc_pct", 0.0)),
                "AUROC": float(m5.get("auroc_pct", 0.0)),
            }
        )

    if chart_rows:
        chart_df = pd.DataFrame(chart_rows)
        metric = st.selectbox("选择对比指标", ["pAUC", "AUPRC", "AUROC"], index=0)
        fig = px.bar(chart_df, x="模型", y=metric, title=f"{metric} 模型对比（%）")
        st.plotly_chart(fig, use_container_width=True)

    if isinstance(main_results, pd.DataFrame) and not main_results.empty:
        st.markdown("### 五折交叉验证波动")
        metric_map = {
            "pAUC": "pauc_tpr80",
            "AUPRC": "auprc",
            "AUROC": "auroc",
            "Sensitivity": "sensitivity",
            "Specificity": "specificity",
        }
        chosen_metric_label = st.selectbox("选择折间指标", list(metric_map.keys()), index=0)
        chosen_metric = metric_map[chosen_metric_label]
        default_models = [
            m for m in ["m0_lightgbm", "m4_multimodal_fusion", "m5_lgke_gnn"] if m in set(main_results["model_name"])
        ]
        model_options = sorted(main_results["model_name"].unique().tolist())
        selected_models = st.multiselect(
            "选择模型",
            model_options,
            default=default_models or model_options[:2],
            format_func=_model_label,
        )
        if selected_models:
            fold_df = main_results[main_results["model_name"].isin(selected_models)].copy()
            fold_df["model_label"] = fold_df["model_name"].map(_model_label)
            fold_df["value_pct"] = pd.to_numeric(fold_df[chosen_metric], errors="coerce") * 100
            fold_fig = px.line(
                fold_df,
                x="fold",
                y="value_pct",
                color="model_label",
                markers=True,
                title=f"{chosen_metric_label} 折间变化（%）",
            )
            st.plotly_chart(fold_fig, use_container_width=True)

    if "main_results_forest" in figures:
        st.image(str(figures["main_results_forest"]), caption="主结果森林图")


def _render_sample_inspection(data: dict[str, object]) -> None:
    sample_models = data["sample_models"]
    st.subheader("样本分数查看（多模型预测抽查）")
    if not isinstance(sample_models, dict) or not sample_models:
        st.warning("暂无样本预测数据。")
        return

    model_keys = _ordered_sample_model_keys(sample_models)
    default_key = "m5_lgke_gnn" if "m5_lgke_gnn" in model_keys else model_keys[0]
    selected_key = st.selectbox(
        "选择模型",
        model_keys,
        index=model_keys.index(default_key),
        format_func=lambda key: str(sample_models[key].get("label", _model_label(key))),
        key="sample_view_model",
    )

    selected_payload = sample_models[selected_key]
    predictions = selected_payload.get("predictions")
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        st.warning("当前模型暂无样本预测数据。")
        return

    id_sets = [
        set(payload["predictions"]["sample_id"].astype(str))
        for payload in sample_models.values()
        if isinstance(payload, dict)
        and isinstance(payload.get("predictions"), pd.DataFrame)
        and not payload["predictions"].empty
    ]
    common_ids = sorted(set.intersection(*id_sets)) if id_sets else []
    if not common_ids:
        st.warning("暂无可用于跨模型对比的共同样本。")
        return

    current_df = predictions[predictions["sample_id"].isin(common_ids)].copy()
    current_df["label"] = current_df["label"].astype(int)

    label_mode = st.segmented_control(
        "样本筛选",
        ["全部", "仅恶性", "仅良性"],
        default="全部",
        key="sample_label_filter",
    )
    if label_mode == "仅恶性":
        current_df = current_df[current_df["label"] == 1]
    elif label_mode == "仅良性":
        current_df = current_df[current_df["label"] == 0]

    sample_options = sorted(current_df["sample_id"].astype(str).tolist())
    if not sample_options:
        st.warning("当前筛选条件下没有可选样本。")
        return
    sample_id = st.selectbox("选择样本 ID", sample_options, key="sample_view_id")

    row = current_df[current_df["sample_id"] == sample_id].iloc[0]
    score = float(row["score"])
    label = int(row["label"])
    th_hs, th_youden = _thresholds_from_metrics(selected_payload.get("metrics", {}))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("风险分数", f"{score:.4f}")
    c2.metric("真实标签", "恶性(1)" if label == 1 else "良性(0)")
    c3.metric("高灵敏度阈值", f"{th_hs:.4f}")
    c4.metric("Youden 阈值", f"{th_youden:.4f}")

    st.markdown(
        f"- 高灵敏度判定：`{_decision(score, th_hs)}`\n"
        f"- Youden 判定：`{_decision(score, th_youden)}`"
    )

    hist = px.histogram(current_df, x="score", nbins=40, title=f"{sample_models[selected_key]['label']} 分数分布")
    hist.add_vline(x=th_hs, line_dash="dash", line_color="red", annotation_text="高灵敏度阈值")
    hist.add_vline(x=th_youden, line_dash="dot", line_color="green", annotation_text="Youden")
    hist.add_vline(x=score, line_dash="solid", line_color="black", annotation_text="当前样本")
    st.plotly_chart(hist, use_container_width=True)

    if len(model_keys) <= 1:
        return

    st.markdown("### 同一样本 · 跨模型对比")
    compare_rows: list[dict[str, object]] = []
    ordered_compare_keys = [key for key in COMPARE_MODEL_ORDER if key in model_keys] + [
        key for key in model_keys if key not in COMPARE_MODEL_ORDER
    ]
    for key in ordered_compare_keys:
        payload = sample_models[key]
        model_df = payload.get("predictions")
        if not isinstance(model_df, pd.DataFrame) or model_df.empty:
            continue
        matched = model_df[model_df["sample_id"] == sample_id]
        if matched.empty:
            continue
        model_score = float(matched.iloc[0]["score"])
        model_label = int(matched.iloc[0]["label"])
        model_th_hs, model_th_youden = _thresholds_from_metrics(payload.get("metrics", {}))
        compare_rows.append(
            {
                "模型Key": key,
                "模型": payload.get("label", _model_label(key)),
                "风险分数": round(model_score, 4),
                "真实标签": "恶性(1)" if model_label == 1 else "良性(0)",
                "高灵敏度判定": _decision(model_score, model_th_hs),
                "Youden 判定": _decision(model_score, model_th_youden),
            }
        )

    if not compare_rows:
        st.info("该样本暂无跨模型可对比结果。")
        return

    compare_df = pd.DataFrame(compare_rows)
    baseline_row = compare_df[compare_df["模型Key"] == "m0_lightgbm"]
    baseline_score = float(baseline_row.iloc[0]["风险分数"]) if not baseline_row.empty else None
    compare_df["相对M0差值"] = compare_df["风险分数"].apply(
        lambda x: round(float(x) - baseline_score, 4) if baseline_score is not None else None
    )
    compare_df["分数排名"] = compare_df["风险分数"].rank(ascending=False, method="min").astype(int)
    st.dataframe(
        compare_df[["模型", "风险分数", "分数排名", "相对M0差值", "真实标签", "高灵敏度判定", "Youden 判定"]],
        use_container_width=True,
        hide_index=True,
    )

    compare_fig = px.bar(compare_df, x="模型", y="风险分数", title="同一样本跨模型风险分数对比")
    st.plotly_chart(compare_fig, use_container_width=True)


def _render_fairness_and_graph(data: dict[str, object]) -> None:
    fairness = data["fairness"]
    ablation = data["graph_ablation"]
    pad = data["pad_adaptation"]
    figures = data["figures"]

    st.subheader("公平性与图结构")

    st.markdown("### ISIC 子群公平性")
    if isinstance(fairness, pd.DataFrame) and not fairness.empty:
        group_options = fairness["group_variable"].unique().tolist()
        group = st.selectbox("分组变量", group_options)
        sub = fairness[fairness["group_variable"] == group].copy()
        for col in ["auroc", "sensitivity", "fnr"]:
            sub[col] = (pd.to_numeric(sub[col], errors="coerce") * 100).round(2)

        metric_options = list(FAIRNESS_METRIC_LABELS.keys())
        plot_col = st.selectbox(
            "公平性柱状图指标",
            metric_options,
            index=0,
            format_func=lambda key: FAIRNESS_METRIC_LABELS[key],
        )
        risk_mask = (
            pd.to_numeric(sub["n_positive"], errors="coerce").fillna(0) < 10
        ) | pd.to_numeric(sub[plot_col], errors="coerce").isna()
        risk_subgroups = [
            _format_subgroup_label(x) for x in sub.loc[risk_mask, "subgroup"].tolist()
        ]
        if risk_subgroups:
            st.warning(
                f"以下子群在当前指标（{FAIRNESS_METRIC_LABELS[plot_col]}）下样本偏少或结果缺失，需谨慎解释："
                f"{', '.join(risk_subgroups)}"
            )
        display_cols = {
            "subgroup": "子群",
            "n_samples": "样本数",
            "n_positive": "阳性数",
            "auroc": "AUROC (%)",
            "sensitivity": "灵敏度 (%)",
            "fnr": "漏诊率 (%)",
            "notes": "备注",
        }
        st.dataframe(
            sub[[c for c in display_cols if c in sub.columns]].rename(columns=display_cols),
            use_container_width=True,
            hide_index=True,
        )

        plot_df = sub.dropna(subset=[plot_col]).copy()
        if not plot_df.empty:
            fair_fig = px.bar(
                plot_df,
                x="subgroup",
                y=plot_col,
                title=f"{group} · {FAIRNESS_METRIC_LABELS[plot_col]}（%）",
            )
            st.plotly_chart(fair_fig, use_container_width=True)
        st.caption("低阳性样本子群仅供探索性参考，不作稳定结论。")
    else:
        st.warning("暂无公平性子群数据。")

    st.markdown("### 图结构消融")
    if isinstance(ablation, pd.DataFrame) and not ablation.empty:
        show_df = ablation[["variant", "n_folds", "pauc_tpr80", "auprc", "fnr"]].copy()
        show_df["variant"] = show_df["variant"].map(lambda v: GRAPH_VARIANT_LABELS.get(v, v))
        show_df = show_df.rename(
            columns={
                "variant": "图结构变体",
                "n_folds": "折数",
                "pauc_tpr80": "pAUC",
                "auprc": "AUPRC",
                "fnr": "FNR",
            }
        )
        st.dataframe(show_df, use_container_width=True, hide_index=True)

        plot_df = ablation.copy()
        plot_df["variant_label"] = plot_df["variant"].map(lambda v: GRAPH_VARIANT_LABELS.get(v, v))
        plot_df["pauc_pct"] = pd.to_numeric(plot_df["pauc_tpr80_mean"], errors="coerce") * 100
        plot_df["auprc_pct"] = pd.to_numeric(plot_df["auprc_mean"], errors="coerce") * 100
        plot_df["fnr_pct"] = pd.to_numeric(plot_df["fnr_mean"], errors="coerce") * 100
        metric_options = list(GRAPH_METRIC_LABELS.keys())
        metric = st.selectbox(
            "消融对比指标",
            metric_options,
            index=0,
            format_func=lambda key: GRAPH_METRIC_LABELS[key],
        )
        abl_fig = px.bar(
            plot_df,
            x="variant_label",
            y=metric,
            title=f"图结构消融 · {GRAPH_METRIC_LABELS[metric]}",
        )
        st.plotly_chart(abl_fig, use_container_width=True)
    else:
        st.warning("暂无图结构消融数据。")

    if "graph_ablation" in figures:
        st.image(str(figures["graph_ablation"]), caption="图结构消融图")

    st.markdown("### PAD 外部域迁移（含目标域适配）")
    if isinstance(pad, pd.DataFrame) and not pad.empty:
        pad_show = pad.copy()
        pad_show["mean_pct"] = (pad_show["mean"] * 100).round(2)
        pad_show["std_pct"] = (pad_show["std"] * 100).round(2)
        pad_display = pad_show[["metric", "mean_pct", "std_pct"]].copy()
        pad_display["metric"] = pad_display["metric"].map(lambda m: PAD_METRIC_LABELS.get(m, m))
        pad_display = pad_display.rename(
            columns={"metric": "指标", "mean_pct": "均值 (%)", "std_pct": "标准差 (%)"}
        )
        st.dataframe(pad_display, use_container_width=True, hide_index=True)

        focus_metrics = ["auroc", "auprc", "sensitivity", "specificity", "fnr", "ece"]
        pad_plot = pad_show[pad_show["metric"].isin(focus_metrics)].copy()
        if not pad_plot.empty:
            pad_plot["metric_label"] = pad_plot["metric"].map(lambda m: PAD_METRIC_LABELS.get(m, m))
            pad_fig = px.bar(pad_plot, x="metric_label", y="mean_pct", title="PAD 外部验证 · 关键指标（%）")
            st.plotly_chart(pad_fig, use_container_width=True)
    else:
        st.warning("暂无 PAD 外部验证数据。")

    if "pad_domain_shift" in figures:
        st.image(str(figures["pad_domain_shift"]), caption="PAD 域迁移结果图")


def main() -> None:
    st.set_page_config(page_title="皮肤病变风险筛查 Demo", layout="wide")
    st.title("多模态皮肤病变风险筛查与公平性评估 Demo")
    st.caption("数据来源：ISIC 2024 主实验结果与 M5 LGKE-GNN 补充分析")

    data = _load_all_data()
    tabs = st.tabs(["概览", "模型结果", "样本查看", "公平性与图结构"])
    with tabs[0]:
        _render_overview(data)
    with tabs[1]:
        _render_model_results(data)
    with tabs[2]:
        _render_sample_inspection(data)
    with tabs[3]:
        _render_fairness_and_graph(data)


if __name__ == "__main__":
    main()

