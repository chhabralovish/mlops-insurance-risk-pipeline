import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models import NUMERIC_FEATURES, CATEGORICAL_FEATURES, load_and_split
from evaluate import evaluate_model, plot_roc_curve, compare_models
from drift import detect_drift, generate_drift_data, get_drift_summary

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLOps Dashboard — Insurance Risk",
    page_icon="🏭",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #2E86AB;
        border-radius: 12px; padding: 20px; text-align: center;
    }
    .metric-value { font-size: 2.5em; font-weight: bold; color: #2E86AB; }
    .metric-label { font-size: 0.85em; color: #aaaaaa; margin-top: 4px; }
    .good { color: #4CAF50 !important; }
    .warn { color: #FFC107 !important; }
    .bad  { color: #FF6B6B !important; }
</style>
""", unsafe_allow_html=True)

st.title("🏭 MLOps Dashboard — Insurance Risk Scoring")
st.caption("Model performance monitoring, experiment comparison and data drift detection")
st.divider()

# ── Load Models ───────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "insurance_risk.csv")

@st.cache_resource
def load_models():
    models = {}
    for name in ["random_forest", "xgboost", "lightgbm"]:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

models = load_models()
df = load_data()

if not models:
    st.error("No trained models found. Run `python train.py` first.")
    st.code("python train.py --tune")
    st.stop()

if df is None:
    st.error("No data found. Run `python data/generate.py` first.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Model Overview",
    "🔬 Experiment Comparison",
    "📈 Drift Detection",
    "🎯 Live Prediction",
    "📋 Data Analysis"
])

# ── TAB 1: Model Overview ─────────────────────────────────────────────────────
with tab1:
    st.subheader("Trained Models Overview")

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split(DATA_PATH)

    results = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_val, y_val)
        results[name] = {"metrics": metrics}

    comparison = compare_models(results)

    # Metrics cards
    best = comparison.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value good'>{best['roc_auc']:.4f}</div>
            <div class='metric-label'>Best ROC AUC</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{best['f1_score']:.4f}</div>
            <div class='metric-label'>Best F1 Score</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{best['model'].replace('_',' ').title()}</div>
            <div class='metric-label'>Best Model</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{len(models)}</div>
            <div class='metric-label'>Models Trained</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("Model Comparison Table")
    display_cols = ["model", "roc_auc", "f1_score", "precision", "recall",
                    "accuracy", "specificity"]
    st.dataframe(
        comparison[[c for c in display_cols if c in comparison.columns]],
        use_container_width=True
    )

    # ROC curves
    st.divider()
    st.subheader("ROC Curves — All Models")
    fig = go.Figure()
    colors = ["#2E86AB", "#F18F01", "#C73E1D"]
    for i, (name, model) in enumerate(models.items()):
        from sklearn.metrics import roc_curve, roc_auc_score
        y_proba = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        auc = roc_auc_score(y_val, y_proba)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, name=f"{name} (AUC={auc:.4f})",
            line=dict(color=colors[i], width=2)
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="Random",
        line=dict(color="gray", dash="dash")
    ))
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_dark", height=450
    )
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: Experiment Comparison ──────────────────────────────────────────────
with tab2:
    st.subheader("Experiment Comparison")

    # Check for MLflow comparison CSV
    comp_path = os.path.join(os.path.dirname(__file__), "..", "model_comparison.csv")
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        st.dataframe(comp_df, use_container_width=True)
    else:
        st.info("Run train.py to generate experiment comparison data.")
        comp_df = comparison

    # Metrics comparison bars
    metrics_to_plot = ["roc_auc", "f1_score", "precision", "recall", "accuracy"]
    available = [m for m in metrics_to_plot if m in comparison.columns]

    fig = make_subplots(rows=1, cols=len(available),
                        subplot_titles=available)
    colors = ["#2E86AB", "#F18F01", "#C73E1D"]

    for i, metric in enumerate(available):
        for j, row in comparison.iterrows():
            fig.add_trace(go.Bar(
                x=[row["model"]], y=[row[metric]],
                marker_color=colors[j % 3],
                name=row["model"],
                showlegend=(i == 0)
            ), row=1, col=i+1)

    fig.update_layout(
        template="plotly_dark", height=400,
        title="All Models — All Metrics Comparison",
        barmode="group"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.divider()
    st.subheader("Feature Importance Comparison")
    sel_model = st.selectbox("Select model", list(models.keys()))
    if sel_model in models:
        try:
            clf = models[sel_model].named_steps["model"]
            preprocessor = models[sel_model].named_steps["preprocessor"]
            cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(
                CATEGORICAL_FEATURES
            )
            all_features = NUMERIC_FEATURES + list(cat_features)
            importances = clf.feature_importances_
            n = min(15, len(importances))
            indices = np.argsort(importances)[-n:]
            feat_labels = [all_features[i] if i < len(all_features)
                          else f"feat_{i}" for i in indices]

            fig_fi = px.bar(
                x=importances[indices], y=feat_labels,
                orientation='h', title=f"Feature Importance — {sel_model}",
                color=importances[indices], color_continuous_scale="Blues",
                template="plotly_dark"
            )
            fig_fi.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_fi, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not plot feature importance: {e}")

# ── TAB 3: Drift Detection ────────────────────────────────────────────────────
with tab3:
    st.subheader("Data Drift Detection")

    drift_magnitude = st.slider(
        "Simulate drift magnitude", 0.0, 1.0, 0.3, 0.1
    )

    if st.button("Run Drift Analysis", type="primary"):
        with st.spinner("Analysing drift..."):
            reference = df.sample(frac=0.5, random_state=42)
            drifted = generate_drift_data(reference, drift_magnitude)
            drift_df = detect_drift(reference, drifted, NUMERIC_FEATURES)
            summary = get_drift_summary(drift_df)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            color = "bad" if summary["high_drift"] > 0 else "good"
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value {color}'>{summary['high_drift']}</div>
                <div class='metric-label'>High Drift Features</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value warn'>{summary['medium_drift']}</div>
                <div class='metric-label'>Medium Drift Features</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value good'>{summary['no_drift']}</div>
                <div class='metric-label'>Stable Features</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{summary['max_psi']:.3f}</div>
                <div class='metric-label'>Max PSI ({summary['max_psi_feature']})</div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.dataframe(drift_df, use_container_width=True)

        # PSI bar chart
        fig_psi = px.bar(
            drift_df.head(12), x="feature", y="psi",
            color="severity",
            color_discrete_map={"High": "#FF6B6B", "Medium": "#FFC107", "Low": "#4CAF50"},
            title="PSI Score per Feature",
            template="plotly_dark"
        )
        fig_psi.add_hline(y=0.1, line_dash="dash", line_color="yellow",
                          annotation_text="Moderate threshold (0.1)")
        fig_psi.add_hline(y=0.2, line_dash="dash", line_color="red",
                          annotation_text="High threshold (0.2)")
        fig_psi.update_layout(height=400)
        st.plotly_chart(fig_psi, use_container_width=True)

# ── TAB 4: Live Prediction ────────────────────────────────────────────────────
with tab4:
    st.subheader("Live Risk Prediction")
    st.info("Enter customer details to get a real-time risk score from the trained model.")

    best_model_name = comparison.iloc[0]["model"]
    best_model = models[best_model_name]

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 85, 45)
        annual_income = st.number_input("Annual Income", 15000, 500000, 75000, 5000)
        credit_score = st.slider("Credit Score", 300, 850, 680)
        num_dependents = st.slider("Dependents", 0, 5, 2)
        debt_to_income = st.slider("Debt to Income Ratio", 0.0, 1.0, 0.3, 0.05)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status",
                                       ["Single", "Married", "Divorced", "Widowed"])
        policy_type = st.selectbox("Policy Type",
                                    ["Health", "Auto", "Life", "Home", "Travel"])
        years_with_insurer = st.slider("Years with Insurer", 0, 30, 5)
        previous_claims = st.slider("Previous Claims", 0, 10, 1)

    with col3:
        bmi = st.slider("BMI", 15.0, 50.0, 26.5, 0.5)
        smoker = st.selectbox("Smoker", [0, 1], format_func=lambda x: "Yes" if x else "No")
        exercise_frequency = st.selectbox("Exercise Frequency",
                                           ["Never", "Rarely", "Sometimes", "Often", "Daily"])
        chronic_conditions = st.slider("Chronic Conditions", 0, 3, 0)
        claim_amount_history = st.number_input("Claim Amount History", 0, 500000, 10000, 5000)

    if st.button("🎯 Predict Risk Score", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            "age": age, "gender": gender, "marital_status": marital_status,
            "annual_income": annual_income, "credit_score": credit_score,
            "num_dependents": num_dependents, "debt_to_income": debt_to_income,
            "years_with_insurer": years_with_insurer, "previous_claims": previous_claims,
            "claim_amount_history": claim_amount_history, "policy_type": policy_type,
            "coverage_amount": 500000, "policy_duration_years": 10,
            "num_policies": 1, "bmi": bmi, "smoker": smoker,
            "exercise_frequency": exercise_frequency,
            "chronic_conditions": chronic_conditions
        }])

        proba = best_model.predict_proba(input_data)[0]
        risk_score = proba[1]

        if risk_score < 0.3:
            color, label, rec = "#4CAF50", "Low Risk", "Standard premium."
        elif risk_score < 0.5:
            color, label, rec = "#FFC107", "Moderate Risk", "Apply moderate loading factor."
        elif risk_score < 0.7:
            color, label, rec = "#FF9800", "High Risk", "High-risk premium loading required."
        else:
            color, label, rec = "#FF6B6B", "Very High Risk", "Escalate for manual underwriter review."

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(risk_score * 100, 1),
                title={"text": f"Risk Score — {label}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 30], "color": "#1a3a1a"},
                        {"range": [30, 50], "color": "#3a3a1a"},
                        {"range": [50, 70], "color": "#3a2a1a"},
                        {"range": [70, 100], "color": "#3a1a1a"},
                    ]
                }
            ))
            fig_gauge.update_layout(
                template="plotly_dark", height=350
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        with col2:
            st.markdown(f"### {label}")
            st.markdown(f"**Risk Score:** {risk_score:.4f}")
            st.markdown(f"**Model Used:** {best_model_name}")
            st.markdown(f"**Recommendation:** {rec}")
            st.progress(risk_score)

# ── TAB 5: Data Analysis ──────────────────────────────────────────────────────
with tab5:
    st.subheader("Dataset Analysis")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Records", f"{len(df):,}")
    with c2:
        st.metric("High Risk Rate", f"{df['high_risk'].mean():.1%}")
    with c3:
        st.metric("Features", len(NUMERIC_FEATURES + CATEGORICAL_FEATURES))

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(df, x="risk_score", nbins=50,
                            title="Risk Score Distribution",
                            template="plotly_dark",
                            color_discrete_sequence=["#2E86AB"])
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.box(df, x="policy_type", y="risk_score",
                      title="Risk Score by Policy Type",
                      template="plotly_dark",
                      color="policy_type")
        st.plotly_chart(fig2, use_container_width=True)

    feat = st.selectbox("Explore feature", NUMERIC_FEATURES)
    col1, col2 = st.columns(2)
    with col1:
        fig3 = px.histogram(df, x=feat, color="high_risk",
                            title=f"{feat} Distribution by Risk",
                            template="plotly_dark", barmode="overlay",
                            color_discrete_map={0: "#2E86AB", 1: "#FF6B6B"})
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        fig4 = px.scatter(df.sample(500), x=feat, y="risk_score",
                          color="high_risk", title=f"{feat} vs Risk Score",
                          template="plotly_dark",
                          color_discrete_map={0: "#2E86AB", 1: "#FF6B6B"})
        st.plotly_chart(fig4, use_container_width=True)
