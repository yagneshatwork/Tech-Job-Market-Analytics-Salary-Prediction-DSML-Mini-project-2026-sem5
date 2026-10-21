"""
app.py  –  Tech Job Market Analytics & Salary Prediction
Streamlit web application with 3 tabs:
1. 📊 Market Explorer (interactive drill-downs, filters, and KPIs)
2. 🔮 Salary Predictor (point estimate, predicted ± test MAE range, band classification, role-country median, what-if curve)
3. 🧠 Model Insights (model benchmarks, bias-variance tuning, feature importance, confusion matrix, gradient descent loss, PCA, limitations)

STRICTLY COMPLIANT:
- Allowed algorithms: Linear/Ridge/Lasso, SGDRegressor, Decision Tree, Random Forest, KNN, Naive Bayes
- Allowed metrics: MAE, RMSE, R2, accuracy, precision, recall, F1, confusion matrix
- NO LightGBM, XGBoost, boosting, SHAP, cross-validation, GridSearchCV/RandomizedSearchCV, MAPE, residual-quantile intervals.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from joblib import load

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tech Job Market Analytics & Salary Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── File Paths ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "jobs.csv"
META_PATH = ROOT / "models" / "metadata.json"
REG_PATH  = ROOT / "models" / "salary_pipeline.joblib"
BAND_PATH = ROOT / "models" / "band_pipeline.joblib"
FIG_DIR   = ROOT / "reports" / "figures"
COMP_CSV  = ROOT / "reports" / "model_comparison.csv"

# ── Visual Styling (CSS) ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    height: 48px; padding: 0 24px;
    border-radius: 8px 8px 0 0;
    font-weight: 600;
}
.kpi-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155; border-radius: 12px;
    padding: 20px 22px; text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
}
.kpi-label { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }
.kpi-value { color: #f8fafc; font-size: 2rem; font-weight: 700; margin: 6px 0; }
.kpi-sub   { color: #64748b; font-size: 0.8rem; }
.band-badge {
    display: inline-block; padding: 6px 20px;
    border-radius: 9999px; font-weight: 700; font-size: 1.15rem;
}
.band-Low  { background:#fef3c7; color:#92400e; border: 1px solid #f59e0b; }
.band-Mid  { background:#dbeafe; color:#1e40af; border: 1px solid #3b82f6; }
.band-High { background:#d1fae5; color:#065f46; border: 1px solid #10b981; }
.synthetic-banner {
    background: linear-gradient(90deg, rgba(124, 58, 237, 0.15), rgba(37, 99, 235, 0.15));
    border: 1px solid rgba(124, 58, 237, 0.35);
    border-radius: 10px; padding: 12px 20px;
    color: #c4b5fd; font-size: 0.9rem;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ── Synthetic Dataset Banner ─────────────────────────────────────────────────
st.markdown("""
<div class="synthetic-banner">
⚠️ <strong>Notice: Synthetic Dataset</strong> – The underlying dataset (90,000 records) exhibits properties of synthetic generation.
All salary forecasts and market comparisons illustrate data science principles and should not be used as real compensation advice.
</div>
""", unsafe_allow_html=True)

st.markdown("# 💼 Tech Job Market Analytics & Salary Prediction")

# ── Loaders ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)

@st.cache_resource(show_spinner="Loading trained models…")
def load_models():
    missing = [p for p in [REG_PATH, BAND_PATH, META_PATH] if not p.exists()]
    if missing:
        return None, None, None
    reg  = load(REG_PATH)
    band = load(BAND_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    return reg, band, meta

df = load_data()
reg_pipe, band_pipe, meta = load_models()

MODELS_OK = (reg_pipe is not None) and (band_pipe is not None) and (meta is not None)
DATA_OK   = (df is not None)

def show_models_missing():
    st.error(
        "⚠️ Model artifacts not found. Please run `salary_project.ipynb` to generate `models/` artifacts:\n\n"
        "```powershell\n.\\.venv\\Scripts\\jupyter nbconvert --to notebook --execute --inplace salary_project.ipynb\n```"
    )

PLOT_THEME = dict(template="plotly_dark")

def apply_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Market Explorer", "🔮 Salary Predictor", "🧠 Model Insights"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1: Market Explorer
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    if not DATA_OK:
        st.error("Dataset not found at `data/jobs.csv`.")
        st.stop()

    st.markdown("### Market Explorer")
    st.caption("Filter by geographic, industry, and role dimensions to analyze compensation patterns.")

    # Sidebar Filters
    with st.sidebar:
        st.markdown("### 🔍 Filter Market Data")
        all_countries = sorted(df["country"].unique())
        sel_countries = st.multiselect("Country", all_countries, default=all_countries)

        all_roles = sorted(df["job_role"].unique())
        sel_roles = st.multiselect("Job Role", all_roles, default=all_roles)

        all_specs = sorted(df["ai_specialization"].unique())
        sel_specs = st.multiselect("AI Specialization", all_specs, default=all_specs)

        all_exp = ["Entry", "Mid", "Senior", "Lead"]
        sel_exp = st.multiselect("Experience Level", all_exp, default=all_exp)

        yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
        sel_years = st.slider("Year", yr_min, yr_max, (yr_min, yr_max))

    filtered = df[
        df["country"].isin(sel_countries) &
        df["job_role"].isin(sel_roles) &
        df["ai_specialization"].isin(sel_specs) &
        df["experience_level"].isin(sel_exp) &
        df["year"].between(sel_years[0], sel_years[1])
    ]

    # KPI Summary Cards
    c1, c2, c3, c4 = st.columns(4)
    avg_sal  = filtered["salary_usd"].mean() if len(filtered) else 0
    med_sal  = filtered["salary_usd"].median() if len(filtered) else 0
    row_cnt  = len(filtered)
    avg_exp  = filtered["experience_years"].mean() if len(filtered) else 0

    c1.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Average Salary</div>
      <div class="kpi-value">${avg_sal:,.0f}</div>
      <div class="kpi-sub">Mean compensation</div>
    </div>""", unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Median Salary</div>
      <div class="kpi-value">${med_sal:,.0f}</div>
      <div class="kpi-sub">50th percentile</div>
    </div>""", unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Matching Jobs</div>
      <div class="kpi-value">{row_cnt:,}</div>
      <div class="kpi-sub">{filtered['country'].nunique()} countries</div>
    </div>""", unsafe_allow_html=True)

    c4.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Avg Experience</div>
      <div class="kpi-value">{avg_exp:.1f} yrs</div>
      <div class="kpi-sub">Candidate tenure</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualizations Row 1
    col_a, col_b = st.columns(2)
    with col_a:
        role_sal = filtered.groupby("job_role")["salary_usd"].median().sort_values().reset_index()
        fig_role = px.bar(
            role_sal, x="salary_usd", y="job_role", orientation="h",
            title="Median Salary by Job Role",
            labels={"salary_usd": "Median Salary ($)", "job_role": ""},
            color="salary_usd", color_continuous_scale="Blues",
            **PLOT_THEME
        )
        apply_theme(fig_role)
        fig_role.update_layout(coloraxis_showscale=False, height=350)
        fig_role.update_xaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig_role, use_container_width=True)

    with col_b:
        ctry_sal = filtered.groupby("country")["salary_usd"].median().sort_values().reset_index()
        fig_ctry = px.bar(
            ctry_sal, x="salary_usd", y="country", orientation="h",
            title="Median Salary by Country",
            labels={"salary_usd": "Median Salary ($)", "country": ""},
            color="salary_usd", color_continuous_scale="Teal",
            **PLOT_THEME
        )
        apply_theme(fig_ctry)
        fig_ctry.update_layout(coloraxis_showscale=False, height=350)
        fig_ctry.update_xaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig_ctry, use_container_width=True)

    # Visualizations Row 2
    col_c, col_d = st.columns(2)
    with col_c:
        fig_box = px.box(
            filtered, x="experience_level", y="salary_usd",
            category_orders={"experience_level": ["Entry", "Mid", "Senior", "Lead"]},
            title="Salary Distribution by Experience Level",
            labels={"salary_usd": "Salary ($)", "experience_level": ""},
            color="experience_level",
            **PLOT_THEME
        )
        apply_theme(fig_box)
        fig_box.update_layout(showlegend=False, height=350)
        fig_box.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig_box, use_container_width=True)

    with col_d:
        yr_stat = filtered.groupby("year")["salary_usd"].agg(Mean="mean", Median="median").reset_index()
        fig_trend = go.Figure()
        apply_theme(fig_trend)
        fig_trend.add_trace(go.Scatter(x=yr_stat["year"], y=yr_stat["Mean"], mode="lines+markers", name="Mean", line=dict(color="#60a5fa", width=2.5)))
        fig_trend.add_trace(go.Scatter(x=yr_stat["year"], y=yr_stat["Median"], mode="lines+markers", name="Median", line=dict(color="#34d399", width=2.5)))
        fig_trend.update_layout(title="Annual Salary Trend (2020–2026)", xaxis_title="Year", yaxis_title="Salary ($)", height=350)
        fig_trend.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig_trend, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2: Salary Predictor
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🔮 Interactive Salary Predictor")
    st.caption("Provide candidate and position details to predict salary, range, and tertile band.")

    if not MODELS_OK:
        show_models_missing()
        st.stop()

    cats = meta["categories"]
    num_ranges = meta["numeric_ranges"]
    test_mae = float(meta.get("test_mae", 9302.0))
    CORE_CAT = meta["core_categorical"]
    CORE_NUM = meta["core_numerical"]
    CORE = CORE_CAT + CORE_NUM

    with st.form("salary_prediction_form"):
        st.markdown("#### Candidate & Position Attributes")
        c1, c2, c3 = st.columns(3)

        with c1:
            country = st.selectbox("Country", cats["country"], index=cats["country"].index("USA") if "USA" in cats["country"] else 0)
            job_role = st.selectbox("Job Role", cats["job_role"], index=cats["job_role"].index("Machine Learning Engineer") if "Machine Learning Engineer" in cats["job_role"] else 0)
            ai_spec = st.selectbox("AI Specialization", cats["ai_specialization"])

        with c2:
            exp_level = st.selectbox("Experience Level", cats["experience_level"], index=cats["experience_level"].index("Mid") if "Mid" in cats["experience_level"] else 0)
            education = st.selectbox("Education Required", cats["education_required"], index=cats["education_required"].index("Master") if "Master" in cats["education_required"] else 0)
            industry = st.selectbox("Industry", cats["industry"], index=cats["industry"].index("Tech") if "Tech" in cats["industry"] else 0)

        with c3:
            company_size = st.selectbox("Company Size", cats["company_size"], index=cats["company_size"].index("Large") if "Large" in cats["company_size"] else 0)
            work_mode = st.selectbox("Work Mode", cats["work_mode"], index=cats["work_mode"].index("Remote") if "Remote" in cats["work_mode"] else 0)
            exp_years = st.slider("Experience Years",
                                  int(num_ranges["experience_years"]["min"]),
                                  int(num_ranges["experience_years"]["max"]),
                                  int(num_ranges["experience_years"]["median"]))
            year = st.selectbox("Market Year", [2026, 2025, 2024, 2023, 2022, 2021, 2020])

        submit_btn = st.form_submit_button("⚡ Predict Salary", use_container_width=True, type="primary")

    if submit_btn:
        input_record = {
            "country": country,
            "job_role": job_role,
            "ai_specialization": ai_spec,
            "experience_level": exp_level,
            "education_required": education,
            "industry": industry,
            "company_size": company_size,
            "work_mode": work_mode,
            "experience_years": exp_years,
            "year": int(year),
        }
        input_df = pd.DataFrame([input_record])

        # Regression prediction
        pred_log = reg_pipe.predict(input_df[CORE])[0]
        pred_usd = float(np.expm1(pred_log))

        # Range: predicted ± test MAE
        low_usd  = max(0.0, pred_usd - test_mae)
        high_usd = pred_usd + test_mae

        # Band classification (Low / Mid / High)
        pred_band = str(band_pipe.predict(input_df[CORE])[0])

        # Role-Country Median
        rc_lookup = {f"{r['country']}|||{r['job_role']}": r["median_salary"] for r in meta.get("role_country_medians", [])}
        rc_median = rc_lookup.get(f"{country}|||{job_role}")

        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")

        r1, r2, r3, r4 = st.columns(4)

        r1.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Predicted Salary</div>
          <div class="kpi-value">${pred_usd:,.0f}</div>
          <div class="kpi-sub">Point estimate (USD)</div>
        </div>""", unsafe_allow_html=True)

        r2.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Predicted ± Test MAE</div>
          <div class="kpi-value">${low_usd:,.0f} – ${high_usd:,.0f}</div>
          <div class="kpi-sub">±${test_mae:,.0f} MAE range</div>
        </div>""", unsafe_allow_html=True)

        band_class = f"band-{pred_band}"
        r3.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Salary Band</div>
          <div class="kpi-value"><span class="band-badge {band_class}">{pred_band}</span></div>
          <div class="kpi-sub">Tertile segment</div>
        </div>""", unsafe_allow_html=True)

        rc_text = f"${rc_median:,.0f}" if rc_median is not None else "N/A"
        r4.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Role-Country Median</div>
          <div class="kpi-value">{rc_text}</div>
          <div class="kpi-sub">{job_role} in {country}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # What-If Sensitivity Chart: Salary vs Experience Years
        st.markdown("#### 📈 What-If Analysis: Salary vs Experience Years")
        st.caption("All other candidate and company attributes held constant at the selected profile.")

        min_exp = int(num_ranges["experience_years"]["min"])
        max_exp = int(num_ranges["experience_years"]["max"])
        whatif_years = list(range(min_exp, max_exp + 1))

        whatif_rows = []
        for y_val in whatif_years:
            row = input_record.copy()
            row["experience_years"] = y_val
            whatif_rows.append(row)

        whatif_df = pd.DataFrame(whatif_rows)
        whatif_pred_log = reg_pipe.predict(whatif_df[CORE])
        whatif_pred_usd = np.expm1(whatif_pred_log)

        fig_whatif = go.Figure()
        apply_theme(fig_whatif)
        fig_whatif.add_trace(go.Scatter(
            x=whatif_years,
            y=whatif_pred_usd,
            mode="lines+markers",
            name="Predicted Salary",
            line=dict(color="#818cf8", width=3),
            fill="tozeroy",
            fillcolor="rgba(129, 140, 248, 0.12)"
        ))
        fig_whatif.add_vline(
            x=exp_years,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text=f"Current: {exp_years} yrs (${pred_usd:,.0f})",
            annotation_position="top left"
        )
        fig_whatif.update_layout(
            title=f"Salary Trajectory by Experience Years ({job_role}, {country})",
            xaxis_title="Experience Years",
            yaxis_title="Predicted Salary ($)",
            height=380
        )
        fig_whatif.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig_whatif, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3: Model Insights
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🧠 Model Performance & Statistical Diagnostics")

    # 1. Benchmark Table
    st.markdown("#### Regression Benchmark Summary")
    if COMP_CSV.exists():
        comp_df = pd.read_csv(COMP_CSV)
        st.dataframe(
            comp_df.style.highlight_max(subset=["Test_R2", "Train_R2"], color="#d1fae5", axis=0)
                         .highlight_min(subset=["Test_MAE_Dollar", "Test_RMSE_Dollar", "Gap_R2"], color="#d1fae5", axis=0)
                         .format({
                             "Train_R2": "{:.4f}",
                             "Test_R2": "{:.4f}",
                             "Gap_R2": "{:.4f}",
                             "Test_MAE_Dollar": "${:,.2f}",
                             "Test_RMSE_Dollar": "${:,.2f}"
                         }),
            use_container_width=True
        )
        st.caption("Green highlights best values per metric. All error metrics evaluated on US Dollar scale after inverse log transformation.")
    else:
        st.info("Run `salary_project.ipynb` to generate `reports/model_comparison.csv`.")

    st.markdown("---")

    # 2. Diagnostic Figures
    st.markdown("#### Model Diagnostics & Explanations")
    col1, col2 = st.columns(2)

    with col1:
        p_train_test = FIG_DIR / "train_vs_test_tuning.png"
        if p_train_test.exists():
            st.markdown("**Bias-Variance Trade-off: Parameter Tuning**")
            st.image(str(p_train_test), use_container_width=True)
            st.caption("Decision Tree depth sweep demonstrates underfitting (low depth) to overfitting (high depth).")

        p_rf_imp = FIG_DIR / "rf_feature_importance.png"
        if p_rf_imp.exists():
            st.markdown("**Random Forest: Top 15 Feature Importances (MDI)**")
            st.image(str(p_rf_imp), use_container_width=True)
            st.caption("Experience years and geographic country indicators dominate predictive importance.")

        p_gd = FIG_DIR / "gradient_descent_loss.png"
        if p_gd.exists():
            st.markdown("**Stochastic Gradient Descent: Training Loss Convergence**")
            st.image(str(p_gd), use_container_width=True)
            st.caption("Epoch-by-epoch MSE loss curve confirming gradient descent convergence.")

    with col2:
        p_act = FIG_DIR / "predicted_vs_actual.png"
        if p_act.exists():
            st.markdown("**Predicted vs Actual Salary (Test Set)**")
            st.image(str(p_act), use_container_width=True)
            st.caption("Scatter plot comparing dollar predictions against ground truth on unseen test data.")

        p_cm = FIG_DIR / "salary_band_confusion_matrix.png"
        if p_cm.exists():
            st.markdown("**Salary Band Confusion Matrix (Tertiles)**")
            st.image(str(p_cm), use_container_width=True)
            st.caption("Accuracy across Low, Mid, and High salary tertiles.")

        p_pca = FIG_DIR / "pca_explained_variance.png"
        if p_pca.exists():
            st.markdown("**PCA Explained Variance Ratio**")
            st.image(str(p_pca), use_container_width=True)
            st.caption("Cumulative variance of 10 macroeconomic market indicators.")

    st.markdown("---")

    # 3. CORE vs EXTENDED Feature Comparison
    st.markdown("#### CORE vs EXTENDED Feature Evaluation")
    st.markdown("""
| Feature Set | Features Included | Test $R^2$ | Test MAE ($) | Takeaway |
|---|---|---|---|---|
| **CORE** | 8 candidate/role categories + experience years + year (10 features) | **~0.928** | **~$9,300** | Highly efficient, user-enterable without external lookups. |
| **EXTENDED** | CORE + 10 macroeconomic & firmographic indicators (20 features) | **~0.928** | **~$9,302** | Zero predictive lift. Added features contribute no incremental variance. |

**Architectural Decision**: Because adding macroeconomic indicators provided virtually zero lift on this dataset, the inference pipeline strictly uses the **CORE** feature set.
""")

    # 4. Limitations
    st.markdown("#### ⚠️ Key Limitations & Methodological Caveats")
    st.warning("""
1. **Synthetic Nature**: The dataset demonstrates clear signs of synthetic generation (near-uniform distributions across industry, work mode, and company size). Results should not be applied to real career decisions.
2. **Deterministic Additivity**: Linear and Ridge regression achieve accuracy on par with non-linear tree ensembles, indicating approximately log-linear underlying rules.
3. **Experience Dominance**: The model's predictive variance is almost entirely driven by `experience_years` and `country`. Other candidate dimensions provide minimal separation.
""")
