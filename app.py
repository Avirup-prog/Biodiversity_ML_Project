# ─────────────────────────────────────────────────────────────────────────────
# Intelligent Biodiversity Risk Assessment System
# Streamlit UI – Premium Dark Theme
# Run: streamlit run app.py
# Place covtype.csv in the same folder
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BiodiversityAI · Risk Assessment",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Base ── */
html, body, .stApp {
    background-color: #080c10 !important;
    font-family: 'Inter', sans-serif;
    color: #c9d1d9;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1400px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a0f14 100%) !important;
    border-right: 1px solid #1a2332 !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1117 0%, #0f2318 40%, #0a1a0d 100%);
    border: 1px solid #1a3a20;
    border-radius: 16px;
    padding: 40px 48px;
    margin: 24px 0 32px 0;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(34,197,94,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(16,185,129,0.05) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    color: #22c55e;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #f0f6fc;
    line-height: 1.15;
    margin-bottom: 10px;
}
.hero-title span { color: #22c55e; }
.hero-subtitle {
    font-size: 0.95rem;
    color: #6e7f8d;
    font-weight: 400;
    margin-bottom: 24px;
    line-height: 1.6;
}
.hero-badges { display: flex; gap: 10px; flex-wrap: wrap; }
.badge {
    background: #111a14;
    border: 1px solid #1e3a24;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: #5eead4;
    font-weight: 500;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117 !important;
    border-bottom: 1px solid #1a2332 !important;
    gap: 4px;
    padding: 0 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #6e7f8d !important;
    border-radius: 8px 8px 0 0 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 10px 20px !important;
    border: none !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: #111a14 !important;
    color: #22c55e !important;
    border-bottom: 2px solid #22c55e !important;
}

/* ── Sidebar labels ── */
.sidebar-logo {
    background: linear-gradient(135deg, #0f2318, #0a1a0d);
    border-bottom: 1px solid #1a3a20;
    padding: 28px 20px 24px 20px;
    margin-bottom: 8px;
}
.sidebar-logo-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #22c55e;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sidebar-logo-sub {
    font-size: 0.72rem;
    color: #3d5a4a;
    margin-top: 4px;
    letter-spacing: 0.05em;
}
.sidebar-section {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3d5a4a;
    padding: 16px 20px 6px 20px;
}
.sidebar-info-box {
    background: #0a1210;
    border: 1px solid #1a2e1f;
    border-radius: 10px;
    padding: 14px 16px;
    margin: 4px 12px 12px 12px;
    font-size: 0.8rem;
    color: #7a9e85;
    line-height: 1.6;
}
.sidebar-info-box strong { color: #22c55e; }

/* ── Sliders ── */
.stSlider label { color: #8b99a8 !important; font-size: 0.82rem !important; }
.stSlider [data-testid="stTickBar"] { color: #3d5a4a !important; }

/* ── Predict Button ── */
.stButton > button {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    color: #f0f6fc !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 32px !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 0 24px rgba(22,163,74,0.25) !important;
    transition: all 0.2s !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
    box-shadow: 0 0 36px rgba(34,197,94,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Input section card ── */
.input-card {
    background: #0d1117;
    border: 1px solid #1a2332;
    border-radius: 14px;
    padding: 28px 28px 20px 28px;
    margin-bottom: 20px;
}
.input-card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    color: #22c55e;
    text-transform: uppercase;
    margin-bottom: 18px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1a2332;
}
.col-header {
    font-size: 0.75rem;
    font-weight: 600;
    color: #3d5a4a;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 14px;
}

/* ── Result cards ── */
.result-low {
    background: linear-gradient(135deg, #052e16, #0a1f0e);
    border: 1px solid #166534;
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
    box-shadow: 0 0 40px rgba(22,163,74,0.15);
}
.result-medium {
    background: linear-gradient(135deg, #1c1202, #130e01);
    border: 1px solid #92400e;
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
    box-shadow: 0 0 40px rgba(217,119,6,0.15);
}
.result-high {
    background: linear-gradient(135deg, #1f0707, #130404);
    border: 1px solid #991b1b;
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
    box-shadow: 0 0 40px rgba(220,38,38,0.15);
}
.result-icon { font-size: 3rem; margin-bottom: 8px; }
.result-label-low  { font-family: 'Space Grotesk', sans-serif; font-size: 1.9rem; font-weight: 700; color: #22c55e; }
.result-label-med  { font-family: 'Space Grotesk', sans-serif; font-size: 1.9rem; font-weight: 700; color: #f59e0b; }
.result-label-high { font-family: 'Space Grotesk', sans-serif; font-size: 1.9rem; font-weight: 700; color: #ef4444; }
.result-desc { font-size: 0.85rem; color: #6e7f8d; margin-top: 6px; line-height: 1.5; }

/* ── Probability bar ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}
.prob-label { font-size: 0.8rem; color: #8b99a8; width: 90px; text-align: right; flex-shrink: 0; }
.prob-track {
    flex: 1;
    background: #111a14;
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
    border: 1px solid #1a2e1f;
}
.prob-fill-low  { height: 100%; background: linear-gradient(90deg, #15803d, #22c55e); border-radius: 6px; transition: width 0.6s ease; }
.prob-fill-med  { height: 100%; background: linear-gradient(90deg, #b45309, #f59e0b); border-radius: 6px; }
.prob-fill-high { height: 100%; background: linear-gradient(90deg, #b91c1c, #ef4444); border-radius: 6px; }
.prob-pct { font-size: 0.78rem; color: #22c55e; width: 42px; flex-shrink: 0; font-weight: 600; }

/* ── Metric cards ── */
.metric-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 24px; }
.metric-card {
    background: #0d1117;
    border: 1px solid #1a2332;
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #22c55e, #10b981);
}
.metric-card.orange::before { background: linear-gradient(90deg, #f59e0b, #fb923c); }
.metric-card.red::before    { background: linear-gradient(90deg, #ef4444, #f97316); }
.metric-card.teal::before   { background: linear-gradient(90deg, #14b8a6, #06b6d4); }
.metric-val   { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; color: #f0f6fc; }
.metric-lbl   { font-size: 0.72rem; color: #4a6572; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }

/* ── Section label ── */
.section-lbl {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    color: #3d5a4a;
    text-transform: uppercase;
    margin: 24px 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #111a14;
}

/* ── Dataframe ── */
.stDataFrame { border: 1px solid #1a2332 !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ── Train Model (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train():
    df = pd.read_csv('covtype_small.csv')
    feature_cols = [
        'Elevation','Aspect','Slope',
        'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points'
    ]
    X = df[feature_cols]
    risk_score = (
        (df['Elevation'] < df['Elevation'].quantile(0.33)).astype(int) +
        (df['Slope'] > df['Slope'].quantile(0.66)).astype(int) +
        (df['Horizontal_Distance_To_Roadways'] < df['Horizontal_Distance_To_Roadways'].quantile(0.33)).astype(int) +
        (df['Horizontal_Distance_To_Fire_Points'] < df['Horizontal_Distance_To_Fire_Points'].quantile(0.33)).astype(int)
    )
    y = risk_score.apply(lambda s: 0 if s <= 1 else (1 if s == 2 else 2))
    sample_df = pd.concat([X, y.rename('Risk')], axis=1)
    sample_df = pd.concat([
        grp.sample(min(len(grp), 3334), random_state=42)
        for _, grp in sample_df.groupby('Risk')
    ]).reset_index(drop=True)
    X_s = sample_df[feature_cols]
    y_s = sample_df['Risk']
    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y_s, test_size=0.2, random_state=42, stratify=y_s
    )
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    model.fit(X_tr_sc, y_train)
    y_pred = model.predict(X_te_sc)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Low','Medium','High'], output_dict=True)
    cm     = confusion_matrix(y_test, y_pred)
    return model, scaler, feature_cols, acc, report, cm, X_s, y_s, y_pred


with st.spinner(""):
    model, scaler, feature_cols, acc, report, cm, X_s, y_s, y_pred = load_and_train()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-text">🌿 BiodiversityAI</div>
        <div class="sidebar-logo-sub">RISK ASSESSMENT PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Project Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-info-box">
        <strong>Student:</strong> Avirup Shyam<br>
        <strong>SRN:</strong> PES1PG25CA033<br>
        <strong>Domain:</strong> Agriculture<br>
        <strong>SDG:</strong> 15 – Life on Land
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Model Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sidebar-info-box">
        <strong>Algorithm:</strong> Logistic Regression<br>
        <strong>Classes:</strong> Low · Medium · High<br>
        <strong>Accuracy:</strong> {acc*100:.1f}%<br>
        <strong>Training Samples:</strong> ~8,000<br>
        <strong>Features:</strong> 10 terrain variables
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Risk Legend</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-info-box">
        🟢 <strong style="color:#22c55e">Low</strong> — Stable, far from threats<br><br>
        🟡 <strong style="color:#f59e0b">Medium</strong> — Some risk factors present<br><br>
        🔴 <strong style="color:#ef4444">High</strong> — Vulnerable, needs attention
    </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-eyebrow">🛰️ &nbsp; AI-Powered Environmental Intelligence</div>
    <div class="hero-title">Intelligent <span>Biodiversity</span><br>Risk Assessment System</div>
    <div class="hero-subtitle">
        Analyze terrain and environmental data to classify regions into biodiversity risk levels.<br>
        Built with Logistic Regression · Forest Cover Type Dataset · SDG 15: Life on Land
    </div>
    <div class="hero-badges">
        <span class="badge">🌿 Supervised Learning</span>
        <span class="badge">📊 Logistic Regression</span>
        <span class="badge">🗺️ 581K+ Records</span>
        <span class="badge">🎯 3-Class Classification</span>
        <span class="badge">🇮🇳 PES University</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍  Predict Risk Level",
    "📊  Model Performance",
    "📈  Visualizations"
])


# ════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="input-card-title">⚙️ &nbsp; Environmental Parameters</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="col-header">🏔️ Terrain Features</div>', unsafe_allow_html=True)
        elevation  = st.slider("Elevation (m)",  1863, 3849, 2500)
        aspect     = st.slider("Aspect (°)",     0, 360, 90)
        slope      = st.slider("Slope (°)",      0, 52, 15)
        h_hydro    = st.slider("Horiz. Distance to Hydrology (m)", 0, 1343, 200)
        v_hydro    = st.slider("Vert. Distance to Hydrology (m)",  -146, 554, 30)

    with col2:
        st.markdown('<div class="col-header">🛣️ Proximity & Light Features</div>', unsafe_allow_html=True)
        h_road     = st.slider("Horiz. Distance to Roadways (m)",    0, 6890, 1500)
        h_fire     = st.slider("Horiz. Distance to Fire Points (m)", 0, 6993, 500)
        shade_9am  = st.slider("Hillshade at 9am",   0, 254, 210)
        shade_noon = st.slider("Hillshade at Noon",  0, 254, 230)
        shade_3pm  = st.slider("Hillshade at 3pm",   0, 254, 150)

    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("🌿  Analyze Biodiversity Risk")

    if predict_btn:
        input_data = pd.DataFrame([{
            'Elevation': elevation, 'Aspect': aspect, 'Slope': slope,
            'Horizontal_Distance_To_Hydrology': h_hydro,
            'Vertical_Distance_To_Hydrology': v_hydro,
            'Horizontal_Distance_To_Roadways': h_road,
            'Hillshade_9am': shade_9am, 'Hillshade_Noon': shade_noon,
            'Hillshade_3pm': shade_3pm,
            'Horizontal_Distance_To_Fire_Points': h_fire
        }])
        scaled   = scaler.transform(input_data)
        pred     = model.predict(scaled)[0]
        proba    = model.predict_proba(scaled)[0]

        st.markdown("---")
        c_res, c_prob = st.columns([1.2, 1], gap="large")

        with c_res:
            if pred == 0:
                st.markdown(f"""
                <div class="result-low">
                    <div class="result-icon">🟢</div>
                    <div class="result-label-low">LOW RISK</div>
                    <div class="result-desc">This region shows stable terrain characteristics<br>
                    with minimal human activity pressure.<br>Biodiversity loss risk is low.</div>
                </div>""", unsafe_allow_html=True)
            elif pred == 1:
                st.markdown(f"""
                <div class="result-medium">
                    <div class="result-icon">🟡</div>
                    <div class="result-label-med">MEDIUM RISK</div>
                    <div class="result-desc">Some concerning environmental factors detected.<br>
                    Monitoring and preventive action<br>is recommended for this region.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-high">
                    <div class="result-icon">🔴</div>
                    <div class="result-label-high">HIGH RISK</div>
                    <div class="result-desc">This region is highly vulnerable to biodiversity loss.<br>
                    Immediate conservation action<br>and policy intervention is advised.</div>
                </div>""", unsafe_allow_html=True)

        with c_prob:
            st.markdown('<div class="section-lbl">Prediction Confidence</div>', unsafe_allow_html=True)
            labels   = ['Low Risk', 'Medium Risk', 'High Risk']
            fills    = ['prob-fill-low', 'prob-fill-med', 'prob-fill-high']
            for i, (lbl, fill, p) in enumerate(zip(labels, fills, proba)):
                pct = p * 100
                st.markdown(f"""
                <div class="prob-row">
                    <div class="prob-label">{lbl}</div>
                    <div class="prob-track">
                        <div class="{fill}" style="width:{pct:.1f}%"></div>
                    </div>
                    <div class="prob-pct">{pct:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            # Key contributors
            st.markdown('<div class="section-lbl" style="margin-top:22px">Key Risk Factors</div>', unsafe_allow_html=True)
            factors = {
                "Proximity to Roads": h_road < 500,
                "Proximity to Fires": h_fire < 500,
                "Low Elevation":      elevation < 2300,
                "High Slope":         slope > 25,
            }
            for name, flagged in factors.items():
                color = "#ef4444" if flagged else "#22c55e"
                icon  = "⚠️" if flagged else "✅"
                st.markdown(f'<div style="font-size:0.8rem;color:{color};margin-bottom:6px">{icon} {name}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TAB 2 — PERFORMANCE
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-lbl">Overall Performance</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-val">{acc*100:.1f}%</div>
            <div class="metric-lbl">Overall Accuracy</div>
        </div>
        <div class="metric-card teal">
            <div class="metric-val">{report['Low']['f1-score']:.2f}</div>
            <div class="metric-lbl">F1 · Low Risk</div>
        </div>
        <div class="metric-card orange">
            <div class="metric-val">{report['Medium']['f1-score']:.2f}</div>
            <div class="metric-lbl">F1 · Medium Risk</div>
        </div>
        <div class="metric-card red">
            <div class="metric-val">{report['High']['f1-score']:.2f}</div>
            <div class="metric-lbl">F1 · High Risk</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_cm, col_rep = st.columns([1, 1.4], gap="large")

    with col_cm:
        st.markdown('<div class="section-lbl">Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low','Medium','High'])
        disp.plot(ax=ax, cmap='Greens', colorbar=False)
        for text in ax.texts: text.set_color('white'); text.set_fontsize(13)
        ax.tick_params(colors='#8b99a8', labelsize=9)
        ax.xaxis.label.set_color('#8b99a8'); ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_color('#8b99a8'); ax.yaxis.label.set_size(9)
        ax.set_title('Confusion Matrix', color='#c9d1d9', fontsize=11, pad=12)
        for sp in ax.spines.values(): sp.set_color('#1a2332')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col_rep:
        st.markdown('<div class="section-lbl">Classification Report</div>', unsafe_allow_html=True)
        rep_df = pd.DataFrame(report).T.round(3)
        st.dataframe(rep_df, use_container_width=True, height=240)

        st.markdown('<div class="section-lbl" style="margin-top:18px">Precision vs Recall</div>', unsafe_allow_html=True)
        classes = ['Low','Medium','High']
        prec = [report[c]['precision'] for c in classes]
        rec  = [report[c]['recall']    for c in classes]
        x    = np.arange(len(classes))
        fig2, ax2 = plt.subplots(figsize=(5, 2.8))
        fig2.patch.set_facecolor('#0d1117')
        ax2.set_facecolor('#0d1117')
        bars1 = ax2.bar(x - 0.2, prec, 0.35, label='Precision', color='#22c55e', alpha=0.85, linewidth=0)
        bars2 = ax2.bar(x + 0.2, rec,  0.35, label='Recall',    color='#14b8a6', alpha=0.85, linewidth=0)
        ax2.set_xticks(x); ax2.set_xticklabels(classes, color='#8b99a8', fontsize=9)
        ax2.tick_params(colors='#8b99a8', labelsize=8)
        ax2.set_ylim(0, 1.1)
        ax2.legend(framealpha=0, labelcolor='#8b99a8', fontsize=8)
        for sp in ax2.spines.values(): sp.set_color('#1a2332')
        ax2.set_facecolor('#0d1117')
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 3 — VISUALIZATIONS
# ════════════════════════════════════════════════════════
with tab3:
    col_v1, col_v2 = st.columns(2, gap="large")

    with col_v1:
        st.markdown('<div class="section-lbl">Predicted Risk Distribution</div>', unsafe_allow_html=True)
        risk_counts = pd.Series(y_pred).value_counts().sort_index()
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        fig3.patch.set_facecolor('#0d1117')
        ax3.set_facecolor('#0d1117')
        clrs = ['#22c55e','#f59e0b','#ef4444']
        bars = ax3.bar(['Low Risk','Medium Risk','High Risk'], risk_counts.values,
                       color=clrs, linewidth=0, width=0.55)
        for bar, val in zip(bars, risk_counts.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                     str(val), ha='center', va='bottom', color='#c9d1d9', fontsize=9)
        ax3.tick_params(colors='#8b99a8', labelsize=9)
        ax3.set_ylabel('Regions', color='#8b99a8', fontsize=9)
        for sp in ['top','right']: ax3.spines[sp].set_visible(False)
        for sp in ['left','bottom']: ax3.spines[sp].set_color('#1a2332')
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)

    with col_v2:
        st.markdown('<div class="section-lbl">Feature Importance (LR Coefficients)</div>', unsafe_allow_html=True)
        coef_vals = np.abs(model.coef_).mean(axis=0)
        coef_df   = pd.DataFrame({'Feature': feature_cols, 'Importance': coef_vals})
        coef_df   = coef_df.sort_values('Importance')
        fig4, ax4 = plt.subplots(figsize=(5, 3.8))
        fig4.patch.set_facecolor('#0d1117')
        ax4.set_facecolor('#0d1117')
        colors_bar = ['#22c55e' if v == coef_df['Importance'].max() else '#1a4a2a' for v in coef_df['Importance']]
        ax4.barh(coef_df['Feature'], coef_df['Importance'], color=colors_bar, linewidth=0, height=0.6)
        ax4.tick_params(colors='#8b99a8', labelsize=7.5)
        ax4.set_xlabel('Mean |Coefficient|', color='#8b99a8', fontsize=9)
        for sp in ['top','right']: ax4.spines[sp].set_visible(False)
        for sp in ['left','bottom']: ax4.spines[sp].set_color('#1a2332')
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)

    st.markdown('<div class="section-lbl" style="margin-top:8px">Risk Score Breakdown by Feature</div>', unsafe_allow_html=True)
    col_v3, col_v4 = st.columns(2, gap="large")

    with col_v3:
        # Elevation distribution colored by risk
        fig5, ax5 = plt.subplots(figsize=(5, 3))
        fig5.patch.set_facecolor('#0d1117')
        ax5.set_facecolor('#0d1117')
        plot_df = pd.concat([X_s[['Elevation']], y_s.rename('Risk')], axis=1)
        for risk_val, clr, lbl in [(0,'#22c55e','Low'),(1,'#f59e0b','Medium'),(2,'#ef4444','High')]:
            subset = plot_df[plot_df['Risk'] == risk_val]['Elevation']
            ax5.hist(subset, bins=30, color=clr, alpha=0.65, label=lbl, linewidth=0)
        ax5.legend(framealpha=0, labelcolor='#8b99a8', fontsize=8)
        ax5.set_xlabel('Elevation (m)', color='#8b99a8', fontsize=9)
        ax5.set_ylabel('Count', color='#8b99a8', fontsize=9)
        ax5.set_title('Elevation Distribution by Risk', color='#c9d1d9', fontsize=10)
        ax5.tick_params(colors='#8b99a8', labelsize=8)
        for sp in ['top','right']: ax5.spines[sp].set_visible(False)
        for sp in ['left','bottom']: ax5.spines[sp].set_color('#1a2332')
        plt.tight_layout()
        st.pyplot(fig5, use_container_width=True)

    with col_v4:
        # Slope distribution colored by risk
        fig6, ax6 = plt.subplots(figsize=(5, 3))
        fig6.patch.set_facecolor('#0d1117')
        ax6.set_facecolor('#0d1117')
        plot_df2 = pd.concat([X_s[['Slope']], y_s.rename('Risk')], axis=1)
        for risk_val, clr, lbl in [(0,'#22c55e','Low'),(1,'#f59e0b','Medium'),(2,'#ef4444','High')]:
            subset = plot_df2[plot_df2['Risk'] == risk_val]['Slope']
            ax6.hist(subset, bins=25, color=clr, alpha=0.65, label=lbl, linewidth=0)
        ax6.legend(framealpha=0, labelcolor='#8b99a8', fontsize=8)
        ax6.set_xlabel('Slope (°)', color='#8b99a8', fontsize=9)
        ax6.set_ylabel('Count', color='#8b99a8', fontsize=9)
        ax6.set_title('Slope Distribution by Risk', color='#c9d1d9', fontsize=10)
        ax6.tick_params(colors='#8b99a8', labelsize=8)
        for sp in ['top','right']: ax6.spines[sp].set_visible(False)
        for sp in ['left','bottom']: ax6.spines[sp].set_color('#1a2332')
        plt.tight_layout()
        st.pyplot(fig6, use_container_width=True)
