import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import traceback

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudSense",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0a0a0f;
    color: #e8e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f0f1a;
    border-right: 1px solid #1e1e2e;
}

/* Main title */
.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: #00ff88;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.main-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #666680;
    font-weight: 300;
    letter-spacing: 0.05em;
    margin-bottom: 2rem;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #00ff88;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e1e2e;
}

/* Metric cards */
.metric-card {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 1.2rem;
    text-align: center;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00ff88;
}

.metric-label {
    font-size: 0.75rem;
    color: #666680;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Result boxes */
.result-fraud {
    background: linear-gradient(135deg, #1a0a0a, #2a0f0f);
    border: 1px solid #ff3366;
    border-left: 4px solid #ff3366;
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}

.result-legit {
    background: linear-gradient(135deg, #0a1a0f, #0f2a1a);
    border: 1px solid #00ff88;
    border-left: 4px solid #00ff88;
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}

.result-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.result-fraud .result-title { color: #ff3366; }
.result-legit .result-title { color: #00ff88; }

.result-subtitle {
    font-size: 0.85rem;
    color: #888899;
}

/* Buttons */
.stButton > button {
    background-color: #00ff88;
    color: #0a0a0f;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    border: none;
    border-radius: 4px;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background-color: #00cc6a;
    transform: translateY(-1px);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0f0f1a;
    border: 1px dashed #2e2e4e;
    border-radius: 8px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0f0f1a;
    border-bottom: 1px solid #1e1e2e;
    gap: 0;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    color: #666680;
    padding: 0.8rem 1.5rem;
    border-radius: 0;
}

.stTabs [aria-selected="true"] {
    color: #00ff88 !important;
    border-bottom: 2px solid #00ff88 !important;
    background: transparent !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1e1e2e;
    border-radius: 8px;
}

/* Divider */
hr {
    border-color: #1e1e2e;
    margin: 2rem 0;
}

/* Sidebar nav items */
.nav-item {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #666680;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.5rem 0;
}

.info-box {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: #888899;
    margin: 0.5rem 0;
}

.tag {
    display: inline-block;
    background: #1e1e2e;
    color: #00ff88;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 3px;
    margin-right: 0.3rem;
}
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    """Load preprocessor and model from artifacts using smart absolute paths."""
    try:
        # 1. Get the directory where THIS streamlit script is located
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Smartly determine the project root
        # If 'src' is right next to this script, this folder is the root.
        # Otherwise, assume the root is one folder up.
        if os.path.exists(os.path.join(SCRIPT_DIR, 'src')):
            PROJECT_ROOT = SCRIPT_DIR
        else:
            PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
            
        # 3. Add the project root to sys.path so Python can find 'src'
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
            
        from src.utils import load_obj
        
        # 4. Build absolute paths to artifacts based on the project root
        preprocessor_path = os.path.join(PROJECT_ROOT, 'artifacts', 'preprocessor.pkl')
        model_path = os.path.join(PROJECT_ROOT, 'artifacts', 'model.pkl')

        preprocessor = load_obj(preprocessor_path)
        model = load_obj(model_path)
        
        return preprocessor, model
        
    except Exception as e:
        st.sidebar.error(f"Pipeline Load Failed: {str(e)}")
        return None, None


@st.cache_data
def load_sample_data():
    """Load a few sample rows for demo predictions."""
    # 3 legit + 2 fraud sample transactions (hardcoded from dataset)
    samples = {
        "Sample 1 — Legit": {
            "Time": 0.0, "V1": -1.3598, "V2": -0.0728, "V3": 2.5363,
            "V4": 1.3782, "V5": -0.3383, "V6": 0.4624, "V7": 0.2396,
            "V8": 0.0987, "V9": 0.3638, "V10": 0.0908, "V11": -0.5516,
            "V12": -0.6178, "V13": -0.9914, "V14": -0.3112, "V15": 1.4682,
            "V16": -0.4704, "V17": 0.2080, "V18": 0.0258, "V19": 0.4040,
            "V20": 0.2514, "V21": -0.0183, "V22": 0.2778, "V23": -0.1105,
            "V24": 0.0669, "V25": 0.1285, "V26": -0.1891, "V27": 0.1336,
            "V28": -0.0211, "Amount": 149.62
        },
        "Sample 2 — Legit": {
            "Time": 406.0, "V1": 1.2292, "V2": 0.1411, "V3": 0.0453,
            "V4": 1.2026, "V5": 0.1912, "V6": 0.2722, "V7": 0.1271,
            "V8": 0.0009, "V9": 0.2142, "V10": 0.2224, "V11": 0.0509,
            "V12": 0.6840, "V13": -0.1435, "V14": -0.1558, "V15": -0.6702,
            "V16": 0.1288, "V17": 0.1890, "V18": 0.0672, "V19": -0.0580,
            "V20": -0.0594, "V21": -0.0226, "V22": -0.2050, "V23": -0.1694,
            "V24": 0.1259, "V25": -0.0089, "V26": 0.0175, "V27": 0.0086,
            "V28": 0.0149, "Amount": 2.69
        },
        "Sample 3 — Fraud": {
            "Time": 406.0, "V1": -2.3122, "V2": 1.9519, "V3": -1.6097,
            "V4": 3.9979, "V5": -0.5221, "V6": -1.4265, "V7": -2.5374,
            "V8": 1.3918, "V9": -2.7700, "V10": -2.7722, "V11": 3.2020,
            "V12": -2.8992, "V13": -0.5955, "V14": -4.2890, "V15": 0.3898,
            "V16": -1.1407, "V17": -2.8306, "V18": -0.0168, "V19": 0.4167,
            "V20": 0.7269, "V21": 0.2811, "V22": -0.1450, "V23": -0.0750,
            "V24": -0.4214, "V25": 0.5664, "V26": 0.3261, "V27": 0.1809,
            "V28": 0.1337, "Amount": 1.00
        },
        "Sample 4 — Fraud": {
            "Time": 472.0, "V1": -3.0435, "V2": -3.1572, "V3": 1.0886,
            "V4": 2.2886, "V5": 1.3597, "V6": -1.0635, "V7": -3.5834,
            "V8": 1.8231, "V9": -0.5761, "V10": -5.5999, "V11": -0.4234,
            "V12": -5.3685, "V13": -2.6068, "V14": -6.4681, "V15": -4.2600,
            "V16": -1.1659, "V17": -6.2186, "V18": -1.1268, "V19": 0.4386,
            "V20": 1.2495, "V21": 0.5313, "V22": 0.7440, "V23": 0.0256,
            "V24": -0.1274, "V25": -0.3629, "V26": 0.2938, "V27": 0.5840,
            "V28": 0.2194, "Amount": 529.00
        },
    }
    return samples


def run_prediction(preprocessor, model, df):
    """Run prediction on a dataframe."""
    scaled = preprocessor.transform(df)
    pred = model.predict(scaled)
    prob = model.predict_proba(scaled)[:, 1]
    return pred, prob


def render_prediction_result(pred, prob, idx=0):
    """Render a single prediction result."""
    is_fraud = pred[idx] == 1
    confidence = prob[idx] if is_fraud else 1 - prob[idx]

    if is_fraud:
        st.markdown(f"""
        <div class="result-fraud">
            <div class="result-title">⚠ FRAUD DETECTED</div>
            <div class="result-subtitle">Confidence: {confidence*100:.1f}% — This transaction has been flagged as potentially fraudulent.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-legit">
            <div class="result-title">✓ LEGITIMATE</div>
            <div class="result-subtitle">Confidence: {confidence*100:.1f}% — This transaction appears to be legitimate.</div>
        </div>
        """, unsafe_allow_html=True)


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-title" style="font-size:1.5rem;">FRAUD<br>SENSE</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle" style="font-size:0.75rem;">Credit Card Fraud Detection</div>', unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation", 
        ["🔍 Predictions", "📊 EDA"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown('<div class="info-box">Model trained on <strong style="color:#00ff88">284,807</strong> transactions with <strong style="color:#ff3366">492</strong> fraud cases (0.17%)</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1rem;">', unsafe_allow_html=True)
    st.markdown('<span class="tag">CatBoost</span><span class="tag">SMOTE</span><span class="tag">sklearn</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── load model ────────────────────────────────────────────────────────────────
preprocessor, model = load_pipeline()
model_loaded = preprocessor is not None and model is not None


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Predictions":

    st.markdown('<div class="main-title">FRAUD DETECTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Run predictions on individual transactions or batch CSV files</div>', unsafe_allow_html=True)

    if not model_loaded:
        st.warning("⚠ Model artifacts not found. Train and save your model first (`artifacts/model.pkl` and `artifacts/preprocessor.pkl`).")
        st.stop()

    tab1, tab2 = st.tabs(["SAMPLE TRANSACTIONS", "UPLOAD CSV"])

    # ── tab 1: sample transactions ────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Pre-loaded samples</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#666680; font-size:0.85rem; margin-bottom:1.5rem;">Select a sample transaction to run a prediction. Samples 3 and 4 are known fraud cases from the dataset.</p>', unsafe_allow_html=True)

        samples = load_sample_data()
        selected = st.selectbox("Select a sample", list(samples.keys()), label_visibility="collapsed")
        sample_df = pd.DataFrame([samples[selected]])

        col1, col2 = st.columns([2, 1])

        with col1:
            with st.expander("View transaction data", expanded=False):
                st.dataframe(sample_df.T.rename(columns={0: "Value"}), width='stretch')

        with col2:
            st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
            if st.button("RUN PREDICTION", key="sample_predict", width='stretch'):
                with st.spinner("Analyzing..."):
                    pred, prob = run_prediction(preprocessor, model, sample_df)
                    render_prediction_result(pred, prob)

                    st.markdown(f"""
                    <div style="display:flex; gap:1rem; margin-top:1rem;">
                        <div class="metric-card" style="flex:1;">
                            <div class="metric-value">{prob[0]*100:.1f}%</div>
                            <div class="metric-label">Fraud Probability</div>
                        </div>
                        <div class="metric-card" style="flex:1;">
                            <div class="metric-value" style="color:#e8e8f0;">{sample_df['Amount'].values[0]:.2f}</div>
                            <div class="metric-label">Amount (€)</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── tab 2: csv upload ─────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Batch prediction via CSV</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#666680; font-size:0.85rem; margin-bottom:1.5rem;">Upload a CSV file with columns: Time, V1–V28, Amount. The Class column is optional and will be ignored.</p>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                # drop Class column if present
                if 'Class' in df.columns:
                    df = df.drop(columns=['Class'])

                st.markdown(f'<p style="color:#666680; font-size:0.8rem;">{len(df)} transaction(s) loaded</p>', unsafe_allow_html=True)
                st.dataframe(df.head(), width='stretch')

                if st.button("RUN BATCH PREDICTION", key="csv_predict", width='stretch'):
                    with st.spinner(f"Analyzing {len(df)} transactions..."):
                        pred, prob = run_prediction(preprocessor, model, df)

                        results_df = df.copy()
                        results_df['Fraud_Probability'] = (prob * 100).round(2)
                        results_df['Prediction'] = ['⚠ FRAUD' if p == 1 else '✓ LEGIT' for p in pred]

                        fraud_count = sum(pred)
                        legit_count = len(pred) - fraud_count

                        st.markdown("---")
                        st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(pred)}</div><div class="metric-label">Total</div></div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ff3366;">{fraud_count}</div><div class="metric-label">Fraud</div></div>', unsafe_allow_html=True)
                        with col3:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{legit_count}</div><div class="metric-label">Legit</div></div>', unsafe_allow_html=True)

                        st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
                        st.dataframe(
                            results_df[['Amount', 'Time', 'Fraud_Probability', 'Prediction']].sort_values('Fraud_Probability', ascending=False),
                            width='stretch'
                        )

                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button("DOWNLOAD RESULTS", csv, "fraud_predictions.csv", "text/csv")

            except Exception as e:
                st.error(f"Error processing file: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":

    st.markdown('<div class="main-title">EXPLORATORY<br>ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Dataset insights and feature distributions</div>', unsafe_allow_html=True)

    # try to load data
    data_path = os.path.join('Data', 'creditcard.csv')
    if not os.path.exists(data_path):
        st.warning("⚠ Dataset not found at `Data/creditcard.csv`. Add the dataset to view EDA.")
        st.stop()

    @st.cache_data
    def load_data():
        return pd.read_csv(data_path)

    data = load_data()

    # ── dataset overview ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    total = len(data)
    fraud = data['Class'].sum()
    legit = total - fraud
    fraud_pct = fraud / total * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total:,}</div><div class="metric-label">Total Transactions</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ff3366;">{fraud:,}</div><div class="metric-label">Fraud Cases</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{legit:,}</div><div class="metric-label">Legit Cases</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ff3366;">{fraud_pct:.2f}%</div><div class="metric-label">Fraud Rate</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── plots ─────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'figure.facecolor': '#0a0a0f',
        'axes.facecolor': '#0f0f1a',
        'axes.edgecolor': '#1e1e2e',
        'axes.labelcolor': '#888899',
        'xtick.color': '#666680',
        'ytick.color': '#666680',
        'text.color': '#e8e8f0',
        'grid.color': '#1e1e2e',
        'grid.alpha': 0.5,
    })

    col1, col2 = st.columns(2)

    # class imbalance bar chart
    with col1:
        st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(['Legitimate', 'Fraud'], [legit, fraud],
                      color=['#00ff88', '#ff3366'], width=0.5, alpha=0.85)
        ax.set_ylabel('Count', fontsize=9)
        for bar, val in zip(bars, [legit, fraud]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                    f'{val:,}', ha='center', va='bottom', fontsize=8, color='#888899')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # amount distribution
    with col2:
        st.markdown('<div class="section-header">Amount Distribution (Fraud vs Legit)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.kdeplot(data[data['Class']==1]['Amount'], ax=ax, label='Fraud',
                    fill=True, color='#ff3366', alpha=0.4, linewidth=1.5)
        sns.kdeplot(data[data['Class']==0]['Amount'], ax=ax, label='Legit',
                    fill=True, color='#00ff88', alpha=0.2, linewidth=1.5)
        ax.set_xlim(0, 2000)
        ax.set_xlabel('Amount (€)', fontsize=9)
        ax.legend(fontsize=8, framealpha=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    # time distribution
    with col3:
        st.markdown('<div class="section-header">Time Distribution (Fraud vs Legit)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.kdeplot(data[data['Class']==1]['Time'], ax=ax, label='Fraud',
                    fill=True, color='#ff3366', alpha=0.4, linewidth=1.5)
        sns.kdeplot(data[data['Class']==0]['Time'], ax=ax, label='Legit',
                    fill=True, color='#00ff88', alpha=0.2, linewidth=1.5)
        ax.set_xlabel('Time (seconds)', fontsize=9)
        ax.legend(fontsize=8, framealpha=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # feature correlation with Class
    with col4:
        st.markdown('<div class="section-header">Feature Correlation with Fraud</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        corr = data.corr()['Class'].drop('Class').sort_values()
        colors = ['#ff3366' if v < 0 else '#00ff88' for v in corr.values]
        corr.plot(kind='bar', ax=ax, color=colors, alpha=0.85, width=0.8)
        ax.set_xlabel('')
        ax.set_ylabel('Correlation', fontsize=9)
        ax.tick_params(axis='x', labelsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()