"""
Smart Data Preprocessing and Model Advisor — Streamlit UI (app.py)
===================================================================
NEW in this version
--------------------
  1. URL / Link Import    — load dataset from any URL (Kaggle, GitHub, Google Sheets, etc.)
  2. Expanded Visualisations — 12+ chart types in the Analyze tab
  3. Custom Chart Builder — pick chart type + columns + options, generate on demand

Run with:  streamlit run app.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

from main import (
    load_dataset, load_from_url,
    dataset_summary, compute_quality_score,
    clean_data, encode_and_scale, detect_problem_type,
    MODEL_CATALOG, train_all_models, get_feature_importances,
    dataset_report_card, generate_report_text, generate_report_markdown, generate_report_pdf,
    # Standard plots
    plot_missing_heatmap, plot_correlation_heatmap, plot_distributions,
    plot_boxplots, plot_violin, plot_kde, plot_categorical_bar,
    plot_pie_chart, plot_pairplot, plot_count_plot,
    # Model plots
    plot_feature_importances, plot_model_comparison, plot_confusion_matrix,
    # Custom builder
    CHART_TYPES, build_custom_chart,
)

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Data Preprocessing and Model Advisor",
    page_icon="📊",
    layout="wide",
)

# ── Minimal background colour theme ──────────────────────────────
st.markdown("""
<style>
/* Main page background — soft blue-grey */
.stApp {
    background-color: #f0f4f8;
}

/* Sidebar — slightly deeper shade */
section[data-testid="stSidebar"] {
    background-color: #dce7f3;
}

/* Tab bar */
.stTabs [data-baseweb="tab-list"] {
    background-color: #dce7f3;
    border-radius: 8px;
    padding: 4px;
}

/* Active tab */
.stTabs [aria-selected="true"] {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 6px;
}

/* Metric cards — white with soft shadow */
div[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 1px solid #c8d8ea;
    border-radius: 8px;
    padding: 12px 16px;
    box-shadow: 0 1px 4px rgba(37,99,235,0.08);
}

/* Expanders */
div[data-testid="stExpander"] {
    background-color: #ffffff;
    border: 1px solid #c8d8ea;
    border-radius: 8px;
}

/* Buttons */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

/* Dataframe border */
div[data-testid="stDataFrame"] {
    border: 1px solid #c8d8ea;
    border-radius: 8px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────
for key in ["raw_df", "clean_df", "target_col", "comparison", "report_card", "feat_names", "X", "y"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ──────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Smart Data Advisor")
    st.caption("Upload or import a dataset, then clean, analyze, and train models.")
    st.divider()

    # ── 1. Dataset Import ──────────────────────────────────────
    st.subheader("1. Load Dataset")

    import_mode = st.radio(
        "Import method",
        ["📁 Upload File", "🔗 Import from URL"],
        horizontal=True,
    )

    if import_mode == "📁 Upload File":
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            try:
                df_new = load_dataset(uploaded_file)
                if st.session_state.raw_df is None or len(df_new) != len(st.session_state.raw_df):
                    st.session_state.raw_df     = df_new
                    st.session_state.clean_df   = None
                    st.session_state.comparison = None
                    st.session_state.report_card = None
                    st.success(f"Loaded {df_new.shape[0]:,} rows × {df_new.shape[1]} cols")
            except ValueError as e:
                st.error(str(e))

    else:  # URL import
        st.caption("Paste a direct link to a CSV or Excel file.")
        with st.expander("📖 URL Tips"):
            st.markdown("""
**GitHub:** Replace `blob` with `raw` in the URL  
`github.com/user/repo/raw/main/data.csv`

**Google Sheets:** File → Share → Publish to web → CSV  

**Kaggle:** Use the raw file download link from the dataset Files tab  

**Any direct CSV link** from data.gov, UCI, etc.
            """)
        url_input = st.text_input("Dataset URL", placeholder="https://...")
        if st.button("⬇️ Load from URL", use_container_width=True):
            if not url_input.strip():
                st.error("Please enter a URL.")
            else:
                with st.spinner("Fetching dataset from URL..."):
                    try:
                        df_new = load_from_url(url_input.strip())
                        st.session_state.raw_df      = df_new
                        st.session_state.clean_df    = None
                        st.session_state.comparison  = None
                        st.session_state.report_card = None
                        st.success(f"✅ Loaded {df_new.shape[0]:,} rows × {df_new.shape[1]} cols")
                    except ValueError as e:
                        st.error(str(e))

    # ── 2. Target column ───────────────────────────────────────
    if st.session_state.raw_df is not None:
        st.divider()
        st.subheader("2. Target Column")
        st.caption("Column to predict. Leave as None for clustering.")
        options = ["None (Clustering)"] + list(st.session_state.raw_df.columns)
        choice  = st.selectbox("Target column", options)
        st.session_state.target_col = None if choice == "None (Clustering)" else choice

    st.divider()

    # ── 3. Train/Test split ────────────────────────────────────
    st.subheader("3. Train / Test Split")
    split_pct = st.slider("Training data %", 60, 90, 80, step=10,
                           help="80 = 80% train / 20% test")
    test_size = round(1 - split_pct / 100, 2)
    st.caption(f"→ **{split_pct}% train** / **{100-split_pct}% test**")

    st.divider()
    st.markdown("**Steps:**")
    st.markdown("""
1. Load dataset (file or URL)
2. Select target column
3. Set train/test split
4. **Clean Data** tab
5. **Analyze** tab — charts
6. **Custom Charts** tab — build your own
7. **Train & Compare** tab
8. **Report Card** tab — export
    """)

# ──────────────────────────────────────────────────────────────────
# MAIN TITLE
# ──────────────────────────────────────────────────────────────────
st.title("📊 Smart Data Preprocessing and Model Advisor")
st.caption("Load · Clean · Visualize · Train · Export — all in one place.")

if st.session_state.raw_df is None:
    st.info("👈 Load a dataset from the sidebar to get started — upload a file or paste a URL.")
    st.stop()

df = st.session_state.raw_df

# ──────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────
(tab_overview, tab_clean, tab_analyze,
 tab_custom, tab_train, tab_report) = st.tabs([
    "📁 Overview",
    "🧹 Clean Data",
    "📈 Analyze & Charts",
    "🎨 Custom Charts",
    "🚀 Train & Compare",
    "📋 Report Card",
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════
with tab_overview:
    st.header("Dataset Overview")
    summary = dataset_summary(df)
    quality = compute_quality_score(df)

    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("Rows",             f"{summary['rows']:,}")
    c2.metric("Columns",          summary['cols'])
    c3.metric("Missing Cells",    f"{summary['missing']:,}")
    c4.metric("Duplicates",       f"{summary['duplicates']:,}")
    c5.metric("Numeric Cols",     summary['numeric'])
    c6.metric("Categorical Cols", summary['categorical'])
    c7.metric("Quality Score",    f"{quality}/100")

    st.divider()
    with st.expander("View Column Details"):
        col_info = pd.DataFrame({
            "Column":    list(df.columns),
            "Data Type": [str(df[c].dtype)          for c in df.columns],
            "Non-Null":  [int(df[c].count())         for c in df.columns],
            "Missing":   [int(df[c].isnull().sum())  for c in df.columns],
            "Unique":    [int(df[c].nunique())        for c in df.columns],
            "Sample":    [str(df[c].dropna().iloc[0]) if df[c].count() > 0 else "N/A" for c in df.columns],
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)

    st.subheader("Raw Data Preview (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — CLEAN DATA
# ══════════════════════════════════════════════════════════════════
with tab_clean:
    st.header("🧹 Data Cleaning")
    st.caption("Fixes missing values (median/mode), removes duplicates, caps outliers via IQR.")

    if st.button("🧹 Run Cleaning Pipeline", use_container_width=True):
        with st.spinner("Cleaning..."):
            cleaned = clean_data(df)
            st.session_state.clean_df = cleaned

    if st.session_state.clean_df is not None:
        cleaned       = st.session_state.clean_df
        before        = dataset_summary(df)
        after         = dataset_summary(cleaned)
        q_raw         = compute_quality_score(df)
        q_after       = compute_quality_score(cleaned)

        st.subheader("Before vs After")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Before Cleaning**")
            st.metric("Missing Cells",  f"{before['missing']:,}")
            st.metric("Duplicate Rows", f"{before['duplicates']:,}")
            st.metric("Quality Score",  f"{q_raw}/100")
        with col_b:
            st.markdown("**After Cleaning**")
            st.metric("Missing Cells",  f"{after['missing']:,}",  delta=f"-{before['missing']-after['missing']}")
            st.metric("Duplicate Rows", f"{after['duplicates']:,}", delta=f"-{before['duplicates']-after['duplicates']}")
            st.metric("Quality Score",  f"{q_after}/100",          delta=f"+{q_after-q_raw} pts")

        st.success(f"Done! Removed {before['duplicates']} duplicates, filled {before['missing']} missing values, capped outliers.")
        st.subheader("Cleaned Data Preview")
        st.dataframe(cleaned.head(10), use_container_width=True)
        st.download_button("⬇️ Download Cleaned CSV", cleaned.to_csv(index=False).encode(),
                           "cleaned_dataset.csv", "text/csv")
    else:
        st.info("Click the button above to run the cleaning pipeline.")

# ══════════════════════════════════════════════════════════════════
# TAB 3 — ANALYZE & CHARTS (expanded)
# ══════════════════════════════════════════════════════════════════
with tab_analyze:
    working_df = st.session_state.clean_df if st.session_state.clean_df is not None else df
    target_col = st.session_state.target_col
    problem    = detect_problem_type(working_df, target_col)

    st.header("📈 Dataset Analysis & Visualisations")

    # Problem type banner
    st.subheader(f"{problem['icon']} Problem Type: {problem['type']}")
    st.info(problem["reason"])

    st.divider()
    st.subheader("🤖 Recommended Models")
    for m in MODEL_CATALOG[problem["type"]]:
        with st.expander(f"📌 {m['name']}"):
            st.write(m["desc"])

    st.divider()
    st.subheader("📊 Visualisations")

    # Twelve chart tabs
    (vt1,vt2,vt3,vt4,vt5,vt6,
     vt7,vt8,vt9,vt10,vt11,vt12) = st.tabs([
        "Missing Values",
        "Correlation",
        "Histograms",
        "Boxplots",
        "Violin Plots",
        "KDE Density",
        "Categorical Bars",
        "Pie Charts",
        "Pairplot",
        "Count Plots",
        "Descriptive Stats",
        "Skewness & Kurtosis",
    ])

    def safe_plot(fig):
        if fig:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with vt1:
        st.caption("Cyan cells = missing values. Helps identify patterns in missingness.")
        safe_plot(plot_missing_heatmap(working_df))
        if working_df.isnull().sum().sum() == 0:
            st.success("🎉 No missing values!")

    with vt2:
        st.caption("How strongly numeric columns correlate. +1 = perfect positive, -1 = perfect negative.")
        safe_plot(plot_correlation_heatmap(working_df))

    with vt3:
        st.caption("Distribution of values in each numeric column.")
        n_hist = st.slider("Max columns to show", 2, min(12, working_df.select_dtypes("number").shape[1]), 6, key="hist_n")
        safe_plot(plot_distributions(working_df, max_cols=n_hist))

    with vt4:
        st.caption("Median line, IQR box, whiskers, and outlier dots.")
        n_box = st.slider("Max columns to show", 2, min(12, working_df.select_dtypes("number").shape[1]), 6, key="box_n")
        safe_plot(plot_boxplots(working_df, max_cols=n_box))

    with vt5:
        st.caption("Like a boxplot but also shows the full distribution shape on both sides.")
        n_vio = st.slider("Max columns to show", 2, min(12, working_df.select_dtypes("number").shape[1]), 6, key="vio_n")
        safe_plot(plot_violin(working_df, max_cols=n_vio))

    with vt6:
        st.caption("Smooth density curve overlaid on histogram. Great for seeing multi-modal distributions.")
        n_kde = st.slider("Max columns to show", 2, min(12, working_df.select_dtypes("number").shape[1]), 6, key="kde_n")
        safe_plot(plot_kde(working_df, max_cols=n_kde))

    with vt7:
        cat_count = working_df.select_dtypes("object").shape[1]
        if cat_count == 0:
            st.warning("No categorical columns found.")
        else:
            st.caption("Top 10 values in each categorical column.")
            n_cat = st.slider("Max columns to show", 1, min(9, cat_count), min(6, cat_count), key="cat_n")
            safe_plot(plot_categorical_bar(working_df, max_cols=n_cat))

    with vt8:
        cat_cols_pie = list(working_df.select_dtypes("object").columns)
        if not cat_cols_pie:
            st.warning("No categorical columns found.")
        else:
            pie_col = st.selectbox("Select column for pie chart", cat_cols_pie, key="pie_col")
            safe_plot(plot_pie_chart(working_df, pie_col))

    with vt9:
        num_count = working_df.select_dtypes("number").shape[1]
        if num_count < 2:
            st.warning("Need at least 2 numeric columns for a pairplot.")
        else:
            st.caption("Scatter matrix — each numeric column plotted against every other.")
            pair_max = st.slider("Max numeric columns", 2, min(6, num_count), min(4, num_count), key="pair_n")
            hue_opts = ["None"] + list(working_df.select_dtypes("object").columns)
            pair_hue = st.selectbox("Colour by (optional)", hue_opts, key="pair_hue")
            if st.button("Generate Pairplot", key="gen_pair"):
                with st.spinner("Building pairplot — may take a few seconds..."):
                    fig = plot_pairplot(working_df, max_cols=pair_max,
                                        hue_col=None if pair_hue == "None" else pair_hue)
                    safe_plot(fig)

    with vt10:
        cat_cols_cnt = list(working_df.select_dtypes("object").columns)
        if not cat_cols_cnt:
            st.warning("No categorical columns found.")
        else:
            cnt_col  = st.selectbox("Column to count", cat_cols_cnt, key="cnt_col")
            hue_opts2= ["None"] + [c for c in cat_cols_cnt if c != cnt_col]
            cnt_hue  = st.selectbox("Hue (optional)", hue_opts2, key="cnt_hue")
            safe_plot(plot_count_plot(working_df, cnt_col,
                                       hue_col=None if cnt_hue == "None" else cnt_hue))

    with vt11:
        st.caption("Summary statistics for all numeric columns.")
        st.dataframe(working_df.describe().T.round(3), use_container_width=True)

    with vt12:
        st.caption("Skewness > 1 or < -1 means heavily skewed. Kurtosis > 3 means heavy tails (many outliers).")
        num_df = working_df.select_dtypes(include="number")
        if num_df.empty:
            st.warning("No numeric columns.")
        else:
            sk_df = pd.DataFrame({
                "Column":   num_df.columns,
                "Mean":     num_df.mean().round(3).values,
                "Std Dev":  num_df.std().round(3).values,
                "Skewness": num_df.skew().round(3).values,
                "Kurtosis": num_df.kurtosis().round(3).values,
            })
            st.dataframe(sk_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# TAB 4 — CUSTOM CHART BUILDER
# ══════════════════════════════════════════════════════════════════
with tab_custom:
    working_df = st.session_state.clean_df if st.session_state.clean_df is not None else df

    st.header("🎨 Custom Chart Builder")
    st.caption("Pick a chart type, select your columns, and generate the chart instantly.")

    num_cols = list(working_df.select_dtypes(include="number").columns)
    cat_cols = list(working_df.select_dtypes(include="object").columns)
    all_cols = list(working_df.columns)

    # Chart type picker
    chart_type = st.selectbox("Chart Type", list(CHART_TYPES.keys()))

    info = CHART_TYPES[chart_type]
    st.caption(f"**Required:** {', '.join(info['needs']) if info['needs'] else 'None'}"
               + (f"  |  **Optional:** {', '.join(info['optional'])}" if info["optional"] else ""))

    params = {}
    col_left, col_right = st.columns(2)

    # ── Parameter inputs per chart type ──────────────────────
    with col_left:
        if chart_type == "Scatter Plot":
            params["x"]   = st.selectbox("X axis (numeric)", num_cols, key="sc_x")
            params["y"]   = st.selectbox("Y axis (numeric)", [c for c in num_cols if c != params["x"]], key="sc_y")

        elif chart_type in ("Line Chart", "Area Chart"):
            params["x"]      = st.selectbox("X axis", all_cols, key="lc_x")
            params["y_cols"] = st.multiselect("Y columns (numeric)", num_cols, default=num_cols[:2], key="lc_y")

        elif chart_type == "Bar Chart":
            params["x"]   = st.selectbox("X axis (categorical)", cat_cols or all_cols, key="bc_x")

        elif chart_type == "Histogram":
            params["col"] = st.selectbox("Column (numeric)", num_cols, key="hi_col")

        elif chart_type == "Box Plot":
            params["y"]   = st.selectbox("Y axis (numeric)", num_cols, key="bp_y")

        elif chart_type in ("Violin Plot", "KDE Density"):
            params["cols"] = st.multiselect("Columns (numeric)", num_cols, default=num_cols[:4], key="vio_cols")

        elif chart_type == "Pie Chart":
            params["col"] = st.selectbox("Column (categorical)", cat_cols or all_cols, key="pie_cb")

        elif chart_type == "Count Plot":
            params["col"] = st.selectbox("Column (categorical)", cat_cols or all_cols, key="cp_col")

        elif chart_type == "Correlation Heatmap":
            st.info("No column selection needed — uses all numeric columns automatically.")

        elif chart_type == "Pairplot":
            params["cols"] = st.multiselect("Numeric columns", num_cols, default=num_cols[:4], key="pp_cols")

        elif chart_type == "Pivot Heatmap":
            params["x"] = st.selectbox("X axis (categorical)", cat_cols or all_cols, key="ph_x")
            params["y"] = st.selectbox("Y axis (categorical)", [c for c in (cat_cols or all_cols) if c != params.get("x")], key="ph_y")

    with col_right:
        # Optional parameters
        if chart_type == "Scatter Plot":
            hue_opts  = ["None"] + cat_cols
            size_opts = ["None"] + num_cols
            h = st.selectbox("Hue / Colour by (optional)", hue_opts, key="sc_hue")
            s = st.selectbox("Size by (optional)",          size_opts, key="sc_size")
            if h != "None": params["hue"]  = h
            if s != "None": params["size"] = s

        elif chart_type == "Bar Chart":
            y_opts = ["None"] + num_cols
            y_sel  = st.selectbox("Y axis / Value column (optional)", y_opts, key="bc_y")
            if y_sel != "None":
                params["y"]   = y_sel
                params["agg"] = st.selectbox("Aggregation", ["mean", "sum", "median", "count"], key="bc_agg")

        elif chart_type == "Histogram":
            params["bins"] = st.slider("Number of bins", 5, 100, 30, key="hi_bins")
            params["kde"]  = st.checkbox("Overlay KDE curve", value=True, key="hi_kde")

        elif chart_type == "Box Plot":
            x_opts = ["None"] + cat_cols
            x_sel  = st.selectbox("Group by / X axis (optional categorical)", x_opts, key="bp_x")
            if x_sel != "None": params["x"] = x_sel

        elif chart_type == "Count Plot":
            hue_opts3 = ["None"] + [c for c in cat_cols if c != params.get("col")]
            h3 = st.selectbox("Hue (optional)", hue_opts3, key="cp_hue")
            if h3 != "None": params["hue"] = h3

        elif chart_type == "Pairplot":
            hue_opts4 = ["None"] + cat_cols
            h4 = st.selectbox("Colour by (optional)", hue_opts4, key="pp_hue")
            if h4 != "None": params["hue"] = h4

        elif chart_type == "Pivot Heatmap":
            val_opts = ["None (use count)"] + num_cols
            v = st.selectbox("Value column (optional)", val_opts, key="ph_val")
            if v != "None (use count)": params["val"] = v

    st.divider()

    if st.button("🎨 Generate Chart", use_container_width=True, key="gen_custom"):
        with st.spinner("Building chart..."):
            fig = build_custom_chart(working_df, chart_type, params)
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# ══════════════════════════════════════════════════════════════════
# TAB 5 — TRAIN & COMPARE
# ══════════════════════════════════════════════════════════════════
with tab_train:
    working_df = st.session_state.clean_df if st.session_state.clean_df is not None else df
    target_col = st.session_state.target_col
    problem    = detect_problem_type(working_df, target_col)

    st.header("🚀 Train & Compare Models")
    st.caption(f"Training all recommended models for **{problem['type']}** with **{split_pct}/{100-split_pct}** split.")

    if st.button("🚀 Train All Models & Compare", use_container_width=True):
        if problem["type"] != "Clustering" and target_col is None:
            st.error("⚠️ Select a target column in the sidebar first.")
            st.stop()
        try:
            if problem["type"] == "Clustering":
                X, feat_names, _, _ = encode_and_scale(working_df)
                y = None
            else:
                X, feat_names, _, _ = encode_and_scale(working_df, target_col)
                raw_y = working_df[target_col]
                if problem["type"] == "Classification":
                    le = LabelEncoder()
                    y  = le.fit_transform(raw_y.astype(str))
                else:
                    y = raw_y.values

            with st.spinner("Training all models..."):
                comparison = train_all_models(X, y, problem["type"], test_size)
                st.session_state.comparison  = comparison
                st.session_state.feat_names  = feat_names
                st.session_state.X           = X
                st.session_state.y           = y
                card = dataset_report_card(working_df, target_col, problem, comparison)
                st.session_state.report_card = card

        except Exception as e:
            st.error(f"Training failed: {e}")
            st.exception(e)

    if st.session_state.comparison is not None:
        comparison = st.session_state.comparison
        results    = comparison["results"]

        st.subheader("📊 Model Comparison Table")
        table_data = []
        for r in results:
            row = {"Model": r["name"], r["metric"]: f"{r['value']}{r['unit']}",
                   "Train Samples": f"{r['train_sz']:,}",
                   "Test Samples": f"{r['test_sz']:,}" if r["test_sz"] > 0 else "N/A",
                   "Best?": "⭐ Yes" if r["name"] == comparison["best"] else ""}
            if "r2" in r: row["R²"] = r["r2"]
            table_data.append(row)
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
        st.success(f"⭐ Best model: **{comparison['best']}**")
        st.divider()

        fig = plot_model_comparison(comparison, problem["type"])
        if fig:
            st.subheader("📊 Visual Comparison")
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.divider()

        if (problem["type"] == "Classification"
                and comparison["y_test"] is not None
                and comparison["best_preds"] is not None):
            st.subheader("🔲 Confusion Matrix (Best Model)")
            st.caption("Diagonal = correctly predicted. Off-diagonal = mistakes.")
            fig = plot_confusion_matrix(comparison["y_test"], comparison["best_preds"])
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.divider()

        X = st.session_state.X
        y = st.session_state.y
        feat_names = st.session_state.feat_names
        if (problem["type"] == "Classification" and y is not None
                and feat_names and len(feat_names) <= 30):
            st.subheader("🌟 Feature Importances")
            imp = get_feature_importances(X, y, feat_names)
            fig = plot_feature_importances(imp)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
    else:
        st.info("Click the button above to train and compare all models.")

# ══════════════════════════════════════════════════════════════════
# TAB 6 — REPORT CARD
# ══════════════════════════════════════════════════════════════════
with tab_report:
    working_df = st.session_state.clean_df if st.session_state.clean_df is not None else df
    target_col = st.session_state.target_col
    problem    = detect_problem_type(working_df, target_col)

    st.header("📋 Dataset Report Card")
    st.caption("Full summary — overview, quality, correlations, problem type, model results.")

    card = (st.session_state.report_card
            or dataset_report_card(working_df, target_col, problem, st.session_state.comparison))

    # Overview
    st.subheader("📁 Dataset Overview")
    ov = card["overview"]
    o1,o2,o3,o4,o5,o6 = st.columns(6)
    o1.metric("Rows", f"{ov['rows']:,}")
    o2.metric("Columns", ov["cols"])
    o3.metric("Missing", f"{ov['missing']:,}")
    o4.metric("Duplicates", f"{ov['duplicates']:,}")
    o5.metric("Numeric", ov["numeric"])
    o6.metric("Categorical", ov["categorical"])

    # Quality
    st.subheader("⭐ Data Quality Score")
    q = card["quality_score"]
    st.metric("Quality Score", f"{q} / 100")
    st.progress(q / 100)
    if q >= 80:   st.success("Great quality! Ready for modelling.")
    elif q >= 50: st.warning("Moderate quality — consider cleaning first.")
    else:         st.error("Low quality — clean the data before modelling.")

    st.divider()

    # Missing values
    st.subheader("🔍 Missing Values")
    if card["missing_pct"]:
        st.dataframe(pd.DataFrame(list(card["missing_pct"].items()), columns=["Column", "Missing %"]),
                     use_container_width=True, hide_index=True)
    else:
        st.success("✅ No missing values.")

    st.divider()

    # Top correlations
    st.subheader("🔗 Top Feature Correlations")
    if card["top_correlations"]:
        st.dataframe(pd.DataFrame(card["top_correlations"]), use_container_width=True, hide_index=True)
    else:
        st.info("Not enough numeric columns.")

    st.divider()

    # Problem type
    st.subheader("🤖 Detected Problem Type")
    pt = card["problem_type"]
    if pt:
        p1, p2 = st.columns(2)
        p1.metric("Type", f"{pt.get('icon','')} {pt.get('type','N/A')}")
        p2.metric("Target", card["target_col"])
        st.info(pt.get("reason", ""))

    st.divider()

    # Model results
    st.subheader("🚀 Model Results")
    if card["model_results"]:
        st.success(f"⭐ Best: **{card['best_model']}**")
        st.dataframe(pd.DataFrame([{
            "Model": r["name"], "Metric": r["metric"],
            "Score": f"{r['value']}{r['unit']}",
            "Best?": "⭐" if r["name"] == card["best_model"] else "",
        } for r in card["model_results"]]), use_container_width=True, hide_index=True)
    else:
        st.info("Train models first (Train & Compare tab).")

    st.divider()

    # Export
    st.subheader("⬇️ Export Report")
    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button("📄 Download TXT", generate_report_text(card).encode(),
                           "data_report.txt", "text/plain", use_container_width=True)
    with e2:
        st.download_button("📝 Download Markdown", generate_report_markdown(card).encode(),
                           "data_report.md", "text/markdown", use_container_width=True)
    with e3:
        if st.button("📑 Generate PDF", use_container_width=True):
            with st.spinner("Generating PDF..."):
                pdf_bytes = generate_report_pdf(card)
            st.download_button("⬇️ Download PDF", pdf_bytes, "data_report.pdf",
                               "application/pdf", use_container_width=True, key="pdf_dl")

    with st.expander("👀 Preview Report (Text)"):
        st.text(generate_report_text(card))

# ──────────────────────────────────────────────────────────────────
# PERSISTENT DOWNLOAD
# ──────────────────────────────────────────────────────────────────
if st.session_state.clean_df is not None:
    st.divider()
    st.download_button("⬇️ Download Cleaned Dataset",
                       st.session_state.clean_df.to_csv(index=False).encode(),
                       "cleaned_dataset.csv", "text/csv", key="bottom_dl")
