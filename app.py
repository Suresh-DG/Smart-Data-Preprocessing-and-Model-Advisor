import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

from main import (
    load_dataset, load_from_url, dataset_summary, compute_quality_score,
    clean_data, encode_and_scale, detect_problem_type,
    MODEL_CATALOG, train_all_models, get_feature_importances,
    dataset_report_card, generate_report_text, generate_report_markdown, generate_report_pdf,
    plot_missing_heatmap, plot_correlation_heatmap, plot_distributions, plot_boxplots,
    plot_violin, plot_kde, plot_categorical_bar, plot_pie_chart, plot_pairplot, plot_count_plot,
    plot_feature_importances, plot_model_comparison, plot_confusion_matrix,
    CHART_TYPES, build_custom_chart,
)

st.set_page_config(page_title="Smart Data Preprocessing and Model Advisor", page_icon="📊", layout="wide")

st.markdown("""<style>
.stApp { background-color: #f0f4f8; }
section[data-testid="stSidebar"] { background-color: #dce7f3; }
.stTabs [data-baseweb="tab-list"] { background-color: #dce7f3; border-radius: 8px; padding: 4px; }
.stTabs [aria-selected="true"] { background-color: #2563eb !important; color: white !important; border-radius: 6px; }
div[data-testid="metric-container"] { background-color: #fff; border: 1px solid #c8d8ea; border-radius: 8px; padding: 12px 16px; box-shadow: 0 1px 4px rgba(37,99,235,0.08); }
div[data-testid="stExpander"] { background-color: #fff; border: 1px solid #c8d8ea; border-radius: 8px; }
.stButton > button { background-color: #2563eb; color: white; border: none; border-radius: 6px; font-weight: 600; }
.stButton > button:hover { background-color: #1d4ed8; color: white; }
div[data-testid="stDataFrame"] { border: 1px solid #c8d8ea; border-radius: 8px; overflow: hidden; }
</style>""", unsafe_allow_html=True)

for k in ["raw_df","clean_df","target_col","comparison","report_card","feat_names","X","y"]:
    if k not in st.session_state: st.session_state[k] = None

def show(fig):
    if fig: st.pyplot(fig, use_container_width=True); plt.close(fig)

def wdf(): return st.session_state.clean_df if st.session_state.clean_df is not None else st.session_state.raw_df

# SIDEBAR
with st.sidebar:
    st.title("📊 Smart Data Advisor")
    st.caption("Upload or import a dataset, then clean, analyze, and train models.")
    st.divider()

    #Load dataset
    st.subheader("1. Load Dataset")
    mode = st.radio("Import method", ["📁 Upload File", "🔗 Import from URL"], horizontal=True)

    if mode == "📁 Upload File":
        f = st.file_uploader("Choose a CSV or Excel file", type=["csv","xlsx","xls"])
        if f:
            try:
                df_new = load_dataset(f)
                st.session_state.update({"raw_df": df_new, "clean_df": None, "comparison": None, "report_card": None})
                st.success(f"Loaded {df_new.shape[0]:,} rows × {df_new.shape[1]} cols")
            except ValueError as e: st.error(str(e))
    else:
        with st.expander("📖 URL Tips"):
            st.markdown("**GitHub:** change `blob` → `raw`\n\n**Google Sheets:** File → Share → Publish as CSV\n\n**Kaggle:** raw file download link from Files tab")
        url = st.text_input("Dataset URL", placeholder="https://...")
        if st.button("⬇️ Load from URL", use_container_width=True):
            if not url.strip(): st.error("Please enter a URL.")
            else:
                with st.spinner("Fetching..."):
                    try:
                        df_new = load_from_url(url.strip())
                        st.session_state.update({"raw_df": df_new, "clean_df": None, "comparison": None, "report_card": None})
                        st.success(f"✅ Loaded {df_new.shape[0]:,} rows × {df_new.shape[1]} cols")
                    except ValueError as e: st.error(str(e))
    #Target column
    if st.session_state.raw_df is not None:
        st.divider()
        st.subheader("2. Target Column")
        opts = ["None (Clustering)"] + list(st.session_state.raw_df.columns)
        ch   = st.selectbox("Target column", opts)
        st.session_state.target_col = None if ch == "None (Clustering)" else ch
    st.divider()
    #Train/test split
    st.subheader("3. Train / Test Split")
    split_pct = st.slider("Training data %", 60, 90, 80, step=10)
    test_size = round(1 - split_pct / 100, 2)
    st.caption(f"→ **{split_pct}% train** / **{100-split_pct}% test**")
    st.divider()
    st.markdown("**Steps:** Load → Select Target → Clean → Analyze → Custom Charts → Train → Report")
st.title("📊 Smart Data Preprocessing and Model Advisor")
st.caption("Load · Clean · Visualize · Train · Export — all in one place.")

if st.session_state.raw_df is None:
    st.info("👈 Load a dataset from the sidebar to get started.")
    st.stop()

df = st.session_state.raw_df
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📁 Overview","🧹 Clean Data","📈 Analyze & Charts","🎨 Custom Charts","🚀 Train & Compare","📋 Report Card"])

#OVERVIEW
with tab1:
    st.header("Dataset Overview")
    s = dataset_summary(df); q = compute_quality_score(df)
    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("Rows", f"{s['rows']:,}"); c2.metric("Columns", s['cols'])
    c3.metric("Missing", f"{s['missing']:,}"); c4.metric("Duplicates", f"{s['duplicates']:,}")
    c5.metric("Numeric", s['numeric']); c6.metric("Categorical", s['categorical']); c7.metric("Quality", f"{q}/100")
    st.divider()
    with st.expander("View Column Details"):
        st.dataframe(pd.DataFrame({
            "Column": df.columns, "Type": [str(df[c].dtype) for c in df.columns],
            "Non-Null": [int(df[c].count()) for c in df.columns],
            "Missing": [int(df[c].isnull().sum()) for c in df.columns],
            "Unique": [int(df[c].nunique()) for c in df.columns],
            "Sample": [str(df[c].dropna().iloc[0]) if df[c].count()>0 else "N/A" for c in df.columns],
        }), use_container_width=True, hide_index=True)
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

#CLEAN DATA
with tab2:
    st.header("🧹 Data Cleaning")
    st.caption("Fixes missing values (median/mode), removes duplicates, caps outliers via IQR.")
    if st.button("🧹 Run Cleaning Pipeline", use_container_width=True):
        with st.spinner("Cleaning..."):
            st.session_state.clean_df = clean_data(df)

    if st.session_state.clean_df is not None:
        cleaned = st.session_state.clean_df
        b = dataset_summary(df); a = dataset_summary(cleaned)
        qb = compute_quality_score(df); qa = compute_quality_score(cleaned)

        st.subheader("Before vs After")
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Before**")
            st.metric("Missing", f"{b['missing']:,}"); st.metric("Duplicates", f"{b['duplicates']:,}"); st.metric("Quality", f"{qb}/100")
        with cb:
            st.markdown("**After**")
            st.metric("Missing", f"{a['missing']:,}", delta=f"-{b['missing']-a['missing']}")
            st.metric("Duplicates", f"{a['duplicates']:,}", delta=f"-{b['duplicates']-a['duplicates']}")
            st.metric("Quality", f"{qa}/100", delta=f"+{qa-qb} pts")

        st.success(f"Done! Removed {b['duplicates']} duplicates, filled {b['missing']} missing values, capped outliers.")
        st.subheader("Cleaned Data Preview")
        st.dataframe(cleaned.head(10), use_container_width=True)
        st.download_button("⬇️ Download Cleaned CSV", cleaned.to_csv(index=False).encode(), "cleaned_dataset.csv", "text/csv")
    else:
        st.info("Click the button above to run the cleaning pipeline.")

#ANALYZE & CHARTS
with tab3:
    wd = wdf(); tc = st.session_state.target_col; prob = detect_problem_type(wd, tc)
    st.header("📈 Dataset Analysis & Visualisations")
    st.subheader(f"{prob['icon']} Problem Type: {prob['type']}"); st.info(prob["reason"])
    st.divider()
    st.subheader("🤖 Recommended Models")
    for m in MODEL_CATALOG[prob["type"]]:
        with st.expander(f"📌 {m['name']}"): st.write(m["desc"])
    st.divider()
    st.subheader("📊 Visualisations")
    vt = st.tabs(["Missing","Correlation","Histograms","Boxplots","Violin","KDE","Cat Bars","Pie","Pairplot","Count","Stats","Skew/Kurt"])
    num_n = wd.select_dtypes("number").shape[1]
    cat_n = wd.select_dtypes("object").shape[1]
    with vt[0]:
        st.caption("Cyan = missing value."); show(plot_missing_heatmap(wd))
        if wd.isnull().sum().sum()==0: st.success("🎉 No missing values!")
    with vt[1]:
        st.caption("Correlation between numeric columns."); show(plot_correlation_heatmap(wd))
    with vt[2]:
        n = st.slider("Max columns", 2, min(12, num_n), 6, key="hn") if num_n >= 2 else 6
        show(plot_distributions(wd, max_cols=n))
    with vt[3]:
        n = st.slider("Max columns", 2, min(12, num_n), 6, key="bn") if num_n >= 2 else 6
        show(plot_boxplots(wd, max_cols=n))
    with vt[4]:
        n = st.slider("Max columns", 2, min(12, num_n), 6, key="vn") if num_n >= 2 else 6
        show(plot_violin(wd, max_cols=n))
    with vt[5]:
        n = st.slider("Max columns", 2, min(12, num_n), 6, key="kn") if num_n >= 2 else 6
        show(plot_kde(wd, max_cols=n))
    with vt[6]:
        if cat_n == 0: st.warning("No categorical columns.")
        else:
            n = st.slider("Max columns", 1, min(9, cat_n), min(6, cat_n), key="cn")
            show(plot_categorical_bar(wd, max_cols=n))
    with vt[7]:
        cats = list(wd.select_dtypes("object").columns)
        if not cats: st.warning("No categorical columns.")
        else: show(plot_pie_chart(wd, st.selectbox("Column", cats, key="pc")))
    with vt[8]:
        if num_n < 2: st.warning("Need at least 2 numeric columns.")
        else:
            pm = st.slider("Max columns", 2, min(6, num_n), min(4, num_n), key="pn")
            ph = st.selectbox("Colour by", ["None"]+list(wd.select_dtypes("object").columns), key="ph")
            if st.button("Generate Pairplot"):
                with st.spinner("Building..."): show(plot_pairplot(wd, pm, None if ph=="None" else ph))
    with vt[9]:
        cats2 = list(wd.select_dtypes("object").columns)
        if not cats2: st.warning("No categorical columns.")
        else:
            cc = st.selectbox("Column", cats2, key="cc")
            ch2 = st.selectbox("Hue", ["None"]+[c for c in cats2 if c!=cc], key="ch")
            show(plot_count_plot(wd, cc, None if ch2=="None" else ch2))
    with vt[10]:
        st.caption("Mean, std, min, max, quartiles."); st.dataframe(wd.describe().T.round(3), use_container_width=True)
    with vt[11]:
        num_df = wd.select_dtypes("number")
        if num_df.empty: st.warning("No numeric columns.")
        else:
            st.dataframe(pd.DataFrame({"Column": num_df.columns, "Mean": num_df.mean().round(3).values,
                "Std": num_df.std().round(3).values, "Skewness": num_df.skew().round(3).values,
                "Kurtosis": num_df.kurtosis().round(3).values}), use_container_width=True, hide_index=True)

#CUSTOM CHART BUILDER
with tab4:
    wd = wdf()
    st.header("🎨 Custom Chart Builder")
    st.caption("Pick a chart type, select your columns, and generate the chart instantly.")

    num_cols = list(wd.select_dtypes("number").columns)
    cat_cols = list(wd.select_dtypes("object").columns)
    all_cols = list(wd.columns)

    ct = st.selectbox("Chart Type", list(CHART_TYPES.keys()))
    info = CHART_TYPES[ct]
    st.caption(f"**Required:** {', '.join(info['needs']) or 'None'}" + (f"  |  **Optional:** {', '.join(info['optional'])}" if info["optional"] else ""))

    params = {}
    cl, cr = st.columns(2)

    with cl:
        if ct == "Scatter Plot":
            params["x"] = st.selectbox("X (numeric)", num_cols, key="sx")
            params["y"] = st.selectbox("Y (numeric)", [c for c in num_cols if c!=params["x"]], key="sy")
        elif ct in ("Line Chart","Area Chart"):
            params["x"] = st.selectbox("X axis", all_cols, key="lx")
            params["y_cols"] = st.multiselect("Y columns", num_cols, default=num_cols[:2], key="ly")
        elif ct == "Bar Chart":
            params["x"] = st.selectbox("X (categorical)", cat_cols or all_cols, key="bx")
        elif ct == "Histogram":
            params["col"] = st.selectbox("Column (numeric)", num_cols, key="hc")
        elif ct == "Box Plot":
            params["y"] = st.selectbox("Y (numeric)", num_cols, key="by")
        elif ct in ("Violin Plot","KDE Density"):
            params["cols"] = st.multiselect("Columns (numeric)", num_cols, default=num_cols[:4], key="vc")
        elif ct == "Pie Chart":
            params["col"] = st.selectbox("Column (categorical)", cat_cols or all_cols, key="pyc")
        elif ct == "Count Plot":
            params["col"] = st.selectbox("Column (categorical)", cat_cols or all_cols, key="cpc")
        elif ct == "Correlation Heatmap":
            st.info("Uses all numeric columns automatically.")
        elif ct == "Pairplot":
            params["cols"] = st.multiselect("Numeric columns", num_cols, default=num_cols[:4], key="ppc")
        elif ct == "Pivot Heatmap":
            params["x"] = st.selectbox("X (categorical)", cat_cols or all_cols, key="phx")
            params["y"] = st.selectbox("Y (categorical)", [c for c in (cat_cols or all_cols) if c!=params.get("x")], key="phy")

    with cr:
        if ct == "Scatter Plot":
            h = st.selectbox("Hue (optional)", ["None"]+cat_cols, key="sh")
            sz = st.selectbox("Size (optional)", ["None"]+num_cols, key="ss")
            if h!="None": params["hue"]=h
            if sz!="None": params["size"]=sz
        elif ct == "Bar Chart":
            y2 = st.selectbox("Y / Value (optional)", ["None"]+num_cols, key="by2")
            if y2!="None": params["y"]=y2; params["agg"]=st.selectbox("Aggregation",["mean","sum","median","count"],key="agg")
        elif ct == "Histogram":
            params["bins"] = st.slider("Bins", 5, 100, 30, key="hb")
            params["kde"]  = st.checkbox("KDE overlay", True, key="hk")
        elif ct == "Box Plot":
            xg = st.selectbox("Group by (optional)", ["None"]+cat_cols, key="bxg")
            if xg!="None": params["x"]=xg
        elif ct == "Count Plot":
            ch3 = st.selectbox("Hue (optional)", ["None"]+[c for c in cat_cols if c!=params.get("col")], key="cph")
            if ch3!="None": params["hue"]=ch3
        elif ct == "Pairplot":
            ph2 = st.selectbox("Colour by (optional)", ["None"]+cat_cols, key="pph")
            if ph2!="None": params["hue"]=ph2
        elif ct == "Pivot Heatmap":
            v = st.selectbox("Value column (optional)", ["None"]+num_cols, key="phv")
            if v!="None": params["val"]=v
    st.divider()
    if st.button("🎨 Generate Chart", use_container_width=True):
        with st.spinner("Building..."): show(build_custom_chart(wd, ct, params))

#TRAIN & COMPARE
with tab5:
    wd = wdf(); tc = st.session_state.target_col; prob = detect_problem_type(wd, tc)
    st.header("🚀 Train & Compare Models")
    st.caption(f"All recommended models for **{prob['type']}** — **{split_pct}/{100-split_pct}** split.")

    if st.button("🚀 Train All Models & Compare", use_container_width=True):
        if prob["type"] != "Clustering" and tc is None:
            st.error("⚠️ Select a target column in the sidebar first."); st.stop()
        try:
            if prob["type"] == "Clustering":
                X, feat_names, _ = encode_and_scale(wd); y = None
            else:
                X, feat_names, _ = encode_and_scale(wd, tc)
                y = LabelEncoder().fit_transform(wd[tc].astype(str)) if prob["type"]=="Classification" else wd[tc].values
            with st.spinner("Training all models..."):
                comp = train_all_models(X, y, prob["type"], test_size)
                st.session_state.update({"comparison": comp, "feat_names": feat_names, "X": X, "y": y,
                                          "report_card": dataset_report_card(wd, tc, prob, comp)})
        except Exception as e:
            st.error(f"Training failed: {e}"); st.exception(e)
    if st.session_state.comparison:
        comp = st.session_state.comparison; res = comp["results"]
        st.subheader("📊 Model Comparison Table")
        rows = [{"Model": r["name"], r["metric"]: f"{r['value']}{r['unit']}",
                 "Train": f"{r['train_sz']:,}", "Test": f"{r['test_sz']:,}" if r["test_sz"]>0 else "N/A",
                 "Best?": "⭐" if r["name"]==comp["best"] else ""} for r in res]
        if "r2" in res[0]: [row.update({"R²": r.get("r2","")}) for row, r in zip(rows, res)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.success(f"⭐ Best model: **{comp['best']}**"); st.divider()

        show(plot_model_comparison(comp, prob["type"])); st.divider()

        if prob["type"]=="Classification" and comp["y_test"] is not None:
            st.subheader("🔲 Confusion Matrix"); st.caption("Diagonal = correct predictions.")
            show(plot_confusion_matrix(comp["y_test"], comp["best_preds"])); st.divider()

        X, y, fn = st.session_state.X, st.session_state.y, st.session_state.feat_names
        if prob["type"]=="Classification" and y is not None and fn and len(fn)<=30:
            st.subheader("🌟 Feature Importances")
            show(plot_feature_importances(get_feature_importances(X, y, fn)))
    else:
        st.info("Click the button above to train and compare all models.")
#REPORT CARD
with tab6:
    wd = wdf(); tc = st.session_state.target_col; prob = detect_problem_type(wd, tc)
    st.header("📋 Dataset Report Card")
    card = st.session_state.report_card or dataset_report_card(wd, tc, prob, st.session_state.comparison)
    st.subheader("📁 Overview")
    ov = card["overview"]
    o1,o2,o3,o4,o5,o6 = st.columns(6)
    o1.metric("Rows",f"{ov['rows']:,}"); o2.metric("Cols",ov["cols"]); o3.metric("Missing",f"{ov['missing']:,}")
    o4.metric("Dupes",f"{ov['duplicates']:,}"); o5.metric("Numeric",ov["numeric"]); o6.metric("Categorical",ov["categorical"])
    st.subheader("⭐ Quality Score")
    q = card["quality_score"]; st.metric("Score", f"{q} / 100"); st.progress(q/100)
    if q>=80: st.success("Great quality! Ready for modelling.")
    elif q>=50: st.warning("Moderate quality — consider cleaning first.")
    else: st.error("Low quality — clean the data before modelling.")
    st.divider()
    st.subheader("🔍 Missing Values")
    if card["missing_pct"]:
        st.dataframe(pd.DataFrame(card["missing_pct"].items(), columns=["Column","Missing %"]), use_container_width=True, hide_index=True)
    else: st.success("✅ No missing values.")
    st.divider()
    st.subheader("🔗 Top Correlations")
    if card["top_correlations"]: st.dataframe(pd.DataFrame(card["top_correlations"]), use_container_width=True, hide_index=True)
    else: st.info("Not enough numeric columns.")
    st.divider()
    st.subheader("🤖 Problem Type")
    pt = card["problem_type"]
    if pt:
        p1, p2 = st.columns(2)
        p1.metric("Type", f"{pt.get('icon','')} {pt.get('type','')}"); p2.metric("Target", card["target_col"])
        st.info(pt.get("reason",""))
    st.divider()
    st.subheader("🚀 Model Results")
    if card["model_results"]:
        st.success(f"⭐ Best: **{card['best_model']}**")
        st.dataframe(pd.DataFrame([{"Model":r["name"],"Metric":r["metric"],"Score":f"{r['value']}{r['unit']}",
                                     "Best?":"⭐" if r["name"]==card["best_model"] else ""} for r in card["model_results"]]),
                     use_container_width=True, hide_index=True)
    else: st.info("Train models first (Train & Compare tab).")
    st.divider()
    st.subheader("⬇️ Export Report")
    e1, e2, e3 = st.columns(3)
    with e1: st.download_button("📄 TXT", generate_report_text(card).encode(), "data_report.txt", "text/plain", use_container_width=True)
    with e2: st.download_button("📝 Markdown", generate_report_markdown(card).encode(), "data_report.md", "text/markdown", use_container_width=True)
    with e3:
        if st.button("📑 Generate PDF", use_container_width=True):
            with st.spinner("Generating PDF..."): pdf = generate_report_pdf(card)
            st.download_button("⬇️ Download PDF", pdf, "data_report.pdf", "application/pdf", use_container_width=True, key="pdf_dl")
    with st.expander("👀 Preview Report"):
        st.text(generate_report_text(card))

#Cleaned Dataset Download
if st.session_state.clean_df is not None:
    st.divider()
    st.download_button("⬇️ Download Cleaned Dataset", st.session_state.clean_df.to_csv(index=False).encode(),
                       "cleaned_dataset.csv", "text/csv", key="bot_dl")
