import io, warnings
from datetime import datetime
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

BG, BG_AX = "#111624", "#0a0d14"
BORDER, TEXT, MUTED = "#242d42", "#e4eaf6", "#6b7a99"
CYAN, PURPLE, RED, YELLOW, GREEN = "#00e5ff", "#7b61ff", "#ff6b6b", "#ffd166", "#00d97e"
PALETTE = [CYAN, PURPLE, GREEN, YELLOW, RED, "#f953c6", "#f7971e"]

def _fig(size=(10, 4)):
    return plt.figure(figsize=size, facecolor=BG)

def _ax(ax):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ax.spines.values(): s.set_edgecolor(BORDER)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    return ax

def _grid_plot(cols_list, plot_fn, title, figsize_row=3.5, hspace=0.45):
    if not cols_list: return None
    nc = min(3, len(cols_list))
    nr = (len(cols_list) + nc - 1) // nc
    fig = _fig((14, figsize_row * nr))
    gs  = gridspec.GridSpec(nr, nc, figure=fig, hspace=hspace, wspace=0.35)
    for i, col in enumerate(cols_list):
        ax = fig.add_subplot(gs[i // nc, i % nc])
        _ax(ax)
        plot_fn(ax, col)
    fig.suptitle(title, color=TEXT, fontsize=13, y=1.01)
    return fig

#DATA LOADING
def load_dataset(f):
    name = f.name.lower()
    if name.endswith(".csv"):   df = pd.read_csv(f)
    elif name.endswith((".xlsx", ".xls")): df = pd.read_excel(f)
    else: raise ValueError("Upload a .csv or .xlsx file.")
    if df.empty: raise ValueError("File is empty.")
    return df
def load_from_url(url):
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    # Auto-fix GitHub and Google Sheets URLs
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    if "docs.google.com/spreadsheets" in url:
        url = url.split("#")[0].replace("/edit", "/export?format=csv")
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
    except requests.exceptions.Timeout:
        raise ValueError("Request timed out.")
    except requests.exceptions.HTTPError as e:
        raise ValueError(f"HTTP error: {e}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Could not reach URL: {e}")
    # Try Excel first, then CSV
    if url.lower().split("?")[0].endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(r.content))
    else:
        try:    df = pd.read_csv(io.StringIO(r.text))
        except: df = pd.read_csv(io.BytesIO(r.content))
    if df.empty: raise ValueError("File at that URL is empty.")
    return df
def dataset_summary(df):
    return {
        "rows": df.shape[0], "cols": df.shape[1],
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numeric": int(df.select_dtypes(include=np.number).shape[1]),
        "categorical": int(df.select_dtypes(exclude=np.number).shape[1]),
    }

#DATA QUALITY SCORE
def compute_quality_score(df):
    total = df.shape[0] * df.shape[1] + 1e-9
    s1 = max(0, 40 * (1 - df.isnull().sum().sum() / total * 5))
    s2 = max(0, 30 * (1 - df.duplicated().sum() / (len(df) + 1e-9) * 5))
    s3 = max(0, 20 - sum(df.nunique() <= 1) * 4)
    s4 = max(0, 10 - sum(
        1 for c in df.select_dtypes("object").columns
        if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.3
    ) * 3)
    return min(100, int(s1 + s2 + s3 + s4))

#DATA CLEANING
def clean_data(df):
    df = df.copy()
    # Fix data types — convert numeric-looking text columns
    for col in df.select_dtypes("object").columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    # Fill missing values
    for col in df.columns:
        if df[col].isnull().sum() == 0: continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)
        else:
            m = df[col].mode()
            if not m.empty: df[col].fillna(m[0], inplace=True)
    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    # Cap outliers using IQR method
    for col in df.select_dtypes(include=np.number).columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        df[col] = df[col].clip(Q1 - 1.5*(Q3-Q1), Q3 + 1.5*(Q3-Q1))
    return df

#FEATURE ENGINEERING
def encode_and_scale(df, target_col=None):
    df = df.copy()
    X  = df.drop(columns=[target_col]) if target_col else df
    # Label encode low-cardinality, one-hot encode high-cardinality
    for col in X.select_dtypes("object").columns:
        if X[col].nunique() <= 20:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        else:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
    X = X.select_dtypes(include=np.number)
    scaler = StandardScaler()
    return scaler.fit_transform(X), list(X.columns), scaler

#PROBLEM TYPE DETECTION
def detect_problem_type(df, target_col=None):
    if not target_col or target_col not in df.columns:
        return {"type": "Clustering",      "reason": "No target selected — finding hidden patterns.", "icon": "🌐"}
    s = df[target_col].dropna()
    if s.dtype == "object" or s.nunique() <= 10:
        return {"type": "Classification", "reason": f"'{target_col}' has {s.nunique()} classes — predicting categories.", "icon": "🏷️"}
    return     {"type": "Regression",     "reason": f"'{target_col}' is continuous ({s.nunique()} unique) — predicting a number.", "icon": "📈"}

#MODEL CATALOG
MODEL_CATALOG = {
    "Classification": [
        {"name": "Random Forest Classifier", "short": "Random Forest", "desc": "Ensemble of 100 trees — robust and accurate."},
        {"name": "Logistic Regression",      "short": "Logistic Reg.", "desc": "Fast linear model — easy to interpret."},
        {"name": "Decision Tree Classifier", "short": "Decision Tree", "desc": "Visual tree of if/else rules — explainable."},
    ],
    "Regression": [
        {"name": "Linear Regression",        "short": "Linear Reg.",  "desc": "Classic baseline — fast and interpretable."},
        {"name": "Decision Tree Regressor",  "short": "Decision Tree","desc": "Captures non-linear patterns without scaling."},
        {"name": "Random Forest Regressor",  "short": "Random Forest","desc": "Ensemble of trees — usually most accurate."},
    ],
    "Clustering": [
        {"name": "K-Means Clustering",       "short": "K-Means",      "desc": "Groups data into K clusters by minimising variance."},
    ],
}

#MODEL TRAINING & COMPARISON
def train_all_models(X, y, problem_type, test_size=0.2):
    results, best_preds, X_test_out, y_test_out = [], None, None, None
    if problem_type == "Classification":
        stratify = y if len(np.unique(y)) > 1 else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
        X_test_out, y_test_out = Xte, yte
        models = [
            ("Random Forest Classifier", "Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("Logistic Regression",      "Logistic Reg.", LogisticRegression(max_iter=1000, random_state=42)),
            ("Decision Tree Classifier", "Decision Tree", DecisionTreeClassifier(random_state=42)),
        ]
        best_val, best_name = -1, ""
        for name, short, mdl in models:
            mdl.fit(Xtr, ytr); pred = mdl.predict(Xte)
            val = round(accuracy_score(yte, pred) * 100, 2)
            results.append({"name": name, "short": short, "metric": "Accuracy", "value": val, "unit": "%", "train_sz": len(Xtr), "test_sz": len(Xte)})
            if val > best_val: best_val, best_name, best_preds = val, name, pred
    elif problem_type == "Regression":
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
        X_test_out, y_test_out = Xte, yte
        models = [
            ("Linear Regression",        "Linear Reg.",  LinearRegression()),
            ("Decision Tree Regressor",  "Decision Tree",DecisionTreeRegressor(random_state=42)),
            ("Random Forest Regressor",  "Random Forest",RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
        best_val, best_name = float("inf"), ""
        for name, short, mdl in models:
            mdl.fit(Xtr, ytr); pred = mdl.predict(Xte)
            val = round(np.sqrt(mean_squared_error(yte, pred)), 4)
            results.append({"name": name, "short": short, "metric": "RMSE", "value": val, "unit": "",
                            "r2": round(r2_score(yte, pred), 4), "train_sz": len(Xtr), "test_sz": len(Xte)})
            if val < best_val: best_val, best_name, best_preds = val, name, pred
    elif problem_type == "Clustering":
        k = min(4, len(X) // 10 + 2)
        mdl = KMeans(n_clusters=k, random_state=42, n_init=10); mdl.fit(X)
        results.append({"name": f"K-Means (k={k})", "short": "K-Means", "metric": "Inertia",
                        "value": round(float(mdl.inertia_), 2), "unit": "", "train_sz": len(X), "test_sz": 0})
        best_name = f"K-Means (k={k})"
    return {"results": results, "best": best_name, "X_test": X_test_out, "y_test": y_test_out, "best_preds": best_preds}
def get_feature_importances(X, y, feat_names, top_n=15):
    mdl = RandomForestClassifier(n_estimators=100, random_state=42)
    mdl.fit(X, y)
    return pd.Series(mdl.feature_importances_, index=feat_names).sort_values(ascending=False).head(top_n)

#REPORT CARD
def dataset_report_card(df, target_col=None, problem=None, comparison=None):
    num_df = df.select_dtypes(include=np.number)
    top_corr = []
    if num_df.shape[1] >= 2:
        cm = num_df.corr().abs()
        np.fill_diagonal(cm.values, 0)
        pairs = cm.stack().reset_index().rename(columns={"level_0": "Feature A", "level_1": "Feature B", 0: "Correlation"})
        pairs = pairs.sort_values("Correlation", ascending=False)
        seen = set()
        for _, row in pairs.iterrows():
            p = tuple(sorted([row["Feature A"], row["Feature B"]]))
            if p not in seen:
                seen.add(p)
                top_corr.append({"Feature A": row["Feature A"], "Feature B": row["Feature B"], "Correlation": round(row["Correlation"], 4)})
            if len(top_corr) >= 10: break
    mp = round(df.isnull().mean() * 100, 2)
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overview": dataset_summary(df),
        "quality_score": compute_quality_score(df),
        "missing_pct": mp[mp > 0].to_dict(),
        "top_correlations": top_corr,
        "problem_type": problem or {},
        "target_col": target_col or "None",
        "model_results": comparison["results"] if comparison else [],
        "best_model": comparison["best"] if comparison else "N/A",
    }

#REPORT EXPORT
def generate_report_text(card):
    ov, sep = card["overview"], "=" * 60
    lines = [sep, "  SMART DATA PREPROCESSING AND MODEL ADVISOR", sep,
             f"  Generated : {card['generated_at']}", f"  Target    : {card['target_col']}", "",
             "DATASET OVERVIEW", "-" * 40,
             f"  Rows: {ov['rows']:,}  |  Columns: {ov['cols']}",
             f"  Missing: {ov['missing']:,}  |  Duplicates: {ov['duplicates']:,}",
             f"  Quality Score: {card['quality_score']} / 100", ""]
    if card["missing_pct"]:
        lines += ["MISSING VALUES", "-" * 40] + [f"  {c:<30} {p:.1f}%" for c, p in card["missing_pct"].items()] + [""]
    if card["top_correlations"]:
        lines += ["TOP CORRELATIONS", "-" * 40] + [f"  {r['Feature A']:<20} ↔ {r['Feature B']:<20} {r['Correlation']:.4f}" for r in card["top_correlations"]] + [""]
    if card["problem_type"]:
        pt = card["problem_type"]
        lines += ["PROBLEM TYPE", "-" * 40, f"  {pt.get('type','N/A')}: {pt.get('reason','')}", ""]
    if card["model_results"]:
        lines += ["MODEL RESULTS", "-" * 40, f"  Best: {card['best_model']}", ""] + [f"  {r['name']:<35} {r['metric']}: {r['value']}{r['unit']}" for r in card["model_results"]]
    lines += ["", sep, "  End of Report", sep]
    return "\n".join(lines)
def generate_report_markdown(card):
    ov = card["overview"]
    lines = [
        "# 📊 Smart Data Preprocessing and Model Advisor",
        f"\n**Generated:** {card['generated_at']}  |  **Target:** `{card['target_col']}`\n\n---",
        "\n## 📁 Dataset Overview\n",
        "| Property | Value |", "|---|---|",
        f"| Rows | {ov['rows']:,} |", f"| Columns | {ov['cols']} |",
        f"| Missing | {ov['missing']:,} |", f"| Duplicates | {ov['duplicates']:,} |",
        f"| **Quality Score** | **{card['quality_score']} / 100** |",
    ]
    lines += ["\n## 🔍 Missing Values\n"]
    if card["missing_pct"]:
        lines += ["| Column | Missing % |", "|---|---|"] + [f"| {c} | {p:.1f}% |" for c, p in card["missing_pct"].items()]
    else:
        lines.append("✅ No missing values.")
    if card["top_correlations"]:
        lines += ["\n## 🔗 Top Correlations\n", "| Feature A | Feature B | Correlation |", "|---|---|---|"]
        lines += [f"| {r['Feature A']} | {r['Feature B']} | {r['Correlation']:.4f} |" for r in card["top_correlations"]]
    if card["problem_type"]:
        pt = card["problem_type"]
        lines += [f"\n## 🤖 Problem Type: {pt.get('icon','')} {pt.get('type','')}\n", pt.get("reason", "")]
    if card["model_results"]:
        lines += [f"\n## 🚀 Results  (Best: ⭐ {card['best_model']})\n", "| Model | Metric | Score |", "|---|---|---|"]
        lines += [f"| {r['name']}{'⭐' if r['name']==card['best_model'] else ''} | {r['metric']} | {r['value']}{r['unit']} |" for r in card["model_results"]]
    lines.append("\n---\n_Generated by Smart Data Preprocessing and Model Advisor_")
    return "\n".join(lines)
def generate_report_pdf(card):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
        S = getSampleStyleSheet()
        h1 = ParagraphStyle("h1", parent=S["Title"],   fontSize=18, textColor=colors.HexColor("#1a1a2e"))
        h2 = ParagraphStyle("h2", parent=S["Heading2"],fontSize=13, textColor=colors.HexColor("#667eea"))
        def tbl(data, widths=None):
            t = Table(data, colWidths=widths)
            t.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#667eea")),
                ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                ("FONTSIZE",(0,0),(-1,-1),9),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f0f4ff"),colors.white]),
                ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#c0c8e0")),
                ("PADDING",(0,0),(-1,-1),6),
            ]))
            return t
        ov = card["overview"]
        story = [
            Paragraph("Smart Data Preprocessing & Model Advisor", h1),
            Paragraph("Dataset Analysis Report", S["Heading3"]),
            Spacer(1, 0.3*cm),
            Paragraph(f"Generated: {card['generated_at']}  |  Target: {card['target_col']}", S["Normal"]),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor("#667eea")),
            Spacer(1, 0.3*cm),
            Paragraph("Dataset Overview", h2),
            tbl([["Property","Value"],["Rows",f"{ov['rows']:,}"],["Columns",str(ov['cols'])],
                 ["Missing",f"{ov['missing']:,}"],["Duplicates",f"{ov['duplicates']:,}"],
                 ["Quality Score",f"{card['quality_score']} / 100"]], [9*cm, 6*cm]),
            Spacer(1, 0.3*cm),
            Paragraph("Missing Values", h2),
        ]
        if card["missing_pct"]:
            story.append(tbl([["Column","Missing %"]]+[[c,f"{p:.1f}%"] for c,p in card["missing_pct"].items()],[11*cm,4*cm]))
        else:
            story.append(Paragraph("No missing values found.", S["Normal"]))
        story.append(Spacer(1, 0.3*cm))
        if card["top_correlations"]:
            story += [Paragraph("Top Correlations", h2),
                      tbl([["Feature A","Feature B","Correlation"]]+[[r["Feature A"],r["Feature B"],str(r["Correlation"])] for r in card["top_correlations"]],[6*cm,6*cm,3*cm]),
                      Spacer(1,0.3*cm)]
        if card["problem_type"]:
            pt = card["problem_type"]
            story += [Paragraph("Detected Problem Type", h2),
                      tbl([["Type",f"{pt.get('icon','')} {pt.get('type','')}"],["Reason",pt.get("reason","")]],[4*cm,11*cm]),
                      Spacer(1,0.3*cm)]
        if card["model_results"]:
            story += [Paragraph("Model Results", h2),
                      Paragraph(f"Best: {card['best_model']}", S["Normal"]),
                      Spacer(1,0.2*cm),
                      tbl([["Model","Metric","Score"]]+[[r["name"]+("⭐" if r["name"]==card["best_model"] else ""),r["metric"],f"{r['value']}{r['unit']}"] for r in card["model_results"]],[9*cm,3*cm,3*cm])]
        doc.build(story)
        buf.seek(0)
        return buf.getvalue()
    except ImportError:
        return generate_report_text(card).encode("utf-8")

#VISUALISATIONS — AUTO CHARTS
def plot_missing_heatmap(df):
    cols = [c for c in df.columns if df[c].isnull().any()]
    if not cols: return None
    fig = _fig((min(14, max(8, len(cols))), 4))
    ax  = fig.add_subplot(111); _ax(ax)
    sns.heatmap(df[cols].isnull(), cbar=False, ax=ax,
                cmap=sns.color_palette([BG_AX, CYAN], as_cmap=True), yticklabels=False)
    ax.set_title("Missing Values Heatmap  (cyan = missing)", fontsize=12)
    plt.tight_layout(); return fig
def plot_correlation_heatmap(df):
    num = df.select_dtypes(include=np.number)
    if num.shape[1] < 2: return None
    corr = num.corr(); s = max(6, min(14, corr.shape[0]))
    fig = _fig((s, s * 0.85)); ax = fig.add_subplot(111); _ax(ax)
    sns.heatmap(corr, annot=corr.shape[0]<=12, fmt=".2f",
                cmap=sns.diverging_palette(230, 10, as_cmap=True), center=0, ax=ax,
                linewidths=0.5, linecolor=BORDER, annot_kws={"size":8,"color":TEXT})
    ax.set_title("Correlation Heatmap", fontsize=12); plt.tight_layout(); return fig
def plot_distributions(df, max_cols=6):
    cols = list(df.select_dtypes(include=np.number).columns[:max_cols])
    return _grid_plot(cols, lambda ax, c: ax.hist(df[c].dropna(), bins=30, color=PURPLE, edgecolor=BG_AX, alpha=0.85) or ax.set_title(c, fontsize=10), "Feature Distributions")
def plot_boxplots(df, max_cols=6):
    cols = list(df.select_dtypes(include=np.number).columns[:max_cols])
    def draw(ax, c):
        ax.boxplot(df[c].dropna(), patch_artist=True,
                   boxprops=dict(facecolor="#192033",color=CYAN), medianprops=dict(color=RED,linewidth=2),
                   whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                   flierprops=dict(marker="o",color=YELLOW,markersize=4,alpha=0.7))
        ax.set_title(c, fontsize=10); ax.set_xticks([])
    return _grid_plot(cols, draw, "Boxplots — Outlier Overview", hspace=0.5)
def plot_violin(df, max_cols=6):
    cols = list(df.select_dtypes(include=np.number).columns[:max_cols])
    def draw(ax, c):
        p = ax.violinplot(df[c].dropna(), showmedians=True)
        for b in p["bodies"]: b.set_facecolor(PURPLE); b.set_alpha(0.7)
        p["cmedians"].set_color(CYAN)
        for k in ["cbars","cmins","cmaxes"]: p[k].set_color(MUTED)
        ax.set_title(c, fontsize=10); ax.set_xticks([])
    return _grid_plot(cols, draw, "Violin Plots", hspace=0.5)
def plot_kde(df, max_cols=6):
    cols = list(df.select_dtypes(include=np.number).columns[:max_cols])
    def draw(ax, c):
        d = df[c].dropna()
        ax.hist(d, bins=25, color=PURPLE, alpha=0.3, edgecolor=BG_AX, density=True)
        sns.kdeplot(d, ax=ax, color=CYAN, linewidth=2); ax.set_title(c, fontsize=10)
    return _grid_plot(cols, draw, "KDE Density Plots")
def plot_categorical_bar(df, max_cols=6):
    cols = list(df.select_dtypes("object").columns[:max_cols])
    def draw(ax, c):
        cnt = df[c].value_counts().head(10)
        ax.bar(range(len(cnt)), cnt.values, color=PALETTE[:len(cnt)], edgecolor=BG_AX, alpha=0.85)
        ax.set_xticks(range(len(cnt)))
        ax.set_xticklabels(cnt.index, rotation=35, ha="right", fontsize=8, color=MUTED)
        ax.set_title(c, fontsize=10)
    return _grid_plot(cols, draw, "Categorical Value Counts", hspace=0.6)
def plot_pie_chart(df, col):
    cnt = df[col].value_counts().head(8)
    if cnt.empty: return None
    fig = _fig((8, 6)); ax = fig.add_subplot(111); ax.set_facecolor(BG)
    _, texts, autotexts = ax.pie(cnt.values, labels=cnt.index, autopct="%1.1f%%",
                                  colors=PALETTE[:len(cnt)], startangle=140, pctdistance=0.82)
    for t in texts: t.set_color(TEXT); t.set_fontsize(9)
    for at in autotexts: at.set_color(BG); at.set_fontsize(8); at.set_fontweight("bold")
    ax.set_title(f"Distribution of '{col}'", color=TEXT, fontsize=12, pad=15)
    plt.tight_layout(); return fig
def plot_pairplot(df, max_cols=5, hue_col=None):
    num_cols = list(df.select_dtypes(include=np.number).columns[:max_cols])
    if len(num_cols) < 2: return None
    plot_df = df[num_cols].copy()
    if hue_col and hue_col in df.columns:
        plot_df[hue_col] = df[hue_col].astype(str)
    with plt.style.context("dark_background"):
        g = sns.pairplot(plot_df, hue=hue_col, diag_kind="kde", plot_kws={"alpha":0.5,"s":15})
        g.figure.patch.set_facecolor(BG)
        for ax in g.axes.flatten():
            if ax: ax.set_facecolor(BG_AX); ax.tick_params(colors=MUTED, labelsize=7)
        g.figure.suptitle("Pairplot", color=TEXT, y=1.01, fontsize=12)
    return g.figure
def plot_count_plot(df, col, hue_col=None):
    fig = _fig((10, 4)); ax = fig.add_subplot(111); _ax(ax)
    sns.countplot(data=df, x=col, order=df[col].value_counts().index[:15], hue=hue_col, ax=ax, palette=PALETTE)
    ax.set_title(f"Count Plot — '{col}'", fontsize=12)
    plt.xticks(rotation=35, ha="right", fontsize=8, color=MUTED); plt.tight_layout(); return fig

#CUSTOM CHART BUILDER
CHART_TYPES = {
    "Scatter Plot":        {"needs": ["x (numeric)", "y (numeric)"],          "optional": ["Hue (categorical)", "Size (numeric)"]},
    "Line Chart":          {"needs": ["x (any)", "y columns (multi-select)"], "optional": []},
    "Area Chart":          {"needs": ["x (any)", "y columns (multi-select)"], "optional": []},
    "Bar Chart":           {"needs": ["x (categorical)"],                     "optional": ["y (numeric)", "Aggregation"]},
    "Histogram":           {"needs": ["column (numeric)"],                    "optional": ["Bins", "Show KDE"]},
    "Box Plot":            {"needs": ["y (numeric)"],                         "optional": ["x / Group by (categorical)"]},
    "Violin Plot":         {"needs": ["column(s) (numeric, multi-select)"],   "optional": []},
    "KDE Density":         {"needs": ["column(s) (numeric, multi-select)"],   "optional": []},
    "Pie Chart":           {"needs": ["column (categorical)"],                "optional": []},
    "Count Plot":          {"needs": ["column (categorical)"],                "optional": ["Hue (categorical)"]},
    "Correlation Heatmap": {"needs": [],                                      "optional": []},
    "Pairplot":            {"needs": ["columns (numeric, multi-select)"],     "optional": ["Hue (categorical)"]},
    "Pivot Heatmap":       {"needs": ["x (categorical)", "y (categorical)"],  "optional": ["Value (numeric)"]},
}

def _scatter(df, x, y, hue=None, size_col=None):
    fig = _fig((10, 5)); ax = fig.add_subplot(111); _ax(ax)
    if hue and hue in df.columns:
        for i, cat in enumerate(df[hue].unique()):
            m = df[hue] == cat
            s = df.loc[m, size_col] * 0.5 if size_col and size_col in df else 25
            ax.scatter(df.loc[m, x], df.loc[m, y], color=PALETTE[i%len(PALETTE)], label=str(cat), s=s, alpha=0.7)
        ax.legend(facecolor=BG_AX, labelcolor=TEXT, fontsize=9)
    else:
        s = df[size_col]*0.5 if size_col and size_col in df else 25
        ax.scatter(df[x], df[y], color=CYAN, s=s, alpha=0.6)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"Scatter — {x} vs {y}", fontsize=12)
    plt.tight_layout(); return fig
def _line_area(df, x, y_cols, area=False):
    fig = _fig((12, 4)); ax = fig.add_subplot(111); _ax(ax)
    for i, c in enumerate(y_cols):
        xv = df[x] if not area else range(len(df))
        if area: ax.fill_between(xv, df[c], alpha=0.4, color=PALETTE[i%len(PALETTE)], label=c)
        ax.plot(xv, df[c], color=PALETTE[i%len(PALETTE)], linewidth=1.8, label=c if not area else None)
    ax.set_title(f"{'Area' if area else 'Line'} Chart — {', '.join(y_cols)}", fontsize=12)
    ax.legend(facecolor=BG_AX, labelcolor=TEXT, fontsize=9)
    plt.xticks(rotation=35, ha="right", fontsize=8, color=MUTED); plt.tight_layout(); return fig
def _bar(df, x, y=None, agg="count"):
    fig = _fig((10, 4)); ax = fig.add_subplot(111); _ax(ax)
    if y and y in df.columns and agg != "count":
        data = df.groupby(x)[y].agg(agg).sort_values(ascending=False).head(15)
        ax.set_ylabel(f"{agg}({y})"); ax.set_title(f"Bar — {agg}({y}) by {x}", fontsize=12)
    else:
        data = df[x].value_counts().head(15)
        ax.set_ylabel("Count"); ax.set_title(f"Bar — Count of {x}", fontsize=12)
    ax.bar(data.index.astype(str), data.values, color=PALETTE[:len(data)], edgecolor=BG_AX, alpha=0.85)
    plt.xticks(rotation=35, ha="right", fontsize=8, color=MUTED); plt.tight_layout(); return fig
def _histogram(df, col, bins=30, kde=True):
    fig = _fig((9, 4)); ax = fig.add_subplot(111); _ax(ax)
    d = df[col].dropna()
    ax.hist(d, bins=bins, color=PURPLE, edgecolor=BG_AX, alpha=0.75, density=kde)
    if kde: sns.kdeplot(d, ax=ax, color=CYAN, linewidth=2)
    ax.set_title(f"Histogram — '{col}'", fontsize=12); ax.set_xlabel(col)
    plt.tight_layout(); return fig
def _boxplot(df, y, x=None):
    fig = _fig((10, 5)); ax = fig.add_subplot(111); _ax(ax)
    if x and x in df.columns:
        sns.boxplot(data=df, x=x, y=y, order=df[x].value_counts().index[:12], ax=ax, palette=PALETTE, linewidth=1.2)
        plt.xticks(rotation=35, ha="right", fontsize=8, color=MUTED)
    else:
        ax.boxplot(df[y].dropna(), patch_artist=True,
                   boxprops=dict(facecolor="#192033",color=CYAN), medianprops=dict(color=RED,linewidth=2),
                   whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                   flierprops=dict(marker="o",color=YELLOW,markersize=4))
        ax.set_xticks([])
    ax.set_title(f"Boxplot — '{y}'" + (f" by '{x}'" if x else ""), fontsize=11)
    plt.tight_layout(); return fig
def _pivot_heatmap(df, x, y, val=None):
    fig = _fig((10, 6)); ax = fig.add_subplot(111); _ax(ax)
    try:
        pivot = df.pivot_table(index=y, columns=x, values=val, aggfunc="mean") if val else df.groupby([y,x]).size().unstack(fill_value=0)
        sns.heatmap(pivot, ax=ax, cmap=sns.color_palette("mako", as_cmap=True),
                    annot=pivot.shape[0]*pivot.shape[1]<=100, fmt=".1f",
                    linewidths=0.3, linecolor=BORDER, annot_kws={"size":7,"color":TEXT})
        ax.set_title(f"Heatmap — {y} × {x}" + (f" (mean {val})" if val else " (count)"), fontsize=11)
    except Exception as e:
        ax.text(0.5, 0.5, f"Cannot build heatmap:\n{e}", transform=ax.transAxes, ha="center", color=MUTED)
    plt.tight_layout(); return fig
def build_custom_chart(df, chart_type, params):
    try:
        ct = chart_type
        if ct == "Scatter Plot":       return _scatter(df, params["x"], params["y"], params.get("hue"), params.get("size"))
        elif ct == "Line Chart":       return _line_area(df, params["x"], params["y_cols"])
        elif ct == "Area Chart":       return _line_area(df, params["x"], params["y_cols"], area=True)
        elif ct == "Bar Chart":        return _bar(df, params["x"], params.get("y"), params.get("agg","count"))
        elif ct == "Histogram":        return _histogram(df, params["col"], int(params.get("bins",30)), params.get("kde",True))
        elif ct == "Box Plot":         return _boxplot(df, params["y"], params.get("x"))
        elif ct == "Violin Plot":      return plot_violin(df[params["cols"]] if "cols" in params else df)
        elif ct == "KDE Density":      return plot_kde(df[params["cols"]] if "cols" in params else df)
        elif ct == "Pie Chart":        return plot_pie_chart(df, params["col"])
        elif ct == "Count Plot":       return plot_count_plot(df, params["col"], params.get("hue"))
        elif ct == "Correlation Heatmap": return plot_correlation_heatmap(df)
        elif ct == "Pairplot":
            sub = df[params.get("cols", list(df.select_dtypes(include=np.number).columns[:5]))].copy()
            if params.get("hue") and params["hue"] in df: sub[params["hue"]] = df[params["hue"]].values
            return plot_pairplot(sub, hue_col=params.get("hue"))
        elif ct == "Pivot Heatmap":    return _pivot_heatmap(df, params["x"], params["y"], params.get("val"))
    except Exception as e:
        fig = _fig((8, 3)); ax = fig.add_subplot(111); _ax(ax)
        ax.text(0.5, 0.5, f"Cannot render chart:\n{e}", transform=ax.transAxes, ha="center", color=RED, fontsize=10)
        return fig

#MODEL RESULT PLOTS
def plot_feature_importances(imp):
    fig = _fig((10, max(3, len(imp)*0.4+1))); ax = fig.add_subplot(111); _ax(ax)
    imp.plot(kind="barh", ax=ax, color=PURPLE, edgecolor=BG_AX)
    ax.invert_yaxis(); ax.set_title("Feature Importances (Random Forest)", fontsize=11)
    plt.tight_layout(); return fig
def plot_model_comparison(comparison, problem_type):
    res = comparison["results"]
    if len(res) <= 1: return None
    vals = [r["value"] for r in res]
    colours = [GREEN if r["name"]==comparison["best"] else PURPLE for r in res]
    fig = _fig((9, 4)); ax = fig.add_subplot(111); _ax(ax)
    bars = ax.bar([r["short"] for r in res], vals, color=colours, edgecolor=BG_AX, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.01,
                f"{val}{res[0]['unit']}", ha="center", va="bottom", color=TEXT, fontsize=10, fontweight="bold")
    ax.set_title(f"Model Comparison — {res[0]['metric']}", fontsize=12); ax.set_ylabel(res[0]['metric'])
    ax.annotate(f"⭐ Best: {comparison['best']}", xy=(0.98,0.95), xycoords="axes fraction", ha="right", color=GREEN, fontsize=9)
    plt.tight_layout(); return fig
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred); s = cm.shape[0]
    fig = _fig((min(10, max(5,s)), min(8, max(4,s)))); ax = fig.add_subplot(111); _ax(ax)
    sns.heatmap(cm, annot=True, fmt="d", cmap=sns.color_palette("Blues",as_cmap=True), ax=ax,
                linewidths=0.5, linecolor=BORDER, annot_kws={"size":10,"color":"white"})
    ax.set_title("Confusion Matrix",fontsize=12); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout(); return fig