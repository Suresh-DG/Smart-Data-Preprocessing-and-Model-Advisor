import io
import warnings
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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (shared by all plot functions)
# ─────────────────────────────────────────────────────────────────
_BG     = "#111624"
_BG_AX  = "#0a0d14"
_BORDER = "#242d42"
_TEXT   = "#e4eaf6"
_MUTED  = "#6b7a99"
_CYAN   = "#00e5ff"
_PURPLE = "#7b61ff"
_RED    = "#ff6b6b"
_YELLOW = "#ffd166"
_GREEN  = "#00d97e"
_PINK   = "#f953c6"
_ORANGE = "#f7971e"

PALETTE = [_CYAN, _PURPLE, _GREEN, _YELLOW, _RED, _PINK, _ORANGE]


def _dark_fig(figsize=(10, 4)):
    return plt.figure(figsize=figsize, facecolor=_BG)


def _dark_ax(ax):
    ax.set_facecolor(_BG_AX)
    ax.tick_params(colors=_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER)
    ax.title.set_color(_TEXT)
    ax.xaxis.label.set_color(_MUTED)
    ax.yaxis.label.set_color(_MUTED)
    return ax


# ══════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING & VALIDATION
# ══════════════════════════════════════════════════════════════════

def load_dataset(uploaded_file) -> pd.DataFrame:
    """Load a CSV or Excel from a Streamlit UploadedFile object."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload .csv or .xlsx.")
    if df.empty:
        raise ValueError("The uploaded file is empty.")
    return df


def load_from_url(url: str) -> pd.DataFrame:
    """
    Load a dataset directly from a URL.

    Supports
    --------
    • Any direct CSV link (GitHub raw, data.gov, UCI, etc.)
    • Kaggle dataset raw links  (must be a direct download URL)
    • Google Sheets published as CSV
    • Any URL ending in .csv, .xlsx, .xls, or that returns CSV content

    How it works
    ------------
    1. Send an HTTP GET request with a browser-like User-Agent header
       (some servers block plain Python requests)
    2. Detect format from URL extension or Content-Type header
    3. Read into a pandas DataFrame

    GitHub tip: replace 'blob' with 'raw' in the URL, e.g.
        https://github.com/user/repo/raw/main/data.csv

    Kaggle tip: use the "Copy link" from the dataset files tab
    (direct download links, not the dataset page URL).

    Parameters
    ----------
    url : str — direct download URL to a CSV or Excel file

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError  — on HTTP error, unsupported format, or empty result
    """
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

    # Convert common non-raw GitHub URLs automatically
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    # Convert Google Sheets share URL to CSV export URL
    if "docs.google.com/spreadsheets" in url and "/edit" in url:
        url = url.replace("/edit", "/export?format=csv")
        if "#" in url:
            url = url.split("#")[0]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise ValueError("Request timed out. Check your internet connection or try a different URL.")
    except requests.exceptions.HTTPError as e:
        raise ValueError(f"HTTP error {response.status_code}: {e}. Make sure the URL is a direct download link.")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Could not reach the URL: {e}")

    content_type = response.headers.get("Content-Type", "")
    url_lower    = url.lower().split("?")[0]   # strip query params for extension check

    # Determine format
    if url_lower.endswith((".xlsx", ".xls")) or "spreadsheetml" in content_type:
        try:
            df = pd.read_excel(io.BytesIO(response.content))
        except Exception as e:
            raise ValueError(f"Could not parse as Excel: {e}")
    else:
        # Default: try CSV
        try:
            df = pd.read_csv(io.StringIO(response.text))
        except Exception:
            # Last resort: try reading raw bytes as CSV
            try:
                df = pd.read_csv(io.BytesIO(response.content))
            except Exception as e:
                raise ValueError(
                    f"Could not parse response as CSV or Excel: {e}. "
                    "Make sure the URL points directly to a data file."
                )

    if df.empty:
        raise ValueError("The file at that URL appears to be empty.")

    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "rows":        df.shape[0],
        "cols":        df.shape[1],
        "missing":     int(df.isnull().sum().sum()),
        "duplicates":  int(df.duplicated().sum()),
        "numeric":     int(df.select_dtypes(include=np.number).shape[1]),
        "categorical": int(df.select_dtypes(exclude=np.number).shape[1]),
    }


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — DATA QUALITY SCORING
# ══════════════════════════════════════════════════════════════════

def compute_quality_score(df: pd.DataFrame) -> int:
    total_cells   = df.shape[0] * df.shape[1] + 1e-9
    missing_ratio = df.isnull().sum().sum() / total_cells
    missing_score = max(0.0, 40.0 * (1 - missing_ratio * 5))
    dup_ratio     = df.duplicated().sum() / (len(df) + 1e-9)
    dup_score     = max(0.0, 30.0 * (1 - dup_ratio * 5))
    constant_cols = int(sum(df.nunique() <= 1))
    const_score   = max(0.0, 20.0 - constant_cols * 4)
    obj_cols      = df.select_dtypes("object").columns
    mixed_count   = sum(
        1 for c in obj_cols
        if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.3
    )
    mix_score = max(0.0, 10.0 - mixed_count * 3)
    return min(100, int(missing_score + dup_score + const_score + mix_score))


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — DATA CLEANING PIPELINE
# ══════════════════════════════════════════════════════════════════

def fix_dtypes(df):
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def handle_missing_values(df):
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype in [np.float64, np.int64, float, int]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col].fillna(mode_val[0], inplace=True)
    return df

def remove_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)

def handle_outliers_iqr(df):
    df = df.copy()
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    return df

def clean_data(df):
    df = fix_dtypes(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers_iqr(df)
    return df


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

def encode_and_scale(df: pd.DataFrame, target_col: str = None):
    df         = df.copy()
    feature_df = df.drop(columns=[target_col]) if target_col else df
    encoders   = {}
    obj_cols   = feature_df.select_dtypes("object").columns
    low_card   = [c for c in obj_cols if feature_df[c].nunique() <= 20]
    high_card  = [c for c in obj_cols if feature_df[c].nunique() >  20]
    for col in low_card:
        le = LabelEncoder()
        feature_df[col] = le.fit_transform(feature_df[col].astype(str))
        encoders[col]   = le
    if high_card:
        feature_df = pd.get_dummies(feature_df, columns=high_card, drop_first=True)
    feature_df = feature_df.select_dtypes(include=np.number)
    scaler     = StandardScaler()
    X_scaled   = scaler.fit_transform(feature_df)
    return X_scaled, list(feature_df.columns), encoders, scaler


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — PROBLEM TYPE DETECTION
# ══════════════════════════════════════════════════════════════════

def detect_problem_type(df: pd.DataFrame, target_col: str = None) -> dict:
    if not target_col or target_col not in df.columns:
        return {"type": "Clustering", "reason": "No target column selected. We'll find hidden patterns.", "icon": "🌐", "colour": "purple"}
    series   = df[target_col].dropna()
    n_unique = series.nunique()
    if series.dtype == "object" or n_unique <= 10:
        return {"type": "Classification", "reason": f"Target '{target_col}' has {n_unique} unique class(es) — predicting categories.", "icon": "🏷️", "colour": "blue"}
    return {"type": "Regression", "reason": f"Target '{target_col}' is continuous ({n_unique} unique values) — predicting a number.", "icon": "📈", "colour": "green"}


# ══════════════════════════════════════════════════════════════════
# SECTION 6 — MODEL CATALOG
# ══════════════════════════════════════════════════════════════════

MODEL_CATALOG = {
    "Classification": [
        {"name": "Random Forest Classifier",  "desc": "Ensemble of 100 decision trees. Robust, handles noise well.", "short": "Random Forest"},
        {"name": "Logistic Regression",       "desc": "Fast linear baseline. Interpretable coefficients.",           "short": "Logistic Reg."},
        {"name": "Decision Tree Classifier",  "desc": "Explainable, visual decision logic.",                         "short": "Decision Tree"},
    ],
    "Regression": [
        {"name": "Linear Regression",         "desc": "Classic fast baseline for continuous targets.",               "short": "Linear Reg."},
        {"name": "Decision Tree Regressor",   "desc": "Captures non-linear patterns without scaling.",               "short": "Decision Tree"},
        {"name": "Random Forest Regressor",   "desc": "Ensemble of trees — usually more accurate than one tree.",    "short": "Random Forest"},
    ],
    "Clustering": [
        {"name": "K-Means Clustering",        "desc": "Partitions data into K groups by minimising variance.",       "short": "K-Means"},
    ],
}


# ══════════════════════════════════════════════════════════════════
# SECTION 7 — MULTI-MODEL TRAINING & COMPARISON
# ══════════════════════════════════════════════════════════════════

def train_all_models(X, y, problem_type: str, test_size: float = 0.2) -> dict:
    results = []
    best_preds = X_test_out = y_test_out = None

    if problem_type == "Classification":
        stratify = y if len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
        X_test_out, y_test_out = X_test, y_test
        models = [
            ("Random Forest Classifier", "Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("Logistic Regression",      "Logistic Reg.", LogisticRegression(max_iter=1000, random_state=42)),
            ("Decision Tree Classifier", "Decision Tree", DecisionTreeClassifier(random_state=42)),
        ]
        best_acc, best_name = -1, ""
        for name, short, model in models:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc   = round(accuracy_score(y_test, preds) * 100, 2)
            results.append({"name": name, "short": short, "metric": "Accuracy", "value": acc, "unit": "%", "train_sz": len(X_train), "test_sz": len(X_test)})
            if acc > best_acc:
                best_acc, best_name, best_preds = acc, name, preds

    elif problem_type == "Regression":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_test_out, y_test_out = X_test, y_test
        models = [
            ("Linear Regression",         "Linear Reg.", LinearRegression()),
            ("Decision Tree Regressor",   "Decision Tree", DecisionTreeRegressor(random_state=42)),
            ("Random Forest Regressor",   "Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
        best_rmse, best_name = float("inf"), ""
        for name, short, model in models:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse  = round(np.sqrt(mean_squared_error(y_test, preds)), 4)
            r2    = round(r2_score(y_test, preds), 4)
            results.append({"name": name, "short": short, "metric": "RMSE", "value": rmse, "unit": "", "r2": r2, "train_sz": len(X_train), "test_sz": len(X_test)})
            if rmse < best_rmse:
                best_rmse, best_name, best_preds = rmse, name, preds

    elif problem_type == "Clustering":
        k     = min(4, len(X) // 10 + 2)
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        results.append({"name": f"K-Means (k={k})", "short": "K-Means", "metric": "Inertia", "value": round(float(model.inertia_), 2), "unit": "", "train_sz": len(X), "test_sz": 0, "labels": model.labels_})
        best_name = f"K-Means (k={k})"

    return {"results": results, "best": best_name, "X_test": X_test_out, "y_test": y_test_out, "best_preds": best_preds}


def get_feature_importances(X, y, feat_names, top_n=15) -> pd.Series:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=False).head(top_n)


# ══════════════════════════════════════════════════════════════════
# SECTION 8 — DATASET REPORT CARD
# ══════════════════════════════════════════════════════════════════

def dataset_report_card(df, target_col=None, problem=None, comparison=None) -> dict:
    summary     = dataset_summary(df)
    quality     = compute_quality_score(df)
    missing_pct = round(df.isnull().mean() * 100, 2)
    num_df      = df.select_dtypes(include=np.number)
    top_corr    = []
    if num_df.shape[1] >= 2:
        corr_matrix = num_df.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        corr_pairs = corr_matrix.stack().reset_index().rename(columns={"level_0": "Feature A", "level_1": "Feature B", 0: "Correlation"}).sort_values("Correlation", ascending=False)
        seen = set()
        for _, row in corr_pairs.iterrows():
            pair = tuple(sorted([row["Feature A"], row["Feature B"]]))
            if pair not in seen:
                seen.add(pair)
                top_corr.append({"Feature A": row["Feature A"], "Feature B": row["Feature B"], "Correlation": round(row["Correlation"], 4)})
            if len(top_corr) >= 10:
                break
    return {
        "generated_at":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overview":         summary,
        "quality_score":    quality,
        "missing_pct":      missing_pct[missing_pct > 0].to_dict(),
        "dtypes":           {c: str(df[c].dtype) for c in df.columns},
        "top_correlations": top_corr,
        "problem_type":     problem or {},
        "target_col":       target_col or "None",
        "model_results":    comparison["results"] if comparison else [],
        "best_model":       comparison["best"]    if comparison else "N/A",
    }


# ══════════════════════════════════════════════════════════════════
# SECTION 9 — REPORT EXPORT (TXT / MD / PDF)
# ══════════════════════════════════════════════════════════════════

def generate_report_text(card: dict) -> str:
    lines = []
    sep   = "=" * 60
    ov    = card["overview"]
    lines += [sep, "  SMART DATA PREPROCESSING AND MODEL ADVISOR", "  Dataset Analysis Report", sep,
              f"  Generated : {card['generated_at']}", f"  Target    : {card['target_col']}", sep, "",
              "DATASET OVERVIEW", "-" * 40,
              f"  Rows            : {ov['rows']:,}", f"  Columns         : {ov['cols']}",
              f"  Missing Cells   : {ov['missing']:,}", f"  Duplicate Rows  : {ov['duplicates']:,}",
              f"  Numeric Cols    : {ov['numeric']}", f"  Categorical Cols: {ov['categorical']}",
              f"  Quality Score   : {card['quality_score']} / 100", ""]
    if card["missing_pct"]:
        lines += ["MISSING VALUES", "-" * 40]
        for col, pct in card["missing_pct"].items():
            lines.append(f"  {col:<30} {pct:.1f}%")
        lines.append("")
    else:
        lines += ["MISSING VALUES", "-" * 40, "  No missing values found.", ""]
    if card["top_correlations"]:
        lines += ["TOP FEATURE CORRELATIONS", "-" * 40]
        for row in card["top_correlations"]:
            lines.append(f"  {row['Feature A']:<20}  ↔  {row['Feature B']:<20}  {row['Correlation']:.4f}")
        lines.append("")
    if card["problem_type"]:
        pt = card["problem_type"]
        lines += ["DETECTED PROBLEM TYPE", "-" * 40, f"  Type   : {pt.get('type','N/A')}", f"  Reason : {pt.get('reason','N/A')}", ""]
    if card["model_results"]:
        lines += ["MODEL TRAINING RESULTS", "-" * 40, f"  Best Model : {card['best_model']}", ""]
        for r in card["model_results"]:
            lines.append(f"  {r['name']:<35} {r['metric']}: {r['value']}{r['unit']}")
        lines.append("")
    lines += [sep, "  End of Report", sep]
    return "\n".join(lines)


def generate_report_markdown(card: dict) -> str:
    ov    = card["overview"]
    lines = [
        "# 📊 Smart Data Preprocessing and Model Advisor",
        "## Dataset Analysis Report", "",
        f"**Generated:** {card['generated_at']}  ", f"**Target Column:** `{card['target_col']}`", "",
        "---", "", "## 📁 Dataset Overview", "",
        "| Property | Value |", "|---|---|",
        f"| Rows | {ov['rows']:,} |", f"| Columns | {ov['cols']} |",
        f"| Missing Cells | {ov['missing']:,} |", f"| Duplicate Rows | {ov['duplicates']:,} |",
        f"| Numeric Columns | {ov['numeric']} |", f"| Categorical Columns | {ov['categorical']} |",
        f"| **Quality Score** | **{card['quality_score']} / 100** |", "",
    ]
    lines += ["## 🔍 Missing Values", ""]
    if card["missing_pct"]:
        lines += ["| Column | Missing % |", "|---|---|"]
        for col, pct in card["missing_pct"].items():
            lines.append(f"| {col} | {pct:.1f}% |")
    else:
        lines.append("✅ No missing values found.")
    lines.append("")
    if card["top_correlations"]:
        lines += ["## 🔗 Top Feature Correlations", "", "| Feature A | Feature B | Correlation |", "|---|---|---|"]
        for row in card["top_correlations"]:
            lines.append(f"| {row['Feature A']} | {row['Feature B']} | {row['Correlation']:.4f} |")
        lines.append("")
    if card["problem_type"]:
        pt = card["problem_type"]
        lines += ["## 🤖 Detected Problem Type", "", f"**Type:** {pt.get('icon','')} {pt.get('type','N/A')}  ", f"**Reason:** {pt.get('reason','N/A')}", ""]
    if card["model_results"]:
        lines += ["## 🚀 Model Training Results", "", f"**Best Model:** ⭐ {card['best_model']}", "", "| Model | Metric | Score |", "|---|---|---|"]
        for r in card["model_results"]:
            star = " ⭐" if r["name"] == card["best_model"] else ""
            lines.append(f"| {r['name']}{star} | {r['metric']} | {r['value']}{r['unit']} |")
        lines.append("")
    lines += ["---", "_Generated by Smart Data Preprocessing and Model Advisor_"]
    return "\n".join(lines)


def generate_report_pdf(card: dict) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        buf  = io.BytesIO()
        doc  = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
        stys = getSampleStyleSheet()
        story = []

        title_style = ParagraphStyle("T2", parent=stys["Title"],   fontSize=18, spaceAfter=4, textColor=colors.HexColor("#1a1a2e"))
        h2_style    = ParagraphStyle("H2", parent=stys["Heading2"], fontSize=13, textColor=colors.HexColor("#667eea"), spaceAfter=4)
        normal      = stys["Normal"]

        def make_table(data, col_widths=None):
            t = Table(data, colWidths=col_widths)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#667eea")),
                ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
                ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",   (0,0), (-1,-1), 9),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f0f4ff"), colors.white]),
                ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#c0c8e0")),
                ("PADDING",    (0,0), (-1,-1), 6),
            ]))
            return t

        ov = card["overview"]
        story += [
            Paragraph("📊 Smart Data Preprocessing & Model Advisor", title_style),
            Paragraph("Dataset Analysis Report", stys["Heading3"]),
            Spacer(1, 0.3*cm),
            Paragraph(f"Generated: {card['generated_at']}  |  Target: {card['target_col']}", normal),
            HRFlowable(width="100%", thickness=1, color=colors.HexColor("#667eea")),
            Spacer(1, 0.4*cm),
            Paragraph("Dataset Overview", h2_style),
            make_table([
                ["Property", "Value"],
                ["Rows", f"{ov['rows']:,}"], ["Columns", str(ov['cols'])],
                ["Missing Cells", f"{ov['missing']:,}"], ["Duplicate Rows", f"{ov['duplicates']:,}"],
                ["Numeric Columns", str(ov['numeric'])], ["Categorical Cols", str(ov['categorical'])],
                ["Quality Score", f"{card['quality_score']} / 100"],
            ], col_widths=[9*cm, 6*cm]),
            Spacer(1, 0.4*cm),
            Paragraph("Missing Values", h2_style),
        ]
        if card["missing_pct"]:
            story.append(make_table([["Column", "Missing %"]] + [[col, f"{pct:.1f}%"] for col, pct in card["missing_pct"].items()], col_widths=[11*cm, 4*cm]))
        else:
            story.append(Paragraph("No missing values found.", normal))
        story.append(Spacer(1, 0.4*cm))
        if card["top_correlations"]:
            story += [Paragraph("Top Feature Correlations", h2_style),
                      make_table([["Feature A","Feature B","Correlation"]] + [[r["Feature A"],r["Feature B"],str(r["Correlation"])] for r in card["top_correlations"]], col_widths=[6*cm,6*cm,3*cm]),
                      Spacer(1, 0.4*cm)]
        if card["problem_type"]:
            pt = card["problem_type"]
            story += [Paragraph("Detected Problem Type", h2_style),
                      make_table([["Type", f"{pt.get('icon','')} {pt.get('type','N/A')}"], ["Reason", pt.get("reason","N/A")]], col_widths=[4*cm,11*cm]),
                      Spacer(1, 0.4*cm)]
        if card["model_results"]:
            story += [Paragraph("Model Training Results", h2_style),
                      Paragraph(f"Best Model: ⭐ {card['best_model']}", normal),
                      Spacer(1, 0.2*cm),
                      make_table([["Model","Metric","Score"]] + [[r["name"]+(" ⭐" if r["name"]==card["best_model"] else ""), r["metric"], f"{r['value']}{r['unit']}"] for r in card["model_results"]], col_widths=[9*cm,3*cm,3*cm])]

        doc.build(story)
        buf.seek(0)
        return buf.getvalue()
    except ImportError:
        return generate_report_text(card).encode("utf-8")


# ══════════════════════════════════════════════════════════════════
# SECTION 10 — STANDARD VISUALISATIONS (Auto)
# ══════════════════════════════════════════════════════════════════

def plot_missing_heatmap(df: pd.DataFrame):
    cols = [c for c in df.columns if df[c].isnull().any()]
    if not cols:
        return None
    fig = _dark_fig((min(14, max(8, len(cols))), 4))
    ax  = fig.add_subplot(111)
    _dark_ax(ax)
    cmap = sns.color_palette([_BG_AX, _CYAN], as_cmap=True)
    sns.heatmap(df[cols].isnull(), cbar=False, ax=ax, cmap=cmap, yticklabels=False)
    ax.set_title("Missing Values Heatmap  (cyan = missing)", fontsize=12)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        return None
    corr  = num_df.corr()
    size  = max(6, min(14, corr.shape[0]))
    fig   = _dark_fig((size, size * 0.85))
    ax    = fig.add_subplot(111)
    _dark_ax(ax)
    cmap  = sns.diverging_palette(230, 10, as_cmap=True)
    sns.heatmap(corr, annot=corr.shape[0] <= 12, fmt=".2f", cmap=cmap, center=0, ax=ax,
                linewidths=0.5, linecolor=_BORDER, annot_kws={"size": 8, "color": _TEXT})
    ax.set_title("Correlation Heatmap", fontsize=12)
    plt.tight_layout()
    return fig


def plot_distributions(df: pd.DataFrame, max_cols: int = 6):
    num_cols = list(df.select_dtypes(include=np.number).columns[:max_cols])
    if not num_cols:
        return None
    ncols = min(3, len(num_cols))
    nrows = (len(num_cols) + ncols - 1) // ncols
    fig   = _dark_fig((14, 3.5 * nrows))
    gs    = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.45, wspace=0.35)
    for i, col in enumerate(num_cols):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        _dark_ax(ax)
        ax.hist(df[col].dropna(), bins=30, color=_PURPLE, edgecolor=_BG_AX, alpha=0.85)
        ax.set_title(col, fontsize=10)
    fig.suptitle("Feature Distributions (Histograms)", color=_TEXT, fontsize=13, y=1.01)
    return fig


def plot_boxplots(df: pd.DataFrame, max_cols: int = 6):
    num_cols = list(df.select_dtypes(include=np.number).columns[:max_cols])
    if not num_cols:
        return None
    ncols = min(3, len(num_cols))
    nrows = (len(num_cols) + ncols - 1) // ncols
    fig   = _dark_fig((14, 3.5 * nrows))
    gs    = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.5, wspace=0.35)
    for i, col in enumerate(num_cols):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        _dark_ax(ax)
        ax.boxplot(df[col].dropna(), patch_artist=True,
                   boxprops=dict(facecolor="#192033", color=_CYAN),
                   medianprops=dict(color=_RED, linewidth=2),
                   whiskerprops=dict(color=_MUTED), capprops=dict(color=_MUTED),
                   flierprops=dict(marker="o", color=_YELLOW, markersize=4, alpha=0.7))
        ax.set_title(col, fontsize=10)
        ax.set_xticks([])
    fig.suptitle("Boxplots — Outlier Overview", color=_TEXT, fontsize=13, y=1.01)
    return fig


def plot_violin(df: pd.DataFrame, max_cols: int = 6):
    """Violin plots — show full distribution shape + quartiles."""
    num_cols = list(df.select_dtypes(include=np.number).columns[:max_cols])
    if not num_cols:
        return None
    ncols = min(3, len(num_cols))
    nrows = (len(num_cols) + ncols - 1) // ncols
    fig   = _dark_fig((14, 3.5 * nrows))
    gs    = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.5, wspace=0.35)
    for i, col in enumerate(num_cols):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        _dark_ax(ax)
        data = df[col].dropna()
        parts = ax.violinplot(data, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(_PURPLE)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color(_CYAN)
        for key in ["cbars", "cmins", "cmaxes"]:
            parts[key].set_color(_MUTED)
        ax.set_title(col, fontsize=10)
        ax.set_xticks([])
    fig.suptitle("Violin Plots — Distribution Shape", color=_TEXT, fontsize=13, y=1.01)
    return fig


def plot_kde(df: pd.DataFrame, max_cols: int = 6):
    """KDE (Kernel Density Estimate) — smooth distribution curves."""
    num_cols = list(df.select_dtypes(include=np.number).columns[:max_cols])
    if not num_cols:
        return None
    ncols = min(3, len(num_cols))
    nrows = (len(num_cols) + ncols - 1) // ncols
    fig   = _dark_fig((14, 3.5 * nrows))
    gs    = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.45, wspace=0.35)
    for i, col in enumerate(num_cols):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        _dark_ax(ax)
        data = df[col].dropna()
        ax.hist(data, bins=25, color=_PURPLE, alpha=0.3, edgecolor=_BG_AX, density=True)
        sns.kdeplot(data, ax=ax, color=_CYAN, linewidth=2)
        ax.set_title(col, fontsize=10)
    fig.suptitle("KDE Density Plots", color=_TEXT, fontsize=13, y=1.01)
    return fig


def plot_categorical_bar(df: pd.DataFrame, max_cols: int = 6):
    """Bar charts for top categories in each categorical column."""
    cat_cols = list(df.select_dtypes(include="object").columns[:max_cols])
    if not cat_cols:
        return None
    ncols = min(3, len(cat_cols))
    nrows = (len(cat_cols) + ncols - 1) // ncols
    fig   = _dark_fig((14, 3.5 * nrows))
    gs    = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.6, wspace=0.35)
    for i, col in enumerate(cat_cols):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        _dark_ax(ax)
        counts = df[col].value_counts().head(10)
        colors_list = PALETTE[:len(counts)]
        ax.bar(range(len(counts)), counts.values, color=colors_list, edgecolor=_BG_AX, alpha=0.85)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=35, ha="right", fontsize=8, color=_MUTED)
        ax.set_title(col, fontsize=10)
    fig.suptitle("Categorical Value Counts", color=_TEXT, fontsize=13, y=1.01)
    return fig


def plot_pie_chart(df: pd.DataFrame, col: str):
    """Pie chart for a categorical column (top 8 categories)."""
    counts = df[col].value_counts().head(8)
    if counts.empty:
        return None
    fig = _dark_fig((8, 6))
    ax  = fig.add_subplot(111)
    ax.set_facecolor(_BG)
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=PALETTE[:len(counts)],
        startangle=140,
        pctdistance=0.82,
    )
    for t in texts:
        t.set_color(_TEXT)
        t.set_fontsize(9)
    for at in autotexts:
        at.set_color(_BG)
        at.set_fontsize(8)
        at.set_fontweight("bold")
    ax.set_title(f"Distribution of '{col}'", color=_TEXT, fontsize=12, pad=15)
    plt.tight_layout()
    return fig


def plot_pairplot(df: pd.DataFrame, max_cols: int = 5, hue_col: str = None):
    """
    Pairwise scatter matrix for all numeric columns.
    Diagonal shows KDE. Off-diagonal shows scatter.
    """
    num_cols = list(df.select_dtypes(include=np.number).columns[:max_cols])
    if len(num_cols) < 2:
        return None
    plot_df = df[num_cols].copy()
    if hue_col and hue_col in df.columns:
        plot_df[hue_col] = df[hue_col].astype(str)

    with plt.style.context("dark_background"):
        pair_grid = sns.pairplot(
            plot_df,
            hue=hue_col,
            diag_kind="kde",
            plot_kws={"alpha": 0.5, "s": 15},
            palette=PALETTE[:plot_df[hue_col].nunique()] if hue_col and hue_col in plot_df else None,
        )
        pair_grid.figure.patch.set_facecolor(_BG)
        for ax in pair_grid.axes.flatten():
            if ax:
                ax.set_facecolor(_BG_AX)
                ax.tick_params(colors=_MUTED, labelsize=7)
        pair_grid.figure.suptitle("Pairplot — Feature Relationships", color=_TEXT, y=1.01, fontsize=12)
    return pair_grid.figure


def plot_count_plot(df: pd.DataFrame, col: str, hue_col: str = None):
    """Count/frequency bar chart for a single categorical column."""
    fig = _dark_fig((10, 4))
    ax  = fig.add_subplot(111)
    _dark_ax(ax)
    order = df[col].value_counts().index[:15]
    sns.countplot(
        data=df, x=col, order=order, hue=hue_col,
        ax=ax, palette=PALETTE,
    )
    ax.set_title(f"Count Plot — '{col}'", fontsize=12)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    plt.xticks(rotation=35, ha="right", fontsize=8, color=_MUTED)
    plt.tight_layout()
    return fig


def plot_line_chart(df: pd.DataFrame, x_col: str, y_cols: list):
    """Line chart — useful for time series or ordered data."""
    fig = _dark_fig((12, 4))
    ax  = fig.add_subplot(111)
    _dark_ax(ax)
    for i, col in enumerate(y_cols):
        ax.plot(df[x_col], df[col], color=PALETTE[i % len(PALETTE)], linewidth=1.8, label=col, alpha=0.9)
    ax.set_title(f"Line Chart — {', '.join(y_cols)} vs {x_col}", fontsize=12)
    ax.set_xlabel(x_col)
    ax.legend(facecolor=_BG2 if "_BG2" in dir() else _BG_AX, labelcolor=_TEXT, fontsize=9)
    plt.xticks(rotation=35, ha="right", fontsize=8, color=_MUTED)
    plt.tight_layout()
    return fig


def plot_area_chart(df: pd.DataFrame, x_col: str, y_cols: list):
    """Stacked area chart for multiple numeric columns."""
    fig = _dark_fig((12, 4))
    ax  = fig.add_subplot(111)
    _dark_ax(ax)
    for i, col in enumerate(y_cols):
        ax.fill_between(range(len(df)), df[col], alpha=0.4, color=PALETTE[i % len(PALETTE)], label=col)
        ax.plot(range(len(df)), df[col], color=PALETTE[i % len(PALETTE)], linewidth=1.2)
    ax.set_title(f"Area Chart — {', '.join(y_cols)}", fontsize=12)
    ax.legend(facecolor=_BG_AX, labelcolor=_TEXT, fontsize=9)
    plt.tight_layout()
    return fig


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str = None, size_col: str = None):
    """Scatter plot with optional colour encoding and size encoding."""
    fig = _dark_fig((10, 5))
    ax  = fig.add_subplot(111)
    _dark_ax(ax)

    if hue_col and hue_col in df.columns:
        categories = df[hue_col].unique()
        for i, cat in enumerate(categories):
            mask = df[hue_col] == cat
            sizes = df.loc[mask, size_col] * 0.5 if size_col and size_col in df.columns else 25
            ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                       color=PALETTE[i % len(PALETTE)], label=str(cat),
                       s=sizes, alpha=0.7, edgecolors=_BG_AX, linewidths=0.3)
        ax.legend(facecolor=_BG_AX, labelcolor=_TEXT, fontsize=9, title=hue_col,
                  title_fontsize=8, labelspacing=0.3)
    else:
        sizes = df[size_col] * 0.5 if size_col and size_col in df.columns else 25
        ax.scatter(df[x_col], df[y_col], color=_CYAN, s=sizes, alpha=0.6,
                   edgecolors=_BG_AX, linewidths=0.3)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Scatter — {x_col} vs {y_col}", fontsize=12)
    plt.tight_layout()
    return fig


def plot_histogram_custom(df: pd.DataFrame, col: str, bins: int = 30, kde: bool = True):
    """Histogram with optional KDE overlay for a single column."""
    fig = _dark_fig((9, 4))
    ax  = fig.add_subplot(111)
    _dark_ax(ax)
    data = df[col].dropna()
    ax.hist(data, bins=bins, color=_PURPLE, edgecolor=_BG_AX, alpha=0.75, density=kde)
    if kde:
        sns.kdeplot(data, ax=ax, color=_CYAN, linewidth=2)
    ax.set_title(f"Histogram — '{col}'", fontsize=12)
    ax.set_xlabel(col)
    plt.tight_layout()
    return fig


def plot_box_custom(df: pd.DataFrame, y_col: str, x_col: str = None):
    """Boxplot — single column, or grouped by a categorical column."""
    fig = _dark_fig((10, 5))
    ax  = fig.add_subplot(111)
    _dark_ax(ax)
    if x_col and x_col in df.columns:
        order = df[x_col].value_counts().index[:12]
        sns.boxplot(data=df, x=x_col, y=y_col, order=order, ax=ax,
                    palette=PALETTE, linewidth=1.2)
        plt.xticks(rotation=35, ha="right", fontsize=8, color=_MUTED)
    else:
        ax.boxplot(df[y_col].dropna(), patch_artist=True, vert=True,
                   boxprops=dict(facecolor="#192033", color=_CYAN),
                   medianprops=dict(color=_RED, linewidth=2),
                   whiskerprops=dict(color=_MUTED), capprops=dict(color=_MUTED),
                   flierprops=dict(marker="o", color=_YELLOW, markersize=4))
        ax.set_xticks([])
    ax.set_title(f"Boxplot — '{y_col}'" + (f" grouped by '{x_col}'" if x_col else ""), fontsize=11)
    plt.tight_layout()
    return fig


def plot_bar_custom(df: pd.DataFrame, x_col: str, y_col: str = None, agg: str = "count"):
    """
    Bar chart — count or aggregated values.
    agg options: 'count', 'mean', 'sum', 'median'
    """
    fig = _dark_fig((10, 4))
    ax  = fig.add_subplot(111)
    _dark_ax(ax)
    if y_col and y_col in df.columns and agg != "count":
        agg_func = {"mean": "mean", "sum": "sum", "median": "median"}.get(agg, "mean")
        grouped  = df.groupby(x_col)[y_col].agg(agg_func).sort_values(ascending=False).head(15)
        ax.bar(grouped.index.astype(str), grouped.values,
               color=PALETTE[:len(grouped)], edgecolor=_BG_AX, alpha=0.85)
        ax.set_ylabel(f"{agg}({y_col})")
        ax.set_title(f"Bar Chart — {agg}({y_col}) by {x_col}", fontsize=12)
    else:
        counts = df[x_col].value_counts().head(15)
        ax.bar(counts.index.astype(str), counts.values,
               color=PALETTE[:len(counts)], edgecolor=_BG_AX, alpha=0.85)
        ax.set_ylabel("Count")
        ax.set_title(f"Bar Chart — Count of {x_col}", fontsize=12)
    plt.xticks(rotation=35, ha="right", fontsize=8, color=_MUTED)
    plt.tight_layout()
    return fig


def plot_heatmap_custom(df: pd.DataFrame, x_col: str, y_col: str, val_col: str = None):
    """Pivot heatmap for two categorical columns, coloured by a numeric value or count."""
    fig = _dark_fig((10, 6))
    ax  = fig.add_subplot(111)
    _dark_ax(ax)
    try:
        if val_col and val_col in df.columns:
            pivot = df.pivot_table(index=y_col, columns=x_col, values=val_col, aggfunc="mean")
        else:
            pivot = df.groupby([y_col, x_col]).size().unstack(fill_value=0)
        cmap = sns.color_palette("mako", as_cmap=True)
        sns.heatmap(pivot, ax=ax, cmap=cmap, annot=pivot.shape[0] * pivot.shape[1] <= 100,
                    fmt=".1f", linewidths=0.3, linecolor=_BORDER,
                    annot_kws={"size": 7, "color": _TEXT})
        title = f"Heatmap — {y_col} × {x_col}" + (f" (mean of {val_col})" if val_col else " (count)")
        ax.set_title(title, fontsize=11)
    except Exception as e:
        ax.text(0.5, 0.5, f"Could not build heatmap:\n{e}", transform=ax.transAxes,
                ha="center", va="center", color=_MUTED, fontsize=10)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# SECTION 11 — CUSTOM CHART BUILDER
# ══════════════════════════════════════════════════════════════════

# All supported chart types for the builder UI
CHART_TYPES = {
    "Scatter Plot":         {"needs": ["x (numeric)", "y (numeric)"],          "optional": ["Hue (categorical)", "Size (numeric)"]},
    "Line Chart":           {"needs": ["x (any)", "y columns (multi-select)"], "optional": []},
    "Area Chart":           {"needs": ["x (any)", "y columns (multi-select)"], "optional": []},
    "Bar Chart":            {"needs": ["x (categorical)"],                     "optional": ["y (numeric)", "Aggregation"]},
    "Histogram":            {"needs": ["column (numeric)"],                    "optional": ["Bins", "Show KDE"]},
    "Box Plot":             {"needs": ["y (numeric)"],                         "optional": ["x / Group by (categorical)"]},
    "Violin Plot":          {"needs": ["column(s) (numeric, multi-select)"],   "optional": []},
    "KDE Density":          {"needs": ["column(s) (numeric, multi-select)"],   "optional": []},
    "Pie Chart":            {"needs": ["column (categorical)"],                "optional": []},
    "Count Plot":           {"needs": ["column (categorical)"],                "optional": ["Hue (categorical)"]},
    "Correlation Heatmap":  {"needs": [],                                      "optional": []},
    "Pairplot":             {"needs": ["columns (numeric, multi-select)"],     "optional": ["Hue (categorical)"]},
    "Pivot Heatmap":        {"needs": ["x (categorical)", "y (categorical)"],  "optional": ["Value (numeric)"]},
}


def build_custom_chart(df: pd.DataFrame, chart_type: str, params: dict):
    """
    Unified dispatcher for the custom chart builder.

    Parameters
    ----------
    df         : DataFrame to plot from
    chart_type : key from CHART_TYPES
    params     : dict of column selections from the UI
                 Keys vary by chart type (see CHART_TYPES above)

    Returns
    -------
    matplotlib Figure or None
    """
    try:
        ct = chart_type

        if ct == "Scatter Plot":
            return plot_scatter(df, params["x"], params["y"],
                                hue_col=params.get("hue"),
                                size_col=params.get("size"))

        elif ct == "Line Chart":
            return plot_line_chart(df, params["x"], params["y_cols"])

        elif ct == "Area Chart":
            return plot_area_chart(df, params["x"], params["y_cols"])

        elif ct == "Bar Chart":
            return plot_bar_custom(df, params["x"],
                                   y_col=params.get("y"),
                                   agg=params.get("agg", "count"))

        elif ct == "Histogram":
            return plot_histogram_custom(df, params["col"],
                                         bins=int(params.get("bins", 30)),
                                         kde=params.get("kde", True))

        elif ct == "Box Plot":
            return plot_box_custom(df, params["y"], x_col=params.get("x"))

        elif ct == "Violin Plot":
            subset = df[params["cols"]].copy() if "cols" in params else df
            return plot_violin(subset)

        elif ct == "KDE Density":
            subset = df[params["cols"]].copy() if "cols" in params else df
            return plot_kde(subset)

        elif ct == "Pie Chart":
            return plot_pie_chart(df, params["col"])

        elif ct == "Count Plot":
            return plot_count_plot(df, params["col"], hue_col=params.get("hue"))

        elif ct == "Correlation Heatmap":
            return plot_correlation_heatmap(df)

        elif ct == "Pairplot":
            subset_cols = params.get("cols", list(df.select_dtypes(include=np.number).columns[:5]))
            plot_df     = df[subset_cols].copy()
            if params.get("hue") and params["hue"] in df.columns:
                plot_df[params["hue"]] = df[params["hue"]].values
            return plot_pairplot(plot_df, hue_col=params.get("hue"))

        elif ct == "Pivot Heatmap":
            return plot_heatmap_custom(df, params["x"], params["y"], val_col=params.get("val"))

    except Exception as e:
        fig = _dark_fig((8, 3))
        ax  = fig.add_subplot(111)
        _dark_ax(ax)
        ax.text(0.5, 0.5, f"Could not render chart:\n{e}",
                transform=ax.transAxes, ha="center", va="center",
                color=_RED, fontsize=10)
        return fig


# ══════════════════════════════════════════════════════════════════
# SECTION 12 — MODEL PLOTS (unchanged)
# ══════════════════════════════════════════════════════════════════

def plot_feature_importances(importances: pd.Series):
    fig = _dark_fig((10, max(3, len(importances) * 0.4 + 1)))
    ax  = fig.add_subplot(111)
    _dark_ax(ax)
    importances.plot(kind="barh", ax=ax, color=_PURPLE, edgecolor=_BG_AX)
    ax.invert_yaxis()
    ax.set_title("Top Feature Importances (Random Forest)", fontsize=11)
    plt.tight_layout()
    return fig


def plot_model_comparison(comparison: dict, problem_type: str):
    results = comparison["results"]
    if len(results) <= 1:
        return None
    names   = [r["short"] for r in results]
    values  = [r["value"] for r in results]
    best    = comparison["best"]
    colours = [_GREEN if r["name"] == best else _PURPLE for r in results]
    fig     = _dark_fig((9, 4))
    ax      = fig.add_subplot(111)
    _dark_ax(ax)
    bars = ax.bar(names, values, color=colours, edgecolor=_BG_AX, width=0.5)
    for bar, val in zip(bars, values):
        unit = results[0]["unit"]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val}{unit}", ha="center", va="bottom", color=_TEXT, fontsize=10, fontweight="bold")
    metric = results[0]["metric"]
    ax.set_title(f"Model Comparison — {metric}", fontsize=12)
    ax.set_ylabel(metric)
    ax.annotate(f"⭐ Best: {best}", xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", color=_GREEN, fontsize=9)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_test, y_pred, labels=None):
    cm  = confusion_matrix(y_test, y_pred)
    fig = _dark_fig((min(10, max(5, cm.shape[0])), min(8, max(4, cm.shape[0]))))
    ax  = fig.add_subplot(111)
    _dark_ax(ax)
    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                linewidths=0.5, linecolor=_BORDER,
                annot_kws={"size": 10, "color": "white"},
                xticklabels=labels or "auto", yticklabels=labels or "auto")
    ax.set_title("Confusion Matrix", fontsize=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    return fig
