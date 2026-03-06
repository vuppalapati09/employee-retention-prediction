# app.py – Streamlit app aligned with EMP1 notebook logic

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Optional models
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

# Paths
DEFAULT_TRAIN_PATH = "/mnt/data/aug_train.csv"
MODEL_SAVE_PATH = "./best_employee_retention_model.joblib"

# Streamlit basic config
st.set_page_config(
    page_title="Employee Retention & Job Change Prediction",
    layout="wide"
)
st.title("Employee Retention Prediction – Job Change Likelihood")

# -------------------------------------------------------------------
# 1. Feature Engineering Functions (same logic as notebook)
# -------------------------------------------------------------------
def clean_experience(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if value == ">20":
        return 21
    if value == "<1":
        return 0
    try:
        return float(value)
    except ValueError:
        return np.nan


def clean_company_size(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()

    # Ranges like "50-99", "100-500"
    if "-" in text:
        parts = text.split("-")
        try:
            low = int(parts[0])
            high = int(parts[1])
            return (low + high) / 2
        except Exception:
            return np.nan

    # Special case "10000+"
    if "10000+" in text:
        return 10000

    # Optional: "<10" → ~5
    if "<10" in text:
        return 5

    return np.nan


def clean_last_new_job(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text == ">4":
        return 5
    if text.lower() == "never":
        return 0
    try:
        return float(text)
    except ValueError:
        return np.nan


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering as in the notebook:
    - experience_years
    - company_size_num
    - last_new_job_num
    - training_per_year
    - experience_bucket
    - city_dev_bucket
    """
    df_fe = df.copy()

    if "experience" in df_fe.columns:
        df_fe["experience_years"] = df_fe["experience"].apply(clean_experience)
    else:
        df_fe["experience_years"] = np.nan

    if "company_size" in df_fe.columns:
        df_fe["company_size_num"] = df_fe["company_size"].apply(clean_company_size)
    else:
        df_fe["company_size_num"] = np.nan

    if "last_new_job" in df_fe.columns:
        df_fe["last_new_job_num"] = df_fe["last_new_job"].apply(clean_last_new_job)
    else:
        df_fe["last_new_job_num"] = np.nan

    if "training_hours" in df_fe.columns:
        df_fe["training_per_year"] = df_fe["training_hours"] / (
            1 + df_fe["experience_years"].fillna(0)
        )
    else:
        df_fe["training_per_year"] = np.nan

    if "experience_years" in df_fe.columns:
        df_fe["experience_bucket"] = pd.cut(
            df_fe["experience_years"],
            bins=[-1, 1, 5, 10, 20, np.inf],
            labels=["0-1", "2-5", "6-10", "11-20", "20+"]
        )

    if "city_development_index" in df_fe.columns:
        df_fe["city_dev_bucket"] = pd.cut(
            df_fe["city_development_index"],
            bins=[0, 0.6, 0.8, 1.0],
            labels=["Low", "Medium", "High"]
        )

    return df_fe

# -------------------------------------------------------------------
# 2. Load CSV helper
# -------------------------------------------------------------------
def load_csv(uploaded, default_path):
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.sidebar.success("Uploaded train CSV loaded")
            return df
        except Exception as e:
            st.sidebar.error(f"Failed to read uploaded CSV: {e}")
            return pd.DataFrame()
    else:
        if os.path.exists(default_path):
            try:
                df = pd.read_csv(default_path)
                st.sidebar.info(f"Loaded default train CSV from {default_path}")
                return df
            except Exception as e:
                st.sidebar.error(f"Failed to read default CSV: {e}")
                return pd.DataFrame()
        else:
            st.sidebar.warning(f"No file at {default_path}")
            return pd.DataFrame()

# -------------------------------------------------------------------
# 3. Sidebar – file upload
# -------------------------------------------------------------------
st.sidebar.header("Data & Model Options")
uploaded_train = st.sidebar.file_uploader("Upload train CSV (with 'target')", type=["csv"])
uploaded_model = st.sidebar.file_uploader("Upload trained pipeline (.joblib)", type=["joblib", "pkl"])

train_df = load_csv(uploaded_train, DEFAULT_TRAIN_PATH)

loaded_pipeline = None
if uploaded_model is not None:
    try:
        uploaded_model.seek(0)
        loaded_pipeline = joblib.load(uploaded_model)
        st.sidebar.success("Uploaded pipeline loaded.")
    except Exception as e:
        st.sidebar.error(f"Failed to load pipeline: {e}")

# -------------------------------------------------------------------
# 4. Dataset preview & basic EDA
# -------------------------------------------------------------------
if train_df.empty:
    st.error("No train data available. Upload a CSV or place 'aug_train.csv' at /mnt/data.")
    st.stop()

st.subheader("Dataset Preview & Basic EDA")
st.write("Train shape:", train_df.shape)
st.dataframe(train_df.head(10))

with st.expander("Data types, missing values & unique counts"):
    info_df = pd.DataFrame({
        "dtype": train_df.dtypes.astype(str),
        "missing": train_df.isna().sum(),
        "unique": train_df.nunique(dropna=False)
    })
    st.dataframe(info_df)

if "target" in train_df.columns:
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(x="target", data=train_df, ax=ax)
    ax.set_title("Target distribution (0 = stay, 1 = change)")
    st.pyplot(fig)

    st.write("Target distribution (proportions):")
    st.write(train_df["target"].value_counts(normalize=True))
else:
    st.warning("No 'target' column found in train CSV. Training cannot proceed without target.")
    st.stop()

# -------------------------------------------------------------------
# 5. Feature engineering & train/validation split
# -------------------------------------------------------------------
st.subheader("Feature Engineering & Train/Validation Split")

train_fe = apply_feature_engineering(train_df)

# Drop rows with missing target
train_fe = train_fe[~train_fe["target"].isna()].copy()
train_fe["target"] = train_fe["target"].astype(int)

# Drop ID-like columns if present
drop_cols = []
for col in ["enrollee_id", "id"]:
    if col in train_fe.columns:
        drop_cols.append(col)

X = train_fe.drop(columns=["target"] + drop_cols)
y = train_fe["target"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

st.write("Train/validation shapes:", X_train.shape, X_val.shape)

# -------------------------------------------------------------------
# 6. Preprocessor builder (same as notebook logic)
# -------------------------------------------------------------------
def build_preprocessor(X_sample: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X_sample.select_dtypes(include="number").columns.tolist()
    categorical_features = X_sample.select_dtypes(exclude="number").columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor, numeric_features, categorical_features

preprocessor, num_feats, cat_feats = build_preprocessor(X_train)

# -------------------------------------------------------------------
# 7. Model definitions (LR, RF, XGB, LGBM)
# -------------------------------------------------------------------
st.subheader("Model Training & Comparison")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}

if XGBClassifier is not None:
    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        use_label_encoder=False
    )

if LGBMClassifier is not None:
    models["LightGBM"] = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

st.write("Models to evaluate:", list(models.keys()))

run_cv = st.checkbox("Run 5-fold Stratified CV (ROC-AUC) for each model", value=False)

# -------------------------------------------------------------------
# 8. Train & evaluate models with SMOTE
# -------------------------------------------------------------------
if st.button("Train & Compare Models"):
    results = []
    best_pipeline = None
    best_name = None
    best_f1 = -1.0

    for name, est in models.items():
        st.write(f"---")
        st.write(f"**Training model: {name}**")

        pipe = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", est)
        ])

        # Optional cross-validation
        if run_cv:
            st.info("Running 5-fold Stratified CV (ROC-AUC)...")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            try:
                cv_scores = []
                for train_idx, test_idx in skf.split(X_train, y_train):
                    X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
                    y_tr, y_te = y_train.iloc[train_idx], y_train.iloc[test_idx]
                    pipe.fit(X_tr, y_tr)
                    y_proba_cv = pipe.predict_proba(X_te)[:, 1]
                    cv_scores.append(roc_auc_score(y_te, y_proba_cv))
                st.write(f"CV ROC-AUC: mean={np.mean(cv_scores):.4f}, std={np.std(cv_scores):.4f}")
            except Exception as e:
                st.error(f"Cross-validation failed for {name}: {e}")

        # Fit on training set
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        y_proba = pipe.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_proba)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC_AUC": roc_auc
        })

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_pipeline = pipe

    # Show comparison table
    if results:
        results_df = pd.DataFrame(results).sort_values(by="F1", ascending=False)
        st.write("### Model Comparison (Validation Set)")
        st.dataframe(results_df.style.highlight_max(axis=0))

    # Visualize best model performance
    if best_pipeline is not None:
        st.write(f"### Best model based on F1: **{best_name}**")

        y_pred_best = best_pipeline.predict(X_val)
        y_proba_best = best_pipeline.predict_proba(X_val)[:, 1]

        cm = confusion_matrix(y_val, y_pred_best)

        # Confusion matrix
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title(f"Confusion Matrix – {best_name}")
        st.pyplot(fig_cm)

        # ROC Curve
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_val, y_proba_best, ax=ax_roc)
        ax_roc.set_title(f"ROC Curve – {best_name}")
        st.pyplot(fig_roc)

        st.write(f"Best model ROC-AUC: {roc_auc_score(y_val, y_proba_best):.4f}")

        # Save best model fitted on full data
        with st.spinner("Training best model on full data & saving pipeline..."):
            full_pipe = ImbPipeline(steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("model", models[best_name])
            ])
            full_pipe.fit(X, y)
            try:
                joblib.dump(full_pipe, MODEL_SAVE_PATH)
                st.success(f"Saved best model pipeline to `{MODEL_SAVE_PATH}`")
            except Exception as e:
                st.error(f"Failed to save model: {e}")

# -------------------------------------------------------------------
# 9. Single profile prediction
# -------------------------------------------------------------------
st.subheader("Single Profile Prediction")

pipeline_for_prediction = loaded_pipeline
if pipeline_for_prediction is None and os.path.exists(MODEL_SAVE_PATH):
    try:
        pipeline_for_prediction = joblib.load(MODEL_SAVE_PATH)
        st.info(f"Loaded saved pipeline from `{MODEL_SAVE_PATH}`")
    except Exception:
        pipeline_for_prediction = None

col1, col2 = st.columns(2)

with col1:
    city_development_index = st.number_input(
        "City development index (0–1)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01
    )
    training_hours = st.number_input(
        "Training hours",
        min_value=0, max_value=10000, value=20, step=1
    )
    experience_years = st.number_input(
        "Experience (years)",
        min_value=0.0, max_value=50.0, value=3.0, step=0.5
    )
    last_new_job_num = st.number_input(
        "Years since last job change (0 = never)",
        min_value=0.0, max_value=50.0, value=2.0, step=1.0
    )

with col2:
    company_size = st.selectbox(
        "Company size",
        ["missing", "<10", "10-49", "50-99", "100-499", "500-999", "1000-4999", "5000-9999", "10000+"]
    )
    if company_size == "<10":
        company_size_num = 5.0
    elif company_size == "10000+":
        company_size_num = 10000.0
    elif company_size == "missing":
        company_size_num = np.nan
    else:
        try:
            a, b = company_size.split("-")
            company_size_num = (int(a) + int(b)) / 2.0
        except Exception:
            company_size_num = np.nan

    gender = st.selectbox("Gender", ["Male", "Female", "Other", "missing"])
    relevent_experience = st.selectbox(
        "Relevant experience",
        ["Has relevant experience", "No relevant experience"]
    )
    enrolled_university = st.selectbox(
        "Enrolled university",
        ["no_enrollment", "Full time course", "Part time course", "missing"]
    )
    education_level = st.selectbox(
        "Education level",
        ["Graduate", "Masters", "High School", "Phd", "Primary School", "Missing"]
    )
    major_discipline = st.selectbox(
        "Major discipline",
        ["STEM", "Arts", "Business", "Humanities", "No Major", "Other", "missing"]
    )
    company_type = st.selectbox(
        "Company type",
        ["Pvt Ltd", "Funded Startup", "Early Stage Startup", "Public Sector", "NGO", "Other", "missing"]
    )

if st.button("Predict job change likelihood"):
    if pipeline_for_prediction is None:
        st.error("No pipeline available for prediction. Train models above or upload a pipeline in the sidebar.")
    else:
        input_df = pd.DataFrame([{
            "city_development_index": city_development_index,
            "training_hours": training_hours,
            "experience_years": experience_years,
            "last_new_job_num": last_new_job_num,
            "company_size_num": company_size_num,
            "gender": gender,
            "relevent_experience": relevent_experience,
            "enrolled_university": enrolled_university,
            "education_level": education_level,
            "major_discipline": major_discipline,
            "company_type": company_type
        }])

        # Align columns with what the preprocessor expects
        try:
            if "preprocessor" in pipeline_for_prediction.named_steps:
                preproc = pipeline_for_prediction.named_steps["preprocessor"]
            elif "preproc" in pipeline_for_prediction.named_steps:
                preproc = pipeline_for_prediction.named_steps["preproc"]
            else:
                preproc = None

            if preproc is not None:
                # ColumnTransformer: transformers_ -> (name, transformer, column_list)
                numeric_cols = list(preproc.transformers_[0][2])
                categorical_cols = list(preproc.transformers_[1][2])
                expected = numeric_cols + categorical_cols

                for col in expected:
                    if col not in input_df.columns:
                        input_df[col] = np.nan
                input_df = input_df[expected]
        except Exception:
            # If introspection fails, try with whatever columns exist
            pass

        try:
            proba = pipeline_for_prediction.predict_proba(input_df)[:, 1][0]
            pred = int(pipeline_for_prediction.predict(input_df)[0])
            st.metric("Probability (looking for job change)", f"{proba:.3f}")
            st.write("Predicted class (0 = Not looking, 1 = Looking):", pred)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.markdown(
    """
**Notes:**
- Feature engineering, preprocessing, and models are aligned with the project notebook.
- SMOTE is used inside the pipeline to handle class imbalance.
- The app provides: EDA, model comparison, performance visualization, and single-profile prediction.
"""
)