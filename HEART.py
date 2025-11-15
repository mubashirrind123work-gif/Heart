# heart_disease_app.py
# Heart Disease Risk Prediction — Streamlit Dashboard
# Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, streamlit

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
sns.set_style("whitegrid")
st.set_page_config(page_title="Heart Disease Risk Predictor", layout="wide")

st.title("Heart Disease Risk Prediction Dashboard")
st.markdown(
    "Predict risk of heart disease using common clinical features. "
    "Upload your CSV or use the sample dataset to explore, visualize, and predict."
)
st.sidebar.header("Data Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV file (must contain the required columns)", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample dataset (recommended if you don't have a CSV)", value=(uploaded_file is None))

@st.cache_data
def load_sample(path="/mnt/data/heart_disease_dataset_1000.csv"):
    try:
        df_sample = pd.read_csv(path)
        return df_sample
    except Exception:
        return None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Uploaded CSV loaded.")
elif use_sample:
    df = load_sample()
    if df is None:
        st.sidebar.error("Sample dataset not found in environment. Please upload a CSV.")
        st.stop()
    st.sidebar.info("Using bundled sample dataset (1000 rows).")
else:
    st.info("Please upload a CSV or check 'Use sample dataset'.")
    st.stop()
required_cols = [
    'Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
    'RestingECG','MaxHR','ExerciseAngina','Oldpeak','Slope','NumVessels',
    'Thalassemia','HeartDisease'
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"The dataset is missing required columns: {missing}")
    st.stop()

df['HeartDisease'] = df['HeartDisease'].astype(int)
for col in ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak','NumVessels']:
    if df[col].dtype == object:
        df[col] = pd.to_numeric(df[col], errors='coerce')

tab1, tab2, tab3 = st.tabs(["Data Preview", "Visualizations", "Model & Predict"])
with tab1:
    st.header("Dataset Preview & Summary")
    st.dataframe(df.head(10), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", df.shape[0])
    with c2:
        st.metric("Columns", df.shape[1])
    with c3:
        st.write("Missing values per column")
        st.dataframe(df.isnull().sum())

    st.markdown("### Class distribution (HeartDisease)")
    counts = df['HeartDisease'].value_counts().rename({0: 'No Disease', 1: 'Disease'})
    fig, ax = plt.subplots(figsize=(6,3))
    sns.barplot(x=counts.index.astype(str), y=counts.values, palette='Set1', ax=ax)
    ax.set_xticklabels(['No Disease', 'Disease'])
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.markdown("### Data types")
    st.dataframe(df.dtypes.astype(str))

with tab2:
    st.header("Exploratory Data Analysis (EDA)")
    st.subheader("Age Distribution by Heart Disease Status")
    fig1, ax1 = plt.subplots(figsize=(8,4))
    sns.histplot(data=df, x='Age', hue='HeartDisease', bins=20, kde=True, stat="count", palette=['#2ca02c','#d62728'], ax=ax1)
    ax1.legend(labels=['No Disease','Disease'])
    st.pyplot(fig1)
    st.subheader("Chest Pain Type vs Disease Rate")
    cp_rate = df.groupby('ChestPainType')['HeartDisease'].mean().reset_index().sort_values('HeartDisease', ascending=False)
    fig2, ax2 = plt.subplots(figsize=(8,4))
    sns.barplot(x='ChestPainType', y='HeartDisease', data=cp_rate, palette='Blues_r', ax=ax2)
    ax2.set_ylabel("Disease Rate")
    st.pyplot(fig2)
    st.subheader("Cholesterol Distribution (mg/dL)")
    fig3, ax3 = plt.subplots(figsize=(8,4))
    sns.boxplot(x='HeartDisease', y='Cholesterol', data=df, palette=['#2ca02c','#d62728'], ax=ax3)
    ax3.set_xticklabels(['No Disease','Disease'])
    st.pyplot(fig3)
    st.subheader("Max Heart Rate vs Age (colored by Heart Disease)")
    fig4, ax4 = plt.subplots(figsize=(8,5))
    sns.scatterplot(x='Age', y='MaxHR', hue='HeartDisease', data=df, palette=['#2ca02c','#d62728'], ax=ax4, alpha=0.7)
    st.pyplot(fig4)
    st.subheader("Correlation Heatmap (numeric features)")
    num_cols = ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak','NumVessels','HeartDisease']
    fig5, ax5 = plt.subplots(figsize=(8,6))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', ax=ax5, fmt=".2f")
    st.pyplot(fig5)

with tab3:
    st.header("Model Training, Evaluation & Manual Prediction")

    st.markdown("### 1) Model settings")
    test_size = st.slider("Test set size (%)", 10, 40, 20)
    random_state = st.number_input("Random state", 0, 9999, 42)

    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']
    X_encoded = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size/100, random_state=random_state, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    st.subheader("Model Performance on Test Set")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Accuracy", f"{acc*100:.2f}%")
        st.metric("ROC AUC", f"{roc_auc:.3f}")
    with c2:
        st.write("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_xticklabels(['No Disease','Disease'])
        ax_cm.set_yticklabels(['No Disease','Disease'])
        st.pyplot(fig_cm)

    st.markdown("#### Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=['No Disease','Disease']))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc, ax_roc = plt.subplots(figsize=(6,4))
    ax_roc.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
    ax_roc.plot([0,1],[0,1],'--', color='gray')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()
    st.pyplot(fig_roc)

    st.subheader("Feature Importances")
    feat_imp = pd.Series(model.feature_importances_, index=X_encoded.columns).sort_values(ascending=False).head(20)
    fig_imp, ax_imp = plt.subplots(figsize=(8,5))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax_imp)
    ax_imp.set_xlabel("Importance")
    st.pyplot(fig_imp)

    st.markdown("---")
    st.subheader("Manual Prediction (single patient)")

    with st.form("manual_input"):
        colA, colB = st.columns(2)
        with colA:
            age_in = st.number_input("Age", min_value=18, max_value=100, value=55)
            sex_in = st.selectbox("Sex", options=[1,0], format_func=lambda x: "Male" if x==1 else "Female")
            cp_in = st.selectbox("Chest Pain Type", options=['TA','ATA','NAP','ASY'])
            restbp_in = st.number_input("Resting BP (mmHg)", 80, 250, 130)
            chol_in = st.number_input("Cholesterol (mg/dL)", 100, 600, 220)
            fbs_in = st.selectbox("Fasting BS > 120 mg/dL?", options=[0,1])
            recg_in = st.selectbox("Resting ECG", options=['Normal','ST','LVH'])
        with colB:
            maxhr_in = st.number_input("Max Heart Rate Achieved", 60, 230, 150)
            exang_in = st.selectbox("Exercise Induced Angina?", options=[0,1])
            oldpeak_in = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
            slope_in = st.selectbox("ST slope", options=[1,2,3])
            vessels_in = st.selectbox("Number of major vessels (0-3)", options=[0,1,2,3])
            thal_in = st.selectbox("Thalassemia", options=['Normal','Fixed','Reversible'])

        submitted = st.form_submit_button("Predict Heart Disease Risk")

    if submitted:
        # Build input DataFrame
        input_df = pd.DataFrame({
            'Age':[age_in],
            'Sex':[sex_in],
            'ChestPainType':[cp_in],
            'RestingBP':[restbp_in],
            'Cholesterol':[chol_in],
            'FastingBS':[fbs_in],
            'RestingECG':[recg_in],
            'MaxHR':[maxhr_in],
            'ExerciseAngina':[exang_in],
            'Oldpeak':[oldpeak_in],
            'Slope':[slope_in],
            'NumVessels':[vessels_in],
            'Thalassemia':[thal_in]
        })

        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

        pred_prob = model.predict_proba(input_encoded)[0][1]
        pred_label = model.predict(input_encoded)[0]

        st.metric("Predicted probability of Heart Disease", f"{pred_prob*100:.2f}%")
        if pred_label == 1:
            st.error("Model predicts HIGH RISK of Heart Disease.")
        else:
            st.success("Model predicts LOW RISK of Heart Disease.")

        st.markdown("**Model guidance:** This is a predictive aid — not a medical diagnosis. Always consult a qualified medical professional for clinical decisions.")
