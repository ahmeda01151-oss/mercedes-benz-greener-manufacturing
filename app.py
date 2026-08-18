import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# إعداد الصفحة
st.set_page_config(page_title="Mercedes-Benz Testing Time", layout="wide", page_icon="🚗")

# تحميل الموديل والأدوات المحفوظة
@st.cache_resource
def load_artifacts():
    return joblib.load('mercedes_artifacts.joblib')

artifacts = load_artifacts()
model = artifacts['model']
scaler = artifacts['scaler']
encoders = artifacts['encoders']
feature_names = artifacts['feature_names']
results_df = artifacts['results_df']

# القائمة الجانبية للتنقل بين الصفحات
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "🔮 Prediction", "📊 Compare Models", "📈 Dashboard"])

# ==========================================
# 1. Home Page
# ==========================================
if page == "🏠 Home":
    st.title("🚗 Mercedes-Benz Greener Manufacturing")
    st.subheader("Optimizing Testing Time on the Production Line")
    
    st.markdown("""
    ### 🎯 Business Objective
    To ensure the safety and reliability of every Mercedes-Benz vehicle, cars undergo rigorous testing configurations.
    The goal of this project is to **predict the testing time (in seconds)** based on vehicle features to reduce carbon footprint and streamline manufacturing.
    
    ---
    ### 🛠️ Pipeline Highlights
    * **Data Preprocessing:** Handled zero-variance features, encoded categorical variables, and eliminated leakage using `StandardScaler`.
    * **Feature Shrinkage:** Applied **Lasso Regression (L1 Penalty)** to reduce features from 364 to 98 key components.
    * **Model Selection:** Gradient Boosting (`XGBoost`) achieved the highest $R^2$ score after 5-fold cross-validation.
    * **Explainability:** Integrated feature importance metrics to explain model decisions.
    """)

# ==========================================
# 2. Prediction Page
# ==========================================
elif page == "🔮 Prediction":
    st.title("🔮 Predict Testing Time")
    st.write("Provide vehicle configuration parameters to estimate inspection duration.")

    tab1, tab2 = st.tabs(["Manual Sample Input", "Upload CSV Batch"])
    
    with tab1:
        st.markdown("##### Categorical Features (`X0` - `X8`)")
        cat_cols = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']
        input_data = {}
        
        c1, c2, c3, c4 = st.columns(4)
        for i, col in enumerate(cat_cols):
            with [c1, c2, c3, c4][i % 4]:
                options = list(encoders[col].classes_)
                input_data[col] = st.selectbox(f"Feature {col}", options)
        
        # ملء الـ Binary features بقيم افتراضية (0)
        for col in feature_names:
            if col not in cat_cols:
                input_data[col] = 0
                
        if st.button("🚀 Calculate Estimated Time", use_container_width=True):
            input_df = pd.DataFrame([input_data])
            for col in cat_cols:
                input_df[col] = encoders[col].transform(input_df[col].astype(str))
                
            input_scaled = scaler.transform(input_df[feature_names])
            prediction = model.predict(input_scaled)[0]
            
            st.success(f"⏱️ Estimated Vehicle Testing Time: **{prediction:.2f} seconds**")

    with tab2:
        uploaded_file = st.file_uploader("Upload CSV file containing features", type=['csv'])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:", batch_df.head(3))
            
            if st.button("Predict for All Records"):
                processed_batch = batch_df.copy()
                for col in cat_cols:
                    if col in processed_batch.columns:
                        processed_batch[col] = encoders[col].transform(processed_batch[col].astype(str))
                
                features_only = processed_batch[feature_names]
                scaled_batch = scaler.transform(features_only)
                batch_df['Predicted_y'] = model.predict(scaled_batch)
                
                st.write("Predictions:", batch_df[['Predicted_y'] + cat_cols].head())
                st.download_button("Download Predictions CSV", batch_df.to_csv(index=False), "predictions.csv")

# ==========================================
# 3. Compare Page
# ==========================================
elif page == "📊 Compare Models":
    st.title("📊 Model Benchmark & Comparison")
    st.write("Evaluation metrics calculated on the 20% validation split:")
    
    st.dataframe(results_df, use_container_width=True)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(data=results_df, x="R2 Score", y="Model", palette="Blues_r", ax=ax[0])
    ax[0].set_title("$R^2$ Score (Higher is Better)")
    
    sns.barplot(data=results_df, x="RMSE", y="Model", palette="Reds_r", ax=ax[1])
    ax[1].set_title("RMSE (Lower is Better)")
    plt.tight_layout()
    st.pyplot(fig)

# ==========================================
# 4. Dashboard Page
# ==========================================
elif page == "📈 Dashboard":
    st.title("📈 Feature Importance & Explanations")
    st.markdown("### Top Influential Features (XGBoost)")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    top_features = [feature_names[i] for i in indices]
    top_scores = importances[indices]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_scores, y=top_features, palette="viridis", ax=ax)
    ax.set_title("Top 15 Most Important Features in Testing Time")
    ax.set_xlabel("Relative Feature Importance Score")
    st.pyplot(fig)