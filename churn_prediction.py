# churn_prediction_dashboard.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import lime
import lime.lime_tabular
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Churn + Sales Dashboard", layout="wide")

# Load dataset
uploaded_file = st.file_uploader("📂 Upload your customer dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown("### 🔍 Dataset Preview")
    st.dataframe(df.head())

    st.markdown("### 🧾 Column Summary")
    st.write(df.dtypes)

    required_columns = ["MonthlyCharges", "tenure", "Churn"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        st.stop()

    st.success(f"✅ Dataset Loaded - {df.shape[0]} rows, {df.shape[1]} columns")

    # Clean & preprocess
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].mean(), inplace=True)

    # Encode categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in ["customerID", "Churn"]:
        if col in cat_cols:
            cat_cols.remove(col)
    df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Feature & target
    drop_cols = [col for col in ["customerID", "Churn"] if col in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["Churn"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Model selection
    st.markdown("### 🧠 Choose a Model")
    model_choice = st.selectbox("Select Model", ["XGBoost", "Logistic Regression", "SVM", "KNN"])

    if model_choice == "XGBoost":
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_choice == "SVM":
        model = SVC(probability=True)
    elif model_choice == "KNN":
        model = KNeighborsClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Accuracy metrics
    acc = accuracy_score(y_test, y_pred)
    st.metric("🎯 Accuracy", f"{acc:.2f}")
    st.markdown("### 📊 Classification Report")
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    st.dataframe(report_df)

    # Customer selection
    st.markdown("### 🔍 Customer Churn Risk")
    X_test_reset = X_test.reset_index(drop=True)
    y_prob_reset = pd.Series(y_prob).reset_index(drop=True)
    customer_ids = df.loc[X_test.index, "customerID"].reset_index(drop=True) if "customerID" in df.columns else pd.Series([f"Customer {i}" for i in range(len(X_test))])

    selected_customer_id = st.selectbox("Select Customer ID", customer_ids)
    cust_index = customer_ids[customer_ids == selected_customer_id].index[0]

    cust_prob = y_prob[cust_index]
    cust_status = "⚠️ At Risk" if cust_prob > 0.5 else "✅ Safe"

    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(cust_prob * 100, 2),
        number={'suffix': "%", 'valueformat': ".2f"},
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        title={'text': "Churn Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "orange"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': cust_prob * 100
            }
        }
    ))
    st.plotly_chart(gauge)
    st.markdown(f"**Customer Status:** {cust_status}")

    # ✅ LIME Explanation
    st.markdown("### 🧠 LIME Explanation")
    with st.spinner("Generating LIME explanation..."):
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=X.columns.tolist(),
            class_names=["No", "Yes"],
            mode="classification"
        )
        lime_exp = explainer.explain_instance(
            data_row=X_test.iloc[cust_index],
            predict_fn=model.predict_proba,
            num_features=10
        )
        fig = lime_exp.as_pyplot_figure(label=1)
        fig.patch.set_facecolor('#f9f9f9')
        fig.suptitle('Top Features Impacting Churn (LIME)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()

    # ✅ Customer Segmentation with KMeans
    st.markdown("### 👥 Customer Segmentation")
    kmeans = KMeans(n_clusters=3, random_state=42)
    segments = kmeans.fit_predict(X)
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X)

    pca_df = pd.DataFrame({
        "PCA1": pca_components[:, 0],
        "PCA2": pca_components[:, 1],
        "Segment": segments
    })

    fig2, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x="PCA1", y="PCA2", hue="Segment", data=pca_df, palette="Set2", ax=ax)
    ax.set_title("Customer Segments by PCA")
    st.pyplot(fig2)

    # ✅ DBSCAN Clustering
    st.subheader("DBSCAN Clustering + Outliers")
    eps_val = st.slider("DBSCAN - Epsilon (eps)", 0.1, 5.0, 1.2, step=0.1)
    min_samples_val = st.slider("DBSCAN - Min Samples", 1, 20, 5)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    db_labels = dbscan.fit_predict(X_scaled)

    # Add DBSCAN cluster labels to PCA data
    pca_df["DBSCAN_Cluster"] = db_labels

    # Count cluster frequencies
    cluster_counts = pd.Series(db_labels).value_counts()

    # Limit legend to top 10 clusters + noise
    top_clusters = cluster_counts.head(10).index.tolist()
    pca_df["Cluster_Label"] = pca_df["DBSCAN_Cluster"].apply(
        lambda x: x if x in top_clusters else "Other"
    )

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x="PCA1",
        y="PCA2",
        hue="Cluster_Label",
        data=pca_df,
        palette="Set2",
        ax=ax3,
        s=40,
        alpha=0.7
    )
    ax3.set_title("DBSCAN Clusters (Top 10 + Noise)")
    ax3.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig3)

    # ✅ Revenue by Churn
    st.markdown("### 💸 Revenue Comparison by Churn Status")
    churn_revenue = df.groupby("Churn")["MonthlyCharges"].sum()
    st.bar_chart(churn_revenue.rename(index={0: "Retained", 1: "Churned"}))

else:
    st.warning("⚠️ Please upload a CSV file to proceed.")
