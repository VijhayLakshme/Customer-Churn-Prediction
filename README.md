# Customer-Churn-Prediction
📊 Customer Churn Prediction 
An interactive Streamlit dashboard for analyzing customer churn, performing sales trend analysis, and segmenting customers using machine learning models like XGBoost, Logistic Regression, SVM, KNN, KMeans, and DBSCAN. This tool provides LIME-based explainability, churn probability gauges, and flexible dataset uploads for real-time predictions.

📁 Features

✅ Upload your own CSV dataset
🔍 View dataset preview and column data types
🤖 Choose from multiple ML models:

-XGBoost
-Logistic Regression
-Support Vector Machine (SVM)
-K-Nearest Neighbors (KNN)

🧠 LIME Explanation for individual churn prediction
🎯 Churn Probability Gauge per customer
👥 Customer Segmentation:

-KMeans clustering

-DBSCAN clustering (with sliders for eps and min_samples)

📈 Sales Trend Analysis (based on tenure & charges)
💸 Revenue Comparison by Churn status
📦 Tech Stack

-Python 3.x
-Streamlit
-XGBoost
-scikit-learn
-LIME
-Pandas, NumPy, Matplotlib, Seaborn
-Plotly (for gauge visualization)

🚀 How to Run
~Go to your terminal (or powershell) and open your folder like this:
cd customer-churn-dashboard

~Install required packages:
pip install -> the following in your terminal
streamlit
pandas
numpy
scikit-learn
xgboost
lime
matplotlib
seaborn
plotly

~Run the Streamlit app:
streamlit run churn_prediction_dashboard.py

📂 Dataset Format
Your dataset should include the following essential columns:
-MonthlyCharges
-tenure
-Churn (with values "Yes" or "No")

Optional: TotalCharges, customerID, and any additional numerical or categorical features for model training.

