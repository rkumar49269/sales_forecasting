# 📊 Demand Intelligence & Sales Forecasting Dashboard

**Live Application:** [View the Dashboard Here!](https://salesanddemanddashboard.streamlit.app/)

## 🚀 Project Overview
This project is an end-to-end Machine Learning and Data Analytics pipeline designed for modern Supply Chain optimization. Built entirely in Python, this interactive Streamlit dashboard transforms raw, historical e-commerce transaction data into actionable business intelligence. 

The system provides executive stakeholders with automated demand forecasts, anomaly detection for extreme supply chain events, and mathematical product segmentation to drive precise inventory management strategies.

## 🧠 Key Features & Modules

* **📈 1. Sales Overview (EDA):** An interactive business intelligence view of historical revenue, filtered by region and category, highlighting seasonal purchasing patterns.
* **🔮 2. Forecast Explorer (Time-Series Forecasting):** Utilizes **Facebook Prophet** to predict 3-month future revenue across multiple business segments, complete with upper and lower statistical confidence intervals.
* **🚨 3. Anomaly Report (Outlier Detection):** A mathematical audit of the supply chain that isolates severe, non-seasonal revenue spikes and crashes, allowing procurement teams to investigate extreme market events.
* **📦 4. Product Segments (Machine Learning):** Applies **K-Means Clustering** and **Principal Component Analysis (PCA)** to group the product catalog into 4 distinct macro-categories (e.g., *High Volume/Stable Demand*, *Low Volume/High Volatility*) to dictate targeted warehouse stocking strategies.

## 🛠️ Technology Stack
* **Language:** Python
* **Frontend/Deployment:** Streamlit, Streamlit Community Cloud
* **Machine Learning:** Scikit-Learn (K-Means, PCA), Prophet (Time-Series)
* **Data Engineering:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn

## 💻 Local Installation & Setup
To run this dashboard locally on your own machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/sales-demand-dashboard.git](https://github.com/your-username/sales-demand-dashboard.git)
   cd sales-demand-dashboard