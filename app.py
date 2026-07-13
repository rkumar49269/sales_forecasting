import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from matplotlib.ticker import FuncFormatter

sns.set_theme(style='whitegrid')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title='Demand Intelligence Dashboard',
    page_icon='📈',
    layout='wide',
)

st.markdown(
    """
    <style>
        :root {
            --bg: #f4f8fb;
            --panel: #ffffff;
            --panel-soft: #edf5fb;
            --border: #d6e4ef;
            --text: #12324a;
            --muted: #547086;
            --primary: #146aa3;
            --primary-hover: #0f5685;
            --accent: #2a8f92;
            --shadow: 0 14px 32px rgba(14, 42, 66, 0.08);
        }
        html, body, [data-testid="stAppViewContainer"] {
            background: var(--bg);
            color: var(--text);
        }
        .main .block-container {
            padding-top: 1.4rem;
            padding-bottom: 1.6rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f7fbfe 0%, #eef6fb 100%);
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label {
            color: var(--text) !important;
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: var(--shadow);
        }
        [data-testid="stMetric"] label,
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {
            color: var(--text) !important;
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            background: var(--panel);
            border-radius: 16px;
            box-shadow: var(--shadow);
        }
        [data-testid="stButton"] > button {
            background: var(--panel);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.78rem 1rem;
            font-weight: 700;
            width: 100%;
            margin-bottom: 0.45rem;
            box-shadow: none;
            transition: all 0.2s ease;
        }
        [data-testid="stButton"] > button:hover {
            border-color: var(--primary);
            color: var(--primary);
            background: #f7fbfe;
        }
        .stAlert {
            border-left: 4px solid var(--primary);
        }
        .stMarkdown h1,
        .stMarkdown h2,
        .stMarkdown h3,
        .stMarkdown h4,
        h1, h2, h3, h4 {
            color: var(--text);
        }
        .stMarkdown p,
        .stMarkdown li,
        .stCaption {
            color: var(--muted);
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.3rem;
        }
        hr {
            border-color: var(--border);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- DATA LOADING ---
@st.cache_data
def load_assets():
    df = pd.read_csv('clean_superstore_data.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    anomalies = pd.read_csv('anomalies_data.csv')
    anomalies['Order Date'] = pd.to_datetime(anomalies['Order Date'])
    clusters = pd.read_csv('cluster_data.csv')

    with open('segment_forecasts.pkl', 'rb') as f:
        forecasts = pickle.load(f)

    return df, anomalies, clusters, forecasts

@st.cache_data
def compute_forecast_metrics(df, forecasts):
    metrics = {}
    for segment, forecast_df in forecasts.items():
        forecast_copy = forecast_df.copy()
        forecast_copy['ds'] = pd.to_datetime(forecast_copy['ds'])

        if segment.endswith(' Region'):
            region = segment.replace(' Region', '')
            actual_sales = (
                df[df['Region'] == region]
                .set_index('Order Date')
                .resample('ME')['Sales']
                .sum()
            )
        else:
            actual_sales = (
                df[df['Category'] == segment]
                .set_index('Order Date')
                .resample('ME')['Sales']
                .sum()
            )

        forecast_monthly = forecast_copy.set_index('ds')['yhat']
        merged = forecast_monthly.to_frame('yhat').join(actual_sales.to_frame('actual'), how='inner').dropna()

        if not merged.empty:
            mae = float((merged['actual'] - merged['yhat']).abs().mean())
            rmse = float(np.sqrt(((merged['actual'] - merged['yhat']) ** 2).mean()))
        else:
            mae = None
            rmse = None

        metrics[segment] = {'MAE': mae, 'RMSE': rmse}

    return metrics

@st.cache_data
def get_forecast_options(forecasts):
    categories = [key for key in forecasts.keys() if not key.endswith(' Region')]
    regions = [key for key in forecasts.keys() if key.endswith(' Region')]
    return categories, regions


def currency_format(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'N/A'
    return f'${value:,.0f}'


def styled_table(dataframe):
    return (
        dataframe.style
        .set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#146aa3'), ('color', 'white'), ('font-weight', '700'), ('border', '1px solid #d6e4ef')]},
            {'selector': 'td', 'props': [('background-color', '#ffffff'), ('color', '#12324a'), ('border', '1px solid #d6e4ef')]},
            {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f7fbfe')]},
            {'selector': 'caption', 'props': [('caption-side', 'top'), ('color', '#547086'), ('font-weight', '600')]},
        ])
        .set_properties(**{'font-size': '0.95rem', 'padding': '0.55rem'})
        .hide(axis='index')
    )


df, anomalies, clusters, forecasts = load_assets()
forecast_categories, forecast_regions = get_forecast_options(forecasts)
forecast_metrics = compute_forecast_metrics(df, forecasts)

# --- SIDEBAR NAVIGATION ---
if 'page' not in st.session_state:
    st.session_state.page = '1. Sales Overview'

PAGE_OPTIONS = [
    '1. Sales Overview',
    '2. Forecast Explorer',
    '3. Anomaly Report',
    '4. Product Segments',
]

with st.sidebar:
    st.markdown('## Navigation')
    for option in PAGE_OPTIONS:
        button_type = 'primary' if st.session_state.page == option else 'secondary'
        if st.button(option, key=option, type=button_type):
            st.session_state.page = option

    st.markdown('---')
    st.markdown('### Quick links')
    st.write('Use the buttons above to switch views.')
    st.write(f'**Current page:** {st.session_state.page}')

page = st.session_state.page

st.markdown(
    f"""
    <div style="
        background: linear-gradient(135deg, #ffffff 0%, #eef6fb 100%);
        border: 1px solid #d6e4ef;
        border-radius: 22px;
        padding: 1.3rem 1.5rem;
        box-shadow: 0 14px 32px rgba(14, 42, 66, 0.08);
        margin-bottom: 1.25rem;
    ">
        <div style="color:#12324a; font-size:2rem; font-weight:800; margin-top:0.2rem;">Demand Intelligence Dashboard</div>
        <div style="color:#547086; font-size:1rem; margin-top:0.35rem; max-width: 900px;">
            Clear sales insights, forecast exploration, anomaly reporting, and product segmentation presented with a professional visual system.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- PAGE 1: SALES OVERVIEW ---
if page == '1. Sales Overview':
    st.title('Sales Overview Dashboard')
    st.markdown('Use the filters to explore sales by region and category.')

    with st.sidebar.expander('Filters', expanded=True):
        selected_region = st.multiselect(
            'Select Region',
            options=sorted(df['Region'].unique()),
            default=sorted(df['Region'].unique()),
        )
        selected_category = st.multiselect(
            'Select Category',
            options=sorted(df['Category'].unique()),
            default=sorted(df['Category'].unique()),
        )

    filtered_df = df[(df['Region'].isin(selected_region)) & (df['Category'].isin(selected_category))]
    monthly_sales = filtered_df.set_index('Order Date').resample('ME')['Sales'].sum().reset_index()
    yearly_sales = filtered_df.groupby('Year')['Sales'].sum().reset_index()

    total_sales = filtered_df['Sales'].sum()
    current_year = filtered_df['Year'].max() if not filtered_df.empty else None
    sales_current_year = filtered_df.loc[filtered_df['Year'] == current_year, 'Sales'].sum() if current_year is not None else 0
    sales_prior_year = filtered_df.loc[filtered_df['Year'] == current_year - 1, 'Sales'].sum() if current_year is not None else 0
    yoy_growth = ((sales_current_year - sales_prior_year) / sales_prior_year) if sales_prior_year else np.nan
    avg_monthly_sales = monthly_sales['Sales'].mean() if not monthly_sales.empty else 0

    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    kpi_col1.metric('Total Sales', currency_format(total_sales))
    kpi_col2.metric(
        'Latest Year Sales',
        currency_format(sales_current_year),
        delta=f'{yoy_growth:.1%}' if not np.isnan(yoy_growth) else 'N/A',
    )
    kpi_col3.metric('Average Monthly Sales', currency_format(avg_monthly_sales))

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader('Total Sales by Year')
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=yearly_sales, x='Year', y='Sales', hue='Year', palette='Blues', ax=ax, edgecolor='white', legend=False)
        ax.set_ylabel('Sales')
        ax.set_xlabel('Year')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
        ax.set_title('Annual Sales')
        st.pyplot(fig)

    with chart_col2:
        st.subheader('Monthly Sales Trend')
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.lineplot(data=monthly_sales, x='Order Date', y='Sales', marker='o', color='#146aa3', ax=ax)
        ax.set_ylabel('Sales')
        ax.set_xlabel('Month')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
        ax.set_title('Monthly Sales Trend')
        st.pyplot(fig)

    extra_col1, extra_col2 = st.columns(2)
    with extra_col1:
        st.subheader('Sales by Region')
        region_sales = filtered_df.groupby('Region')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=region_sales, x='Sales', y='Region', hue='Region', palette='Blues', ax=ax, legend=False)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
        st.pyplot(fig)

    with extra_col2:
        st.subheader('Sales by Category')
        category_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index().sort_values('Sales', ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=category_sales, x='Sales', y='Category', hue='Category', palette='GnBu', ax=ax, legend=False)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
        st.pyplot(fig)

# --- PAGE 2: FORECAST EXPLORER ---
elif page == '2. Forecast Explorer':
    st.title('Forecast Explorer')
    st.markdown('Choose a forecast segment and horizon to inspect the model output.')

    forecast_scope = st.selectbox('Forecast by', ['Category', 'Region'])
    forecast_keys = forecast_categories if forecast_scope == 'Category' else forecast_regions
    segment = st.selectbox('Select segment', forecast_keys)
    horizon = st.slider('Forecast horizon (months ahead)', min_value=1, max_value=3, value=3)

    forecast_df = forecasts[segment].copy()
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    forecast_df = forecast_df.sort_values('ds').reset_index(drop=True)
    latest_forecast = forecast_df.tail(horizon).copy()
    latest_forecast['Month'] = latest_forecast['ds'].dt.strftime('%b %Y')

    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    metrics = forecast_metrics.get(segment, {})
    mae_value = metrics.get('MAE')
    rmse_value = metrics.get('RMSE')
    col_metrics1.metric('Selected Segment', segment)
    col_metrics2.metric('MAE', currency_format(mae_value))
    col_metrics3.metric('RMSE', currency_format(rmse_value))

    if mae_value is None or rmse_value is None:
        st.info('Forecast performance is available only when actual monthly sales overlap with the selected segment.')

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(forecast_df['ds'], forecast_df['yhat'], color='#146aa3', lw=2.5, label='Forecast')
    ax.fill_between(
        forecast_df['ds'],
        forecast_df['yhat_lower'],
        forecast_df['yhat_upper'],
        color='#146aa3',
        alpha=0.2,
        label='Confidence interval',
    )
    ax.set_title(f'Forecast trajectory for {segment}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Sales')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend()
    st.pyplot(fig)

    st.subheader('Forecast Output')
    display_df = latest_forecast[['Month', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    display_df.columns = ['Date', 'Predicted Sales', 'Lower Bound (Worst Case)', 'Upper Bound (Best Case)']
    display_df['Predicted Sales'] = display_df['Predicted Sales'].map(currency_format)
    display_df['Lower Bound (Worst Case)'] = display_df['Lower Bound (Worst Case)'].map(currency_format)
    display_df['Upper Bound (Best Case)'] = display_df['Upper Bound (Best Case)'].map(currency_format)
    st.dataframe(styled_table(display_df), use_container_width=True, hide_index=True)

# --- PAGE 3: ANOMALY REPORT ---
elif page == '3. Anomaly Report':
    st.title('Supply Chain Anomaly Report')
    st.markdown('Visualizing outliers in weekly revenue and the dates where anomalies were detected.')

    weekly_sales = df.set_index('Order Date').resample('W')['Sales'].sum().reset_index()
    anomalies_sorted = anomalies.sort_values('Order Date')

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(weekly_sales['Order Date'], weekly_sales['Sales'], label='Weekly Sales', color='#146aa3', alpha=0.85)
    ax.scatter(anomalies_sorted['Order Date'], anomalies_sorted['Sales'], color='#8d3b4b', s=100, label='Anomaly', zorder=5)
    ax.set_title('Weekly Sales with Detected Anomalies')
    ax.set_xlabel('Week')
    ax.set_ylabel('Sales')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend()
    st.pyplot(fig)

    st.subheader('Detected Anomaly Dates')
    anomaly_display = anomalies_sorted[['Order Date', 'Sales', 'Global_Sales', 'Anomaly_Iso']].copy()
    anomaly_display['Order Date'] = anomaly_display['Order Date'].dt.strftime('%Y-%m-%d')
    anomaly_display['Sales'] = anomaly_display['Sales'].map(currency_format)
    anomaly_display['Global_Sales'] = anomaly_display['Global_Sales'].map(currency_format)
    anomaly_display.rename(columns={
        'Order Date': 'Date',
        'Sales': 'Sales Value',
        'Global_Sales': 'Global Sales',
        'Anomaly_Iso': 'Anomaly Label',
    }, inplace=True)
    st.dataframe(styled_table(anomaly_display), use_container_width=True, hide_index=True)

# --- PAGE 4: PRODUCT DEMAND SEGMENTS ---
elif page == '4. Product Segments':
    st.title('Product Demand Segmentation')
    st.markdown('Review the cluster map and sub-category demand segments.')

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.scatterplot(
        data=clusters,
        x='PCA1',
        y='PCA2',
        hue='Business_Label',
        palette='GnBu',
        s=170,
        edgecolor='black',
        ax=ax,
    )
    ax.set_title('2D PCA Cluster Map')
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.legend(title='Demand Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
    st.pyplot(fig)

    st.subheader('Sub-Category Assignments')
    display_clusters = clusters[['Sub-Category', 'Business_Label', 'Total_Sales', 'YoY_Growth', 'Sales_Volatility']].copy()
    display_clusters['Total_Sales'] = display_clusters['Total_Sales'].map(currency_format)
    display_clusters['YoY_Growth'] = display_clusters['YoY_Growth'].map(lambda x: f'{x:.1%}')
    display_clusters['Sales_Volatility'] = display_clusters['Sales_Volatility'].map(currency_format)
    st.dataframe(
        styled_table(display_clusters.sort_values(['Business_Label', 'Sub-Category'])),
        use_container_width=True,
        hide_index=True,
    )
