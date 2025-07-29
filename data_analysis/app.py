import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Network Performance EDA",
    page_icon="📡",
    layout="wide"
)

# --- Plotting Configuration ---
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# --- Caching Functions for Performance ---
@st.cache_data
def load_data(file_path):
    """Loads and caches the dataset."""
    df = pd.read_csv(file_path)
    df['operator'] = df['operator'].astype('category')
    return df

@st.cache_data
def clean_data(_df):
    """Applies the cleaning strategy from the notebook."""
    df_clean = _df.copy()
    df_clean['has_scell'] = df_clean['SCell_RSRP_max'].notna()
    scell_cols = [c for c in df_clean.columns if c.startswith('SCell_') and pd.api.types.is_numeric_dtype(df_clean[c])]
    df_clean[scell_cols] = df_clean[scell_cols].fillna(0)
    pcell_cols_to_fill = ['PCell_Downlink_bandwidth_MHz', 'PCell_Cell_Identity', 'PCell_freq_MHz']
    for p_col in pcell_cols_to_fill:
        if df_clean[p_col].isnull().sum() > 0:
            mode_val = df_clean[p_col].mode()[0]
            df_clean[p_col] = df_clean[p_col].fillna(mode_val)
    return df_clean

@st.cache_data
def engineer_features(_df):
    """Creates time-based features."""
    df_eng = _df.copy()
    df_eng['datetime'] = pd.to_datetime(df_eng['timestamp'], unit='s')
    df_eng['hour_of_day'] = df_eng['datetime'].dt.hour
    df_eng['day_of_week'] = df_eng['datetime'].dt.strftime('%a')
    return df_eng

# --- Main Application ---
st.title("📡 Exploratory Data Analysis of Network Performance")
st.markdown("This application provides a comprehensive EDA of the network performance dataset, with an improved loading experience and a non-reloading interactive map.")

# --- Enhanced Loading Sequence with st.status ---
with st.status("🚀 Initializing Analysis...", expanded=True) as status:
    st.write("Loading raw data from CSV...")
    df = load_data('Train.csv')
    time.sleep(1) # Small delay for better UX
    
    st.write("Cleaning data and imputing missing values...")
    df_clean = clean_data(df)
    time.sleep(1)

    st.write("Engineering time-based features...")
    df_eng = engineer_features(df_clean)
    time.sleep(0.5)
    
    status.update(label="✅ Analysis Ready!", state="complete", expanded=False)


# --- 1. Data Loading and Initial Inspection ---
st.header("1. Data Loading and Initial Inspection")
st.write(f"**Number of rows:** `{df.shape[0]:,}`")
st.write(f"**Number of columns:** `{df.shape[1]}`")

with st.expander("Show Raw Data Sample and Schema"):
    st.subheader("First 5 Rows")
    st.dataframe(df.head())
    st.subheader("Dataset Schema (Data Types)")
    st.dataframe(df.dtypes.astype(str).rename('Data Type'))

# --- 2. Handling Missing Values ---
st.header("2. Handling Missing Values and Cleaning")
col1, col2 = st.columns(2)
with col1:
    with st.expander("Null Values Before Cleaning", expanded=True):
        null_counts = df.isnull().sum().rename('Null Count').to_frame()
        st.dataframe(null_counts[null_counts['Null Count'] > 0])
with col2:
    with st.expander("Null Values After Cleaning", expanded=True):
        st.write("Cleaned data has 'has_scell' feature added and missing values imputed.")
        null_counts_after = df_clean.isnull().sum().rename('Null Count').to_frame()
        st.dataframe(null_counts_after[null_counts_after['Null Count'] > 0])

# --- 3. Feature Engineering ---
st.header("3. Feature Engineering")
st.write("Added `datetime`, `hour_of_day`, and `day_of_week` features.")
with st.expander("Show Sample of Engineered Data"):
    st.dataframe(df_eng[['id', 'datetime', 'hour_of_day', 'day_of_week', 'has_scell']].head())

# --- 4. Univariate Analysis ---
st.header("4. Univariate Analysis")
st.subheader("4.1. Target Variable Distribution")
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    sns.histplot(df_eng['target'], kde=True, bins=50, ax=ax)
    ax.set_title('Distribution of Target Variable')
    ax.set_xlabel('Target (Assumed Throughput)')
    st.pyplot(fig)
with col2:
    fig, ax = plt.subplots()
    sns.histplot(np.log1p(df_eng['target']), kde=True, bins=50, color='green', ax=ax)
    ax.set_title('Log-Transformed Target Distribution')
    ax.set_xlabel('Log(1 + Target)')
    st.pyplot(fig)

st.subheader("4.2. Categorical & Boolean Feature Distributions")
categorical_features = ['device', 'operator', 'area', 'day_of_week', 'has_scell']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, col_name in enumerate(categorical_features):
    counts = df_eng[col_name].value_counts().reset_index()
    counts.columns = [col_name, 'count']
    sns.barplot(
        data=counts, x='count', y=col_name, ax=axes[i],
        order=counts.sort_values('count', ascending=False)[col_name],
        hue=col_name, palette='viridis', legend=False
    )
    axes[i].set_title(f'Distribution of {col_name}')
    axes[i].set_xlabel('Count')
    axes[i].set_ylabel('')
fig.delaxes(axes[4])
fig.delaxes(axes[5])
plt.tight_layout()
st.pyplot(fig)

# --- 5. Bivariate and Multivariate Analysis ---
st.header("5. Bivariate and Multivariate Analysis")
st.info("Using a sample of the data for performance-intensive plots like scatterplots.")
sample_df = df_eng.sample(n=min(50000, len(df_eng)), random_state=42)

st.subheader("5.1. Target vs. Top Numerical Predictors")
with st.spinner('Generating scatter plots...'):
    top_predictors = ['PCell_Downlink_Average_MCS', 'PCell_SNR_max', 'PCell_RSRP_max']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, col in enumerate(top_predictors):
        sns.scatterplot(x=sample_df[col], y=sample_df['target'], ax=axes[i], alpha=0.5, hue=sample_df['operator'].astype(str), palette='bright')
        axes[i].set_title(f'Target vs. {col}')
        axes[i].set_ylabel('Target (Throughput)')
        axes[i].set_yscale('log')
    plt.tight_layout()
    st.pyplot(fig)

st.subheader("5.2. Target vs. Categorical Features")
with st.spinner('Generating box plots...'):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    categorical_to_plot = ['area', 'operator', 'has_scell']
    palettes = ['crest', 'flare', 'magma']
    for i, col in enumerate(categorical_to_plot):
        sns.boxplot(x=col, y='target', data=sample_df, ax=axes[i], palette=palettes[i])
        axes[i].set_title(f'Target by {col}')
        axes[i].set_yscale('log')
        axes[i].set_ylabel('Target (Log Scale)')
    plt.tight_layout()
    st.pyplot(fig)

st.subheader("5.3. Correlation Matrix")
with st.spinner("Generating correlation matrix and heatmap..."):
    numeric_cols = df_eng.select_dtypes(include=np.number).columns.tolist()
    cols_to_remove = ['timestamp', 'COG', 'id']
    numeric_cols = [c for c in numeric_cols if c not in cols_to_remove]
    corr_matrix = df_eng[numeric_cols].corr()

    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(20, 16))
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, ax=ax)
        ax.set_title('Correlation Matrix of Numerical Features')
        st.pyplot(fig)
    with col2:
        st.write("**Top Correlations with Target:**")
        corr_target = corr_matrix['target'].sort_values(ascending=False).drop('target').to_frame()
        st.dataframe(corr_target.head(15))

# --- 6. Geospatial Analysis ---
st.header("6. Geospatial Analysis")
geo_sample_pd = df_eng.sample(n=min(2000, len(df_eng)), random_state=42)

st.write("🗺️ Map of Network Performance (Sampled Data)")
st.markdown("`Green` = High Throughput, `Orange` = Medium, `Red` = Low. **You can now zoom and pan without reloading the page!**")

map_center = [geo_sample_pd['Latitude'].mean(), geo_sample_pd['Longitude'].mean()]
world_map = folium.Map(location=map_center, zoom_start=12, tiles="cartodbpositron")

log_target = np.log1p(geo_sample_pd['target'])
min_log, max_log = log_target.min(), log_target.max()

def get_color(val):
    if pd.isna(val): return 'gray'
    norm = (np.log1p(val) - min_log) / (max_log - min_log + 1e-6)
    if norm < 0.33: return 'red'
    elif norm < 0.66: return 'orange'
    else: return 'green'

for idx, row in geo_sample_pd.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=3,
        color=get_color(row['target']),
        fill=True,
        popup=f"Target: {row['target']:.0f}<br>Operator: {row['operator']}"
    ).add_to(world_map)

# **THE FIX IS HERE**: returned_objects=[] prevents the map from triggering a rerun
st_folium(world_map, width=1200, height=500, returned_objects=[])

# --- 7. Time-Series Analysis ---
st.header("7. Time-Series Analysis")
with st.spinner("Aggregating and plotting time-series data..."):
    hourly_perf = df_eng.groupby(['hour_of_day', 'operator'])['target'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(x='hour_of_day', y='target', data=hourly_perf, hue='operator', marker='o', palette='plasma', ax=ax)
    ax.set_title('Average Target Throughput by Hour of Day')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Target Throughput')
    ax.set_xticks(np.arange(0, 24, 1))
    ax.grid(True, which='both', linestyle='--')
    st.pyplot(fig)

# --- 8. Summary and Key Insights ---
st.header("8. Summary and Key Insights")
st.markdown("""
This interactive EDA confirms several key findings about the network's performance:

1.  **Primary Performance Drivers:** The correlation analysis and scatter plots consistently show that **`PCell_Downlink_Average_MCS`**, **`PCell_SNR_max`**, and **`PCell_RSRP_max`** are the most critical factors for achieving high throughput.

2.  **Carrier Aggregation Benefit:** The presence of a Secondary Cell (`has_scell` = True) consistently leads to a higher median throughput, a fact validated across the entire dataset.

3.  **Operator and Location Differences:** Significant performance variations exist between operators and across different geographical areas, as seen in the boxplots and the geospatial map. This points to real-world differences in network infrastructure and deployment quality.

4.  **Temporal Patterns:** Network performance fluctuates throughout the day, often dipping during peak usage hours (e.g., late afternoon and evening), though this varies by operator.
""")