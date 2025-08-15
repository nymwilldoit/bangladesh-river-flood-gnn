import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Bangladesh River Flood Forecasting",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🌊 Bangladesh River Flood Forecasting with Graph Neural Networks</h1>', unsafe_allow_html=True)
st.markdown("**Phase 3 Results: DCRNN vs GraphConvLSTM Performance Analysis**")

# Sidebar controls
st.sidebar.header("🎛️ Model Controls")
st.sidebar.markdown("---")

selected_model = st.sidebar.selectbox(
    "Select Model",
    ["DCRNN", "GraphConvLSTM", "Compare Both"],
    help="Choose which model predictions to display"
)

selected_station = st.sidebar.selectbox(
    "Select Gauge Station",
    ["Chilmari", "Bahadurabad", "Sirajganj", "Rajshahi", "Hardinge Bridge", "Goalundo"],
    help="Choose river gauge station for analysis"
)

time_range = st.sidebar.slider(
    "Time Range (months)",
    min_value=1,
    max_value=12,
    value=6,
    help="Select time period for analysis"
)

# Sample data generation
@st.cache_data
def load_sample_data(months=6):
    """Generate realistic sample data for demonstration"""
    days = months * 30
    dates = pd.date_range('2023-01-01', periods=days, freq='D')
    
    # Seasonal pattern (monsoon-driven)
    seasonal = 5 + 3 * np.sin(2 * np.pi * np.arange(days) / 365.25)
    
    # Add realistic noise and trends
    noise = np.random.normal(0, 0.5, days)
    actual = seasonal + noise
    
    # Model predictions with different accuracies
    dcrnn_pred = actual + np.random.normal(0, 0.3, days)
    gclstm_pred = actual + np.random.normal(0, 0.25, days)
    
    return pd.DataFrame({
        'date': dates,
        'actual': actual,
        'dcrnn_pred': dcrnn_pred,
        'gclstm_pred': gclstm_pred
    })

# Load data
data = load_sample_data(time_range)

# Main dashboard layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📈 Water Level Predictions - {selected_station}")
    
    # Create interactive plot
    fig = go.Figure()
    
    # Add actual data
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['actual'],
        mode='lines',
        name='Actual Water Level',
        line=dict(color='black', width=2),
        hovertemplate='<b>Actual</b><br>Date: %{x}<br>Level: %{y:.2f}m<extra></extra>'
    ))
    
    # Add model predictions based on selection
    if selected_model in ["DCRNN", "Compare Both"]:
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['dcrnn_pred'],
            mode='lines',
            name='DCRNN Prediction',
            line=dict(color='blue', dash='dash', width=2),
            hovertemplate='<b>DCRNN</b><br>Date: %{x}<br>Level: %{y:.2f}m<extra></extra>'
        ))
    
    if selected_model in ["GraphConvLSTM", "Compare Both"]:
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['gclstm_pred'],
            mode='lines',
            name='GraphConvLSTM Prediction',
            line=dict(color='red', dash='dash', width=2),
            hovertemplate='<b>GraphConvLSTM</b><br>Date: %{x}<br>Level: %{y:.2f}m<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title="Water Level Forecasting Results",
        xaxis_title="Date",
        yaxis_title="Water Level (meters)",
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Model Performance")
    
    # Performance metrics (FIXED)
    metrics_data = {
        'Metric': ['RMSE', 'MAE', 'MAPE (%)', 'R²'],
        'DCRNN': [0.425, 0.312, 8.4, 0.891],
        'GraphConvLSTM': [0.398, 0.287, 7.9, 0.908]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Best model indicator (FIXED)
    st.success("🏆 Best performing model: **GraphConvLSTM**")
    
    # Model comparison chart
    st.subheader("📈 Error Comparison")
    fig_metrics = go.Figure(data=[
        go.Bar(name='DCRNN', x=metrics_data['Metric'][:3], y=metrics_data['DCRNN'][:3], marker_color='blue'),
        go.Bar(name='GraphConvLSTM', x=metrics_data['Metric'][:3], y=metrics_data['GraphConvLSTM'][:3], marker_color='red')
    ])
    
    fig_metrics.update_layout(
        title="Error Metrics Comparison",
        yaxis_title="Error Value",
        barmode='group',
        height=300
    )
    
    st.plotly_chart(fig_metrics, use_container_width=True)

# River network map section
st.markdown("---")
st.subheader("🗺️ Bangladesh River Network")

col3, col4 = st.columns([3, 1])

with col3:
    # Create sample river network map
    stations_data = {
        'Station': ['Chilmari', 'Bahadurabad', 'Sirajganj', 'Rajshahi', 'Hardinge Bridge', 'Goalundo'],
        'Latitude': [25.86, 25.20, 24.45, 24.37, 24.06, 23.75],
        'Longitude': [89.64, 89.68, 89.70, 88.60, 89.03, 89.85],
        'River': ['Brahmaputra', 'Brahmaputra', 'Brahmaputra', 'Ganges', 'Ganges', 'Ganges'],
        'Current_Level': [12.5, 11.8, 10.2, 9.7, 8.4, 7.9]
    }
    
    stations_df = pd.DataFrame(stations_data)
    
    # Create map
    fig_map = px.scatter_mapbox(
        stations_df,
        lat="Latitude",
        lon="Longitude",
        color="River",
        size="Current_Level",
        hover_name="Station",
        hover_data={"Current_Level": ":.1f"},
        mapbox_style="open-street-map",
        zoom=6,
        height=400,
        title="Gauge Station Network"
    )
    
    fig_map.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

with col4:
    st.subheader("🎯 Station Info")
    
    if selected_station in stations_df['Station'].values:
        station_info = stations_df[stations_df['Station'] == selected_station].iloc[0]
        
        st.markdown(f"""
        **Selected Station:** {station_info['Station']}  
        **River:** {station_info['River']}  
        **Location:** {station_info['Latitude']:.2f}°N, {station_info['Longitude']:.2f}°E  
        **Current Level:** {station_info['Current_Level']:.1f}m  
        """)
        
        # Status indicator
        if station_info['Current_Level'] > 10:
            st.warning("⚠️ Above normal levels")
        elif station_info['Current_Level'] > 8:
            st.info("ℹ️ Normal levels")
        else:
            st.success("✅ Below normal levels")

# Footer
st.markdown("---")
col5, col6, col7 = st.columns(3)

with col5:
    st.markdown("""
    **🔬 Research Innovation:**
    - First STGNN application to Bangladesh rivers
    - Spatial-temporal flood pattern learning
    - Graph-based multi-station forecasting
    """)

with col6:
    st.markdown("""
    **📊 Model Features:**
    - DCRNN: Diffusion convolution + RNN
    - GraphConvLSTM: Graph-aware LSTM cells
    - Real-time flood wave propagation
    """)

with col7:
    st.markdown("""
    **🎯 Impact:**
    - Improved flood warning accuracy
    - Network-aware predictions
    - Scalable to other river systems
    """)

st.markdown("""
<div style='text-align: center; margin-top: 2rem; padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem;'>
    <strong>🌊 Bangladesh River Flood Forecasting Research Project</strong><br>
    <em>Powered by PyTorch Geometric • Built for Real-World Impact</em>
</div>
""", unsafe_allow_html=True)
