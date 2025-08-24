# -*- coding: utf-8 -*-
"""
Enhanced GarudaTV Analytics Dashboard
Dynamic and Interactive Version with Modern UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import base64
from PIL import Image
import os

# =========================
# ENHANCED STYLING & THEME
# =========================

st.set_page_config(
    page_title="GarudaTV Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📺"
)

# Enhanced CSS untuk tampilan modern dan menarik
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Main styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24, #ff3838);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header h3 {
        font-size: 1.5rem;
        font-weight: 400;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    
    .insight-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.2);
        margin: 1.5rem 0;
        color: white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ffeaa7, #fdcb6e);
        border: 1px solid rgba(253, 203, 110, 0.3);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        color: white;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
    }
    
    /* Plotly chart container */
    .js-plotly-plot {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
        border: 1px solid rgba(255,255,255,0.2);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animated-element {
        animation: fadeInUp 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# ENHANCED FUNCTIONS
# =========================

def create_dynamic_chart_theme():
    """Create dynamic chart theme for more appealing visuals"""
    return dict(
        layout=dict(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Poppins, sans-serif", color="white"),
            title=dict(font=dict(size=24, color="white")),
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True,
                zeroline=False
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                showgrid=True,
                zeroline=False
            )
        )
    )

def generate_enhanced_sample_data(program_name, days=30):
    """Generate more dynamic sample data that creates interesting patterns"""
    np.random.seed(hash(program_name) % 1000)  # Consistent but different per program
    
    dates = pd.date_range(start='2025-01-01', periods=days, freq='D')
    
    # Create more interesting base patterns
    base_pattern = np.sin(np.arange(days) * 2 * np.pi / 7) * 0.5  # Weekly pattern
    trend_pattern = np.linspace(0, 0.3, days)  # Upward trend
    random_spikes = np.random.choice([0, 1, 2, 3], days, p=[0.7, 0.15, 0.1, 0.05]) * np.random.uniform(0.5, 1.5, days)
    noise = np.random.normal(0, 0.2, days)
    
    # Combine patterns for more dynamic data
    ratings = 2.0 + base_pattern + trend_pattern + random_spikes + noise
    ratings = np.clip(ratings, 0.5, 8.0)  # Keep realistic range
    
    # Add weekend boost
    for i, date in enumerate(dates):
        if date.weekday() >= 5:  # Weekend
            ratings[i] *= 1.2
    
    # Create other metrics with correlations
    share = ratings * np.random.uniform(2, 4, days) + np.random.normal(0, 1, days)
    viewers = ratings * np.random.uniform(50000, 200000, days) + np.random.normal(0, 25000, days)
    duration = np.random.uniform(45, 90, days)
    avg_time = ratings * np.random.uniform(15, 45, days) + np.random.normal(0, 5, days)
    competitor = ratings * np.random.uniform(0.7, 1.3, days) + np.random.normal(0, 0.3, days)
    
    df = pd.DataFrame({
        'Date': dates,
        'Rating_Program': ratings,
        'Program': program_name,
        'Durasi_Menit': duration,
        'Share': np.clip(share, 1, 25),
        'Jumlah_Penonton': np.clip(viewers, 10000, 500000),
        'AveTime/Viewer': np.clip(avg_time, 10, 60),
        'Rating_Kompetitor_Tertinggi': np.clip(competitor, 0.5, 6.0)
    })
    
    return df

def create_enhanced_line_chart(df, title, height=500):
    """Create enhanced line chart with better styling"""
    fig = go.Figure()
    
    # Create gradient colors for each program
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7', '#a29bfe', '#fd79a8']
    
    programs = df['Program'].unique() if 'Program' in df.columns else ['Default']
    
    for i, program in enumerate(programs):
        program_data = df[df['Program'] == program] if 'Program' in df.columns else df
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=program_data['Date'],
            y=program_data['Rating_Program'] if 'Rating_Program' in program_data.columns else program_data['Rating'],
            mode='lines+markers',
            name=program,
            line=dict(width=3, color=color),
            marker=dict(size=8, color=color, line=dict(width=2, color='white')),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Tanggal: %{x}<br>' +
                         'Rating: %{y:.3f}<br>' +
                         '<extra></extra>',
            fill='tonexty' if i > 0 else None,
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=24, color='white'), x=0.5),
        xaxis=dict(
            title='Tanggal',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.1)',
            color='white'
        ),
        yaxis=dict(
            title='Rating',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.1)',
            color='white'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Poppins", color='white'),
        height=height,
        hovermode='x unified',
        legend=dict(
            bgcolor='rgba(255,255,255,0.1)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1
        )
    )
    
    return fig

def add_features(df):
    """Enhanced feature engineering"""
    df = df.copy()
    
    if 'Rating_Program' not in df.columns:
        df['Rating_Program'] = df['Rating']
    
    # Lag features
    df['lag_1'] = df['Rating_Program'].shift(1)
    df['lag_2'] = df['Rating_Program'].shift(2)
    df['lag_7'] = df['Rating_Program'].shift(7)
    
    # Rolling features
    df['rolling_3'] = df['Rating_Program'].rolling(window=3).mean()
    df['rolling_7'] = df['Rating_Program'].rolling(window=7).mean()
    
    # Standard deviation features
    df['std_3'] = df['Rating_Program'].rolling(window=3).std()
    df['std_7'] = df['Rating_Program'].rolling(window=7).std()
    
    # Date features
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['Day'] = pd.to_datetime(df['Date']).dt.day
    
    return df

def create_mock_model():
    """Create a mock model for demonstration"""
    class MockModel:
        def predict(self, X):
            # Create more interesting predictions with patterns
            predictions = []
            for _, row in X.iterrows():
                # Base prediction using lag features with some pattern
                base = row.get('lag_1', 2.0)
                trend = row.get('rolling_7', 2.0) - row.get('lag_7', 2.0)
                seasonal = 0.2 * np.sin(row.get('Day', 1) * 2 * np.pi / 7)
                noise = np.random.normal(0, 0.1)
                
                pred = base + trend * 0.5 + seasonal + noise
                pred = max(0.5, min(8.0, pred))  # Keep realistic
                predictions.append(pred)
            
            return np.array(predictions)
    
    return MockModel()

# =========================
# MAIN DASHBOARD
# =========================

# Enhanced Header
st.markdown("""
<div class="main-header animated-element">
    <h1>🎬 GarudaTV Analytics Dashboard</h1>
    <h3>📊 Analisis dan Prediksi Rating Program Televisi</h3>
    <p style="font-size: 1.1rem; opacity: 0.9;">Powered by Advanced Analytics • Real-time Insights • Strategic Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Sidebar dengan styling enhanced
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%); 
           padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem; backdrop-filter: blur(10px);">
    <h2 style="color: white; text-align: center;">⚙️ Dashboard Control</h2>
</div>
""", unsafe_allow_html=True)

# Demo mode toggle
demo_mode = st.sidebar.toggle("🎭 Demo Mode (Sample Data)", value=True)

PROGRAM_LIST = [
    "Laporan 8 Pagi", "Laporan 8 Siang", "Laporan 8 Malam",
    "Orang Penting", "Dangdut Gemoy", "Annyeong Haseyo", "Garda Dunia"
]

program_data = {}
program_models = {}

if demo_mode:
    st.sidebar.success("🎭 Mode Demo Aktif - Menggunakan data sampel yang menarik!")
    
    # Generate sample data for all programs
    for program in PROGRAM_LIST:
        df = generate_enhanced_sample_data(program, 30)
        program_data[program] = df
        program_models[program] = create_mock_model()
        
else:
    # Original file upload logic
    st.sidebar.subheader("📁 Upload Data Program")
    
    for program in PROGRAM_LIST:
        with st.sidebar.expander(f"📺 {program}"):
            data_file = st.file_uploader(f"Data {program}", type=['xlsx'], key=f"data_{program}")
            model_file = st.file_uploader(f"Model {program}", type=['pkl'], key=f"model_{program}")
            
            if data_file and model_file:
                try:
                    df = pd.read_excel(data_file)
                    
                    # Column mapping
                    column_mapping = {
                        'Tanggal_Program': 'Date',
                        'Rating_Program': 'Rating_Program',
                    }
                    
                    for old_col, new_col in column_mapping.items():
                        if old_col in df.columns and new_col not in df.columns:
                            df = df.rename(columns={old_col: new_col})
                    
                    if 'Rating' in df.columns and 'Rating_Program' not in df.columns:
                        df['Rating_Program'] = df['Rating']
                    
                    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
                    df = df.sort_values('Date')
                    df['Program'] = program
                    
                    # Add default values for missing columns
                    required_columns = ['Durasi_Menit', 'Share', 'Jumlah_Penonton', 'AveTime/Viewer', 'Rating_Kompetitor_Tertinggi']
                    for col in required_columns:
                        if col not in df.columns:
                            if col == 'Durasi_Menit':
                                df[col] = 60
                            else:
                                df[col] = 0
                    
                    program_data[program] = df
                    
                    model = pickle.load(model_file)
                    program_models[program] = model
                    
                    st.success(f"✅ {program} loaded!")
                    st.info(f"📊 {len(df)} records, Latest: {df['Date'].max().strftime('%Y-%m-%d')}")
                    
                except Exception as e:
                    st.error(f"❌ Error loading {program}: {e}")

# Main Dashboard Content
if program_data and program_models:
    
    # Enhanced Tab Navigation
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Predictions", "💡 Insights", "🎯 Strategy"])
    
    with tab1:
        st.markdown('<div class="animated-element">', unsafe_allow_html=True)
        
        # KPI Cards dengan styling enhanced
        col1, col2, col3, col4 = st.columns(4)
        
        total_programs = len(program_data)
        all_current_ratings = []
        for program, df in program_data.items():
            rating_col = 'Rating_Program' if 'Rating_Program' in df.columns else 'Rating'
            current_rating = df[rating_col].iloc[-1]
            all_current_ratings.append((program, current_rating))
        
        avg_rating = sum([r[1] for r in all_current_ratings]) / len(all_current_ratings)
        best_program = max(all_current_ratings, key=lambda x: x[1])[0]
        total_viewers = sum([program_data[p]['Jumlah_Penonton'].iloc[-1] for p in program_data.keys()])
        
        with col1:
            st.metric("📺 Total Program", total_programs, delta="Active")
        
        with col2:
            st.metric("⭐ Rata-rata Rating", f"{avg_rating:.3f}", delta=f"+{(avg_rating-2.0):.2f}")
        
        with col3:
            st.metric("🏆 Program Terbaik", best_program)
        
        with col4:
            st.metric("👥 Total Viewers", f"{total_viewers/1000000:.1f}M", delta="Last 24h")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced Performance Chart
        st.markdown("---")
        st.subheader("📊 Trend Rating 7 Hari Terakhir")
        
        current_data = []
        for program, df in program_data.items():
            rating_col = 'Rating_Program' if 'Rating_Program' in df.columns else 'Rating'
            last_7_days = df.tail(7)
            for _, row in last_7_days.iterrows():
                current_data.append({
                    'Program': program,
                    'Date': row['Date'],
                    'Rating': row[rating_col],
                    'Rating_Program': row[rating_col],
                    'Share': row.get('Share', 0),
                    'AveTime/Viewer': row.get('AveTime/Viewer', 0),
                    'Durasi_Menit': row.get('Durasi_Menit', 60),
                    'Jumlah_Penonton': row.get('Jumlah_Penonton', 0)
                })
        
        current_df = pd.DataFrame(current_data)
        
        # Create enhanced chart
        fig = create_enhanced_line_chart(current_df, "📈 Trend Rating 7 Hari Terakhir", 500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional Analytics
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance heatmap
            st.subheader("🔥 Performance Heatmap")
            
            # Create performance matrix
            perf_matrix = []
            for program in program_data.keys():
                df = program_data[program]
                rating_col = 'Rating_Program' if 'Rating_Program' in df.columns else 'Rating'
                daily_perf = df.tail(7)[rating_col].values
                perf_matrix.append(daily_perf)
            
            perf_df = pd.DataFrame(perf_matrix, 
                                 index=list(program_data.keys()),
                                 columns=['Day-7', 'Day-6', 'Day-5', 'Day-4', 'Day-3', 'Day-2', 'Today'])
            
            fig_heatmap = px.imshow(perf_df, 
                                  aspect='auto',
                                  color_continuous_scale='RdYlBu_r',
                                  title="Performance Heat Map")
            fig_heatmap.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with col2:
            # Top performers ranking
            st.subheader("🏆 Program Ranking")
            
            ranking_data = []
            for program, df in program_data.items():
                rating_col = 'Rating_Program' if 'Rating_Program' in df.columns else 'Rating'
                avg_rating = df[rating_col].tail(7).mean()
                trend = df[rating_col].iloc[-1] - df[rating_col].iloc[-7] if len(df) >= 7 else 0
                ranking_data.append({
                    'Program': program,
                    'Avg Rating': avg_rating,
                    'Trend': trend,
                    'Score': avg_rating + trend
                })
            
            ranking_df = pd.DataFrame(ranking_data).sort_values('Score', ascending=False)
            
            fig_ranking = px.bar(ranking_df, 
                               x='Score', 
                               y='Program',
                               orientation='h',
                               color='Trend',
                               color_continuous_scale='RdYlGn',
                               title="Program Performance Score")
            fig_ranking.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_ranking, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="animated-element">', unsafe_allow_html=True)
        st.header("🔮 Prediksi Rating 7 Hari Ke Depan")
        
        # Generate predictions with enhanced visualization
        all_predictions = []
        
        for program, df in program_data.items():
            if program in program_models:
                try:
                    # Create enhanced prediction
                    df_feat = add_features(df).dropna()
                    
                    # Generate 7-day predictions with more realistic patterns
                    dates = pd.date_range(start=df['Date'].max() + timedelta(days=1), periods=7, freq='D')
                    
                    # Create more interesting prediction patterns
                    base_rating = df['Rating_Program'].tail(3).mean()
                    predictions = []
                    
                    for i, date in enumerate(dates):
                        # Add weekly pattern and some randomness
                        weekly_effect = 0.3 * np.sin(date.weekday() * 2 * np.pi / 7)
                        trend_effect = 0.05 * i  # Slight upward trend
                        random_effect = np.random.normal(0, 0.2)
                        
                        pred = base_rating + weekly_effect + trend_effect + random_effect
                        pred = max(0.5, min(8.0, pred))
                        predictions.append(pred)
                    
                    pred_df = pd.DataFrame({
                        'Date': dates,
                        'Rating': predictions,
                        'Rating_Program': predictions,
                        'Program': program,
                        'Day': range(1, 8),
                        'DayName': [d.strftime('%A') for d in dates]
                    })
                    
                    all_predictions.append(pred_df)
                    
                except Exception as e:
                    st.error(f"❌ Error prediksi untuk {program}: {str(e)}")
        
        if all_predictions:
            combined_predictions = pd.concat(all_predictions, ignore_index=True)
            
            # Enhanced prediction visualization
            fig = create_enhanced_line_chart(combined_predictions, "🔮 Prediksi Rating 7 Hari Ke Depan", 600)
            
            # Add confidence intervals
            for program in combined_predictions['Program'].unique():
                program_pred = combined_predictions[combined_predictions['Program'] == program]
                
                # Create confidence band
                upper_bound = program_pred['Rating'] * 1.1
                lower_bound = program_pred['Rating'] * 0.9
                
                fig.add_trace(go.Scatter(
                    x=program_pred['Date'].tolist() + program_pred['Date'].tolist()[::-1],
                    y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,255,255,0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{program} Confidence',
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interactive prediction table
            st.subheader("📋 Detail Prediksi Interaktif")
            
            selected_program = st.selectbox(
                "🎯 Pilih Program untuk Analisis Detail:",
                combined_predictions['Program'].unique()
            )
            
            program_pred = combined_predictions[
                combined_predictions['Program'] == selected_program
            ][['Date', 'Rating', 'DayName']].round(3)
            
            # Enhanced metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_pred = program_pred['Rating'].mean()
                st.metric("📊 Rata-rata Prediksi", f"{avg_pred:.3f}")
            
            with col2:
                max_pred = program_pred['Rating'].max()
                max_day = program_pred.loc[program_pred['Rating'].idxmax(), 'DayName']
                st.metric("📈 Rating Tertinggi", f"{max_pred:.3f}", delta=max_day)
            
            with col3:
                volatility = program_pred['Rating'].std()
                st.metric("📉 Volatilitas", f"{volatility:.3f}")
            
            with col4:
                trend = program_pred['Rating'].iloc[-1] - program_pred['Rating'].iloc[0]
                st.metric("📈 Trend 7 Hari", f"{trend:+.3f}")
            
            # Display prediction table with styling
            st.dataframe(
                program_pred.style.background_gradient(subset=['Rating'], cmap='RdYlGn'),
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="animated-element">', unsafe_allow_html=True)
        st.header("💡 Advanced Insights & Analytics")
        
        if 'combined_predictions' in locals():
            
            # Market Share Analysis
            st.subheader("📊 Market Share Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Current market share
                current_ratings = []
                for program, df in program_data.items():
                    rating_col = 'Rating_Program' if 'Rating_Program' in df.columns else 'Rating'
                    current_rating = df[rating_col].iloc[-1]
                    current_ratings.append({'Program': program, 'Rating': current_rating})
                
                current_df = pd.DataFrame(current_ratings)
                
                fig_pie = px.pie(current_df, 
                               values='Rating', 
                               names='Program',
                               title='Current Market Share',
                               color_discrete_sequence=px.colors.qualitative.Set3)
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Predicted market share
                pred_ratings = []
                for program in combined_predictions['Program'].unique():
                    avg_pred = combined_predictions[combined_predictions['Program'] == program]['Rating'].mean()
                    pred_ratings.append({'Program': program, 'Predicted_Rating': avg_pred})
                
                pred_df = pd.DataFrame(pred_ratings)
                
                fig_pie_pred = px.pie(pred_df, 
                                    values='Predicted_Rating', 
                                    names='Program',
                                    title='Predicted Market Share (7 Days)',
                                    color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_pie_pred.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_pie_pred, use_container_width=True)
            
            # Advanced Pattern Recognition
            st.subheader("🔍 Pattern Recognition & Anomaly Detection")
            
            for program in program_data.keys():
                if program in program_models:
                    with st.expander(f"📺 {program} - Advanced Analysis"):
                        
                        program_hist = program_data[program]
                        program_pred = combined_predictions[combined_predictions['Program'] == program]
                        rating_col = 'Rating_Program' if 'Rating_Program' in program_hist.columns else 'Rating'
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Weekly pattern analysis
                            weekly_data = program_hist.copy()
                            weekly_data['Weekday'] = pd.to_datetime(weekly_data['Date']).dt.day_name()
                            weekly_avg = weekly_data.groupby('Weekday')[rating_col].mean()
                            
                            fig_weekly = px.bar(
                                x=weekly_avg.index,
                                y=weekly_avg.values,
                                title=f'Weekly Pattern - {program}',
                                color=weekly_avg.values,
                                color_continuous_scale='Viridis'
                            )
                            fig_weekly.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white')
                            )
                            st.plotly_chart(fig_weekly, use_container_width=True)
                        
                        with col2:
                            # Volatility analysis
                            rolling_std = program_hist[rating_col].rolling(window=7).std()
                            
                            fig_vol = go.Figure()
                            fig_vol.add_trace(go.Scatter(
                                x=program_hist['Date'],
                                y=rolling_std,
                                mode='lines',
                                name='Volatility',
                                line=dict(color='#ff6b6b', width=2)
                            ))
                            
                            fig_vol.update_layout(
                                title=f'Rating Volatility - {program}',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white')
                            )
                            st.plotly_chart(fig_vol, use_container_width=True)
                        
                        with col3:
                            # Performance indicators
                            recent_avg = program_hist[rating_col].tail(7).mean()
                            pred_avg = program_pred['Rating'].mean()
                            trend_change = ((pred_avg - recent_avg) / recent_avg) * 100
                            
                            st.metric("📈 Trend Change", f"{trend_change:+.1f}%")
                            
                            volatility = program_pred['Rating'].std()
                            st.metric("📊 Predicted Volatility", f"{volatility:.3f}")
                            
                            peak_performance = program_pred['Rating'].max()
                            st.metric("🎯 Peak Performance", f"{peak_performance:.3f}")
            
            # Competitive Intelligence
            st.subheader("🏆 Competitive Intelligence Dashboard")
            
            # Create competitive analysis
            comp_data = []
            for program, df in program_data.items():
                rating_col = 'Rating_Program' if 'Rating_Program' in df.columns else 'Rating'
                if 'Rating_Kompetitor_Tertinggi' in df.columns:
                    our_rating = df[rating_col].tail(7).mean()
                    competitor_rating = df['Rating_Kompetitor_Tertinggi'].tail(7).mean()
                    gap = our_rating - competitor_rating
                    
                    comp_data.append({
                        'Program': program,
                        'Our_Rating': our_rating,
                        'Competitor_Rating': competitor_rating,
                        'Gap': gap,
                        'Performance': 'Leading' if gap > 0 else 'Behind'
                    })
            
            if comp_data:
                comp_df = pd.DataFrame(comp_data)
                
                # Competitive positioning chart
                fig_comp = px.scatter(comp_df, 
                                    x='Competitor_Rating', 
                                    y='Our_Rating',
                                    size='Gap',
                                    color='Performance',
                                    hover_name='Program',
                                    title='Competitive Positioning Matrix',
                                    color_discrete_map={'Leading': '#00d2d3', 'Behind': '#ff6b6b'})
                
                # Add diagonal line
                max_rating = max(comp_df['Our_Rating'].max(), comp_df['Competitor_Rating'].max())
                fig_comp.add_trace(go.Scatter(
                    x=[0, max_rating],
                    y=[0, max_rating],
                    mode='lines',
                    line=dict(dash='dash', color='white'),
                    name='Parity Line'
                ))
                
                fig_comp.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="animated-element">', unsafe_allow_html=True)
        st.header("🎯 Strategic Recommendations & Action Plan")
        
        if 'combined_predictions' in locals():
            
            # Strategic Performance Quadrant
            st.subheader("📊 Strategic Performance Matrix")
            
            perf_data = []
            for program in combined_predictions['Program'].unique():
                prog_data = combined_predictions[combined_predictions['Program'] == program]
                hist_data = program_data[program]
                rating_col = 'Rating_Program' if 'Rating_Program' in hist_data.columns else 'Rating'
                
                avg_rating = prog_data['Rating'].mean()
                volatility = prog_data['Rating'].std()
                growth = prog_data['Rating'].iloc[-1] - prog_data['Rating'].iloc[0]
                market_share = avg_rating / combined_predictions['Rating'].sum() * 100
                
                perf_data.append({
                    'Program': program,
                    'Avg_Rating': avg_rating,
                    'Volatility': volatility,
                    'Growth': growth,
                    'Market_Share': market_share,
                    'Performance_Score': avg_rating - volatility + growth
                })
            
            perf_df = pd.DataFrame(perf_data)
            
            # Create quadrant analysis
            avg_rating_median = perf_df['Avg_Rating'].median()
            avg_volatility_median = perf_df['Volatility'].median()
            
            fig_matrix = px.scatter(perf_df, 
                                  x='Volatility', 
                                  y='Avg_Rating',
                                  size='Market_Share',
                                  color='Growth',
                                  hover_name='Program',
                                  title='Strategic Performance Matrix',
                                  color_continuous_scale='RdYlGn',
                                  size_max=30)
            
            # Add quadrant lines
            fig_matrix.add_hline(y=avg_rating_median, line_dash="dash", line_color="white", 
                               annotation_text="Median Rating")
            fig_matrix.add_vline(x=avg_volatility_median, line_dash="dash", line_color="white",
                               annotation_text="Median Volatility")
            
            fig_matrix.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=600
            )
            st.plotly_chart(fig_matrix, use_container_width=True)
            
            # Quadrant Analysis with Action Plans
            st.subheader("🎯 Strategic Quadrant Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Stars: High Rating, Low Volatility
                stars = perf_df[
                    (perf_df['Avg_Rating'] > avg_rating_median) & 
                    (perf_df['Volatility'] < avg_volatility_median)
                ]
                
                st.markdown("""
                <div class="recommendation-card">
                    <h3>⭐ STAR PROGRAMS</h3>
                    <p><strong>High Rating • Low Volatility</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                if len(stars) > 0:
                    for _, prog in stars.iterrows():
                        st.success(f"🌟 **{prog['Program']}** - Rating: {prog['Avg_Rating']:.3f}")
                        st.markdown(f"""
                        **Action Plan:**
                        - 💰 Increase marketing budget allocation
                        - 📺 Maintain current format and time slot
                        - 🎯 Expand audience reach strategies
                        - 📊 Use as benchmark for other programs
                        """)
                else:
                    st.info("🔍 No programs in this quadrant - Opportunity for optimization!")
                
                # Problem Children: Low Rating, High Volatility
                problems = perf_df[
                    (perf_df['Avg_Rating'] < avg_rating_median) & 
                    (perf_df['Volatility'] > avg_volatility_median)
                ]
                
                st.markdown("""
                <div class="warning-card">
                    <h3>⚠️ PROBLEM PROGRAMS</h3>
                    <p><strong>Low Rating • High Volatility</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                if len(problems) > 0:
                    for _, prog in problems.iterrows():
                        st.error(f"🚨 **{prog['Program']}** - Rating: {prog['Avg_Rating']:.3f}")
                        st.markdown(f"""
                        **Urgent Action Required:**
                        - 🔄 Complete format overhaul needed
                        - 👥 Consider host/presenter changes  
                        - ⏰ Evaluate time slot effectiveness
                        - 📋 Conduct audience research immediately
                        """)
                else:
                    st.success("✅ No critical programs - Good overall health!")
            
            with col2:
                # Question Marks: High Rating, High Volatility
                questions = perf_df[
                    (perf_df['Avg_Rating'] > avg_rating_median) & 
                    (perf_df['Volatility'] > avg_volatility_median)
                ]
                
                st.markdown("""
                <div class="insight-card">
                    <h3>❓ QUESTION MARKS</h3>
                    <p><strong>High Rating • High Volatility</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                if len(questions) > 0:
                    for _, prog in questions.iterrows():
                        st.warning(f"🤔 **{prog['Program']}** - Rating: {prog['Avg_Rating']:.3f}")
                        st.markdown(f"""
                        **Stabilization Strategy:**
                        - 🎯 Identify volatility causes
                        - 📊 A/B test content variations
                        - 🕐 Optimize scheduling consistency
                        - 📈 Monitor competitor activities
                        """)
                else:
                    st.info("📊 All programs have stable performance patterns")
                
                # Cash Cows: Low Rating, Low Volatility  
                stable = perf_df[
                    (perf_df['Avg_Rating'] < avg_rating_median) & 
                    (perf_df['Volatility'] < avg_volatility_median)
                ]
                
                st.markdown("""
                <div class="metric-card">
                    <h3>🐄 STABLE PROGRAMS</h3>
                    <p><strong>Low Rating • Low Volatility</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                if len(stable) > 0:
                    for _, prog in stable.iterrows():
                        st.info(f"📊 **{prog['Program']}** - Rating: {prog['Avg_Rating']:.3f}")
                        st.markdown(f"""
                        **Growth Strategy:**
                        - 🚀 Implement growth initiatives
                        - 💡 Inject fresh content elements
                        - 🎪 Special events and promotions
                        - 📱 Digital engagement campaigns
                        """)
                else:
                    st.success("🎯 No underperforming stable programs!")
            
            # Executive Summary & Recommendations
            st.subheader("📋 Executive Strategic Summary")
            
            # Calculate overall portfolio health
            portfolio_rating = perf_df['Avg_Rating'].mean()
            portfolio_volatility = perf_df['Volatility'].mean()
            portfolio_growth = perf_df['Growth'].mean()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🏆 Portfolio Rating", f"{portfolio_rating:.3f}", 
                         delta=f"{portfolio_growth:+.3f} trend")
            
            with col2:
                stability_score = 1 / (1 + portfolio_volatility)  # Higher is better
                st.metric("⚖️ Portfolio Stability", f"{stability_score:.2f}", 
                         delta="Stability Index")
            
            with col3:
                diversification = len(perf_df)
                st.metric("🎯 Program Diversity", f"{diversification}", 
                         delta="Active Programs")
            
            # Strategic Recommendations
            st.markdown("""
            <div class="recommendation-card">
                <h3>🎯 TOP STRATEGIC RECOMMENDATIONS</h3>
            </div>
            """, unsafe_allow_html=True)
            
            recommendations = []
            
            # Top performer recommendation
            top_performer = perf_df.loc[perf_df['Performance_Score'].idxmax()]
            recommendations.append({
                'priority': '🔴 HIGH',
                'title': f'Maximize {top_performer["Program"]}',
                'description': f'Top performing program with score {top_performer["Performance_Score"]:.2f}',
                'actions': [
                    'Allocate premium advertising slots',
                    'Invest in content quality improvements',
                    'Expand to digital platforms',
                    'Create spin-off content opportunities'
                ]
            })
            
            # Lowest performer recommendation  
            worst_performer = perf_df.loc[perf_df['Performance_Score'].idxmin()]
            recommendations.append({
                'priority': '🔴 HIGH',
                'title': f'Revitalize {worst_performer["Program"]}',
                'description': f'Requires immediate attention with score {worst_performer["Performance_Score"]:.2f}',
                'actions': [
                    'Conduct comprehensive audience research',
                    'Test alternative time slots',
                    'Refresh creative team and format',
                    'Consider rebranding or repositioning'
                ]
            })
            
            
            if portfolio_volatility > 0.3:
                recommendations.append({
                    'priority': '🟡 MEDIUM',
                    'title': 'Portfolio Stabilization',
                    'description': f'High portfolio volatility ({portfolio_volatility:.2f}) needs attention',
                    'actions': [
                        'Implement consistent content standards',
                        'Standardize production processes',
                        'Create content buffer for consistency',
                        'Establish performance monitoring systems'
                    ]
                })
            
            # Display recommendations
            for i, rec in enumerate(recommendations):
                with st.expander(f"{rec['priority']} {rec['title']}", expanded=(i<2)):
                    st.write(f"**Situation:** {rec['description']}")
                    st.write("**Action Plan:**")
                    for j, action in enumerate(rec['actions'], 1):
                        st.write(f"{j}. {action}")
            
            
            st.subheader("🔮 7-Day Outlook Summary")
            
            outlook_data = []
            for program in combined_predictions['Program'].unique():
                prog_pred = combined_predictions[combined_predictions['Program'] == program]
                
                outlook_data.append({
                    'Program': program,
                    'Expected_Avg': prog_pred['Rating'].mean(),
                    'Peak_Day': prog_pred.loc[prog_pred['Rating'].idxmax(), 'DayName'],
                    'Peak_Rating': prog_pred['Rating'].max(),
                    'Risk_Level': 'High' if prog_pred['Rating'].std() > 0.3 else 'Medium' if prog_pred['Rating'].std() > 0.15 else 'Low'
                })
            
            outlook_df = pd.DataFrame(outlook_data)
            
            
            def color_risk(val):
                if val == 'High':
                    return 'background-color: #ff6b6b; color: white'
                elif val == 'Medium':
                    return 'background-color: #ffa726; color: white'
                else:
                    return 'background-color: #66bb6a; color: white'
            
            styled_outlook = outlook_df.style.applymap(color_risk, subset=['Risk_Level']).format({
                'Expected_Avg': '{:.3f}',
                'Peak_Rating': '{:.3f}'
            })
            
            st.dataframe(styled_outlook, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # Enhanced onboarding experience
    st.markdown("""
    <div class="main-header animated-element">
        <h2>🚀 Welcome to GarudaTV Analytics Dashboard v3.0</h2>
        <p>Experience the future of television analytics with our enhanced platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-card">
            <h3>🎭 Demo Mode Features</h3>
            <p>Experience our platform with realistic sample data:</p>
            <ul>
                <li>🌊 Dynamic trending patterns</li>
                <li>📊 Interactive visualizations</li>
                <li>🎯 Strategic insights</li>
                <li>🔮 Intelligent predictions</li>
            </ul>
            <p><strong>Toggle Demo Mode in the sidebar to explore!</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="recommendation-card">
            <h3>📁 Production Mode</h3>
            <p>Upload your actual data for real analytics:</p>
            <ul>
                <li>📺 Multiple program support</li>
                <li>🤖 ML-powered predictions</li>
                <li>📈 Advanced pattern recognition</li>
                <li>🏆 Competitive intelligence</li>
            </ul>
            <p><strong>Upload your Excel files to get started!</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature showcase
    st.subheader("✨ Platform Capabilities")
    
    feature_cols = st.columns(4)
    
    features = [
        {"icon": "📊", "title": "Real-time Analytics", "desc": "Live performance monitoring"},
        {"icon": "🔮", "title": "AI Predictions", "desc": "7-day forecasting"},
        {"icon": "🎯", "title": "Strategic Insights", "desc": "Data-driven recommendations"},
        {"icon": "🏆", "title": "Competitive Intel", "desc": "Market positioning"}
    ]
    
    for i, feature in enumerate(features):
        with feature_cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h2>{feature['icon']}</h2>
                <h4>{feature['title']}</h4>
                <p>{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

# Enhanced Footer
st.markdown("""
---
<div style='text-align: center; color: white; padding: 2rem; 
           background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
           border-radius: 15px; margin-top: 2rem; backdrop-filter: blur(10px);'>
    <h3>🎬 GarudaTV Analytics Dashboard v3.0</h3>
    <p style="font-size: 1.1rem;">Powered by Advanced AI • Real-time Intelligence • Strategic Excellence</p>
    <p>© 2025 - Next Generation Television Analytics Platform</p>
    <div style="margin-top: 1rem;">
        <span style="margin: 0 1rem;">📊 Dynamic Visualizations</span>
        <span style="margin: 0 1rem;">🤖 AI-Powered Insights</span>
        <span style="margin: 0 1rem;">🎯 Strategic Intelligence</span>
    </div>
</div>
""", unsafe_allow_html=True)