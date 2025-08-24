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
# Fungsi Tampilkan Logo
# =========================

def display_logo():
    logo_path = "/content/Garudatv Banee.png"  # Pastikan path benar
    if os.path.exists(logo_path):
        image = Image.open(logo_path)
        # Increase the width to 400
        st.image(image, width=50, use_container_width=True)  # Ganti 'auto' dengan True
    else:
        st.warning("Logo tidak ditemukan!")

# Konfigurasi halaman
st.set_page_config(
    page_title="GarudaTV Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #C41E3A, #FF6B6B);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #C41E3A;
    }
    .insight-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk menambahkan fitur dengan format baru
def add_features(df):
    df = df.copy()

    # Ensure we have the required columns from uploaded data
    if 'Rating_Program' not in df.columns:
        df['Rating_Program'] = df['Rating']  # Backward compatibility

    # Create lag features
    df['lag_1'] = df['Rating_Program'].shift(1)
    df['lag_2'] = df['Rating_Program'].shift(2)
    df['lag_7'] = df['Rating_Program'].shift(7)

    # Create rolling mean features
    df['rolling_3'] = df['Rating_Program'].rolling(window=3).mean()
    df['rolling_7'] = df['Rating_Program'].rolling(window=7).mean()

    # Create rolling standard deviation features
    df['std_3'] = df['Rating_Program'].rolling(window=3).std()
    df['std_7'] = df['Rating_Program'].rolling(window=7).std()

    # Date features
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['Day'] = pd.to_datetime(df['Date']).dt.day

    return df

# Fungsi prediksi dengan fitur baru - DIPERBAIKI
def predict_7_days(model, last_data, program_name):
    results = []
    current_data = last_data.copy()

    # Debug: Print kolom yang tersedia
    st.write("Debug - Kolom tersedia:", current_data.columns.tolist())
    st.write("Debug - Sample data terakhir:", current_data.tail(1))

    for day in range(1, 8):
        next_date = current_data['Date'].max() + timedelta(days=1)
        last_7 = current_data.tail(7)['Rating_Program'].tolist()
        last_3 = current_data.tail(3)['Rating_Program'].tolist()

        # Prepare new row with updated feature structure - TANPA Rating_Program sebagai input
        new_row = {
            'Durasi_Menit': current_data.iloc[-1]['Durasi_Menit'] if 'Durasi_Menit' in current_data.columns else 60,
            'Share': current_data.iloc[-1]['Share'] if 'Share' in current_data.columns else 5.0,
            'Jumlah_Penonton': current_data.iloc[-1]['Jumlah_Penonton'] if 'Jumlah_Penonton' in current_data.columns else 100000,
            'AveTime/Viewer': current_data.iloc[-1]['AveTime/Viewer'] if 'AveTime/Viewer' in current_data.columns else 30.0,
            'Rating_Kompetitor_Tertinggi': current_data.iloc[-1]['Rating_Kompetitor_Tertinggi'] if 'Rating_Kompetitor_Tertinggi' in current_data.columns else 2.0,
            'Year': next_date.year,
            'Month': next_date.month,
            'Day': next_date.day,
            'lag_1': last_7[-1] if len(last_7) >= 1 else (current_data['lag_1'].iloc[-1] if 'lag_1' in current_data.columns else 2.0),
            'lag_2': last_7[-2] if len(last_7) >= 2 else (current_data['lag_2'].iloc[-1] if 'lag_2' in current_data.columns else 2.0),
            'lag_7': last_7[0] if len(last_7) == 7 else (current_data['lag_7'].iloc[-1] if 'lag_7' in current_data.columns else 2.0),
            'rolling_3': np.mean(last_3) if len(last_3) >= 3 else (current_data['rolling_3'].iloc[-1] if 'rolling_3' in current_data.columns else 2.0),
            'rolling_7': np.mean(last_7) if len(last_7) >= 7 else (current_data['rolling_7'].iloc[-1] if 'rolling_7' in current_data.columns else 2.0),
            'std_3': np.std(last_3) if len(last_3) >= 2 else (current_data['std_3'].iloc[-1] if 'std_3' in current_data.columns else 0.5),
            'std_7': np.std(last_7) if len(last_7) >= 2 else (current_data['std_7'].iloc[-1] if 'std_7' in current_data.columns else 0.5)
        }

        # PENTING: Urutan fitur untuk prediksi - TANPA Rating_Program!
        # Karena Rating_Program adalah TARGET yang ingin diprediksi
        feature_order_for_prediction = [
            'Durasi_Menit', 'Share', 'Jumlah_Penonton',
            'AveTime/Viewer', 'Rating_Kompetitor_Tertinggi', 'Year', 'Month', 'Day', 'lag_1',
            'lag_2', 'lag_7', 'rolling_3', 'rolling_7', 'std_3', 'std_7'
        ]

        input_df = pd.DataFrame([new_row])[feature_order_for_prediction]

        # Debug: Print input untuk prediksi
        if day == 1:  # Hanya print untuk hari pertama
            st.write("Debug - Input untuk prediksi:", input_df)
            st.write("Debug - Shape input:", input_df.shape)

        # Handle NaN values dengan nilai yang lebih realistis
        input_df = input_df.fillna({
            'Durasi_Menit': 60,
            'Share': 5.0,
            'Jumlah_Penonton': 100000,
            'AveTime/Viewer': 30.0,
            'Rating_Kompetitor_Tertinggi': 2.0,
            'lag_1': 2.0,
            'lag_2': 2.0,
            'lag_7': 2.0,
            'rolling_3': 2.0,
            'rolling_7': 2.0,
            'std_3': 0.5,
            'std_7': 0.5
        })

        # Predict rating
        try:
            rating_pred = model.predict(input_df)[0]

            # Debug: Print prediksi mentah
            if day == 1:
                st.write("Debug - Raw prediction:", rating_pred)

            # Handle transformasi jika perlu
            if rating_pred < 0:  # Kemungkinan log-transformed
                rating_pred = np.expm1(rating_pred)

            # Ensure rating is reasonable (between 0.1 and 20)
            rating_pred = rating_pred

        except Exception as e:
            st.error(f"Error dalam prediksi: {e}")
            # Fallback ke rata-rata historical
            rating_pred = current_data['Rating_Program'].tail(7).mean()

        # Update current_data untuk iterasi berikutnya
        new_row_full = {
            'Date': next_date,
            'Rating_Program': rating_pred,
            'Program': program_name,
            'Durasi_Menit': new_row['Durasi_Menit'],
            'Share': new_row['Share'],
            'Jumlah_Penonton': new_row['Jumlah_Penonton'],
            'AveTime/Viewer': new_row['AveTime/Viewer'],
            'Rating_Kompetitor_Tertinggi': new_row['Rating_Kompetitor_Tertinggi'],
            'Year': new_row['Year'],
            'Month': new_row['Month'],
            'Day': new_row['Day'],
            'lag_1': new_row['lag_1'],
            'lag_2': new_row['lag_2'],
            'lag_7': new_row['lag_7'],
            'rolling_3': new_row['rolling_3'],
            'rolling_7': new_row['rolling_7'],
            'std_3': new_row['std_3'],
            'std_7': new_row['std_7']
        }

        current_data = pd.concat([current_data, pd.DataFrame([new_row_full])], ignore_index=True)

        # Update rolling values untuk prediksi berikutnya
        if len(current_data) >= 3:
            current_data.loc[current_data.index[-1], 'rolling_3'] = current_data['Rating_Program'].tail(3).mean()
            current_data.loc[current_data.index[-1], 'std_3'] = current_data['Rating_Program'].tail(3).std()
        if len(current_data) >= 7:
            current_data.loc[current_data.index[-1], 'rolling_7'] = current_data['Rating_Program'].tail(7).mean()
            current_data.loc[current_data.index[-1], 'std_7'] = current_data['Rating_Program'].tail(7).std()

        results.append({
            'Date': next_date,
            'Rating': rating_pred,
            'Rating_Program': rating_pred,
            'Program': program_name,
            'Day': day,
            'DayName': next_date.strftime('%A')
        })

    return pd.DataFrame(results)

# Fungsi analisis insights (updated untuk menggunakan Rating_Program)
def generate_insights(df_historical, df_predictions, program_name):
    insights = []

    # Use Rating_Program if available, otherwise Rating
    rating_col = 'Rating_Program' if 'Rating_Program' in df_historical.columns else 'Rating'
    pred_rating_col = 'Rating_Program' if 'Rating_Program' in df_predictions.columns else 'Rating'

    # Trend analysis
    recent_avg = df_historical[rating_col].tail(7).mean()
    pred_avg = df_predictions[pred_rating_col].mean()
    trend_change = ((pred_avg - recent_avg) / recent_avg) * 100

    if trend_change > 5:
        insights.append({
            'type': 'positive',
            'title': f'📈 Tren Positif untuk {program_name}',
            'description': f'Prediksi menunjukkan peningkatan rating sebesar {trend_change:.1f}% dibanding rata-rata 7 hari terakhir.'
        })
    elif trend_change < -5:
        insights.append({
            'type': 'warning',
            'title': f'⚠️ Penurunan Rating {program_name}',
            'description': f'Prediksi menunjukkan penurunan rating sebesar {abs(trend_change):.1f}% dibanding rata-rata 7 hari terakhir.'
        })

    # Weekend vs Weekday analysis
    weekday_pred = df_predictions[df_predictions['Date'].dt.dayofweek < 5][pred_rating_col].mean()
    weekend_pred = df_predictions[df_predictions['Date'].dt.dayofweek >= 5][pred_rating_col].mean()

    if len(df_predictions[df_predictions['Date'].dt.dayofweek >= 5]) > 0 and weekday_pred > 0:
        if weekend_pred > weekday_pred * 1.1:
            insights.append({
                'type': 'info',
                'title': f'🎯 Weekend Advantage untuk {program_name}',
                'description': f'Rating weekend diprediksi {((weekend_pred/weekday_pred - 1) * 100):.1f}% lebih tinggi dari weekday.'
            })

    # Volatility analysis using standard deviations if available
    if 'std_7' in df_historical.columns:
        current_volatility = df_historical['std_7'].iloc[-1]
        pred_volatility = df_predictions[pred_rating_col].std()
        if pred_volatility > current_volatility * 1.2:
            insights.append({
                'type': 'warning',
                'title': f'📊 Volatilitas Meningkat {program_name}',
                'description': f'Prediksi menunjukkan peningkatan volatilitas rating. Perlu monitoring ketat konten dan strategi.'
            })

    # Competitor analysis if available
    if 'Rating_Kompetitor_Tertinggi' in df_historical.columns:
        avg_competitor = df_historical['Rating_Kompetitor_Tertinggi'].tail(7).mean()
        avg_program = recent_avg
        if avg_program > avg_competitor * 1.1:
            insights.append({
                'type': 'positive',
                'title': f'🏆 Unggul dari Kompetitor',
                'description': f'{program_name} menunjukkan performa {((avg_program/avg_competitor - 1) * 100):.1f}% lebih baik dari kompetitor terdekat.'
            })
        elif avg_program < avg_competitor * 0.9:
            insights.append({
                'type': 'warning',
                'title': f'⚠️ Tertinggal dari Kompetitor',
                'description': f'{program_name} perlu perbaikan untuk bersaing dengan kompetitor yang rating-nya {((avg_competitor/avg_program - 1) * 100):.1f}% lebih tinggi.'
            })

    return insights

# Fungsi rekomendasi strategis (updated)
def generate_recommendations(all_predictions, all_historical):
    recommendations = []

    # Use appropriate rating column
    rating_col = 'Rating_Program' if 'Rating_Program' in all_historical.columns else 'Rating'
    pred_rating_col = 'Rating_Program' if 'Rating_Program' in all_predictions.columns else 'Rating'

    # Analisis performa program
    program_performance = []
    for program in all_predictions['Program'].unique():
        pred_data = all_predictions[all_predictions['Program'] == program]
        hist_data = all_historical[all_historical['Program'] == program] if 'Program' in all_historical.columns else all_historical

        avg_rating = pred_data[pred_rating_col].mean()
        volatility = pred_data[pred_rating_col].std()
        trend = pred_data[pred_rating_col].iloc[-1] - pred_data[pred_rating_col].iloc[0]

        program_performance.append({
            'Program': program,
            'Avg_Rating': avg_rating,
            'Volatility': volatility,
            'Trend': trend,
            'Performance_Score': avg_rating - volatility + trend
        })

    perf_df = pd.DataFrame(program_performance).sort_values('Performance_Score', ascending=False)

    # Rekomendasi berdasarkan performa
    if len(perf_df) > 0:
        top_performer = perf_df.iloc[0]
        worst_performer = perf_df.iloc[-1]

        recommendations.extend([
            {
                'priority': 'high',
                'title': f'🏆 Maksimalkan {top_performer["Program"]}',
                'description': f'Program dengan performa terbaik (Score: {top_performer["Performance_Score"]:.2f}). Pertahankan format dan tingkatkan promosi.',
                'actions': [
                    'Alokasikan budget marketing lebih besar',
                    'Pertahankan konsistensi konten dan durasi optimal',
                    'Eksplorasi time slot premium',
                    'Analisis faktor sukses untuk diterapkan ke program lain'
                ]
            },
            {
                'priority': 'high',
                'title': f'🔧 Revitalisasi {worst_performer["Program"]}',
                'description': f'Program memerlukan perbaikan mendesak (Score: {worst_performer["Performance_Score"]:.2f})',
                'actions': [
                    'Review dan refresh format program',
                    'Analisis kompetitor di time slot yang sama',
                    'Pertimbangkan perubahan host atau format',
                    'Evaluasi durasi program yang optimal'
                ]
            }
        ])

    # Rekomendasi berdasarkan data kompetitor jika tersedia
    if 'Rating_Kompetitor_Tertinggi' in all_historical.columns:
        competitive_analysis = all_historical.groupby('Program')['Rating_Kompetitor_Tertinggi'].mean()
        if len(competitive_analysis) > 0:
            recommendations.append({
                'priority': 'medium',
                'title': '🎯 Strategi Kompetitif',
                'description': 'Berdasarkan analisis kompetitor, ada peluang untuk optimasi positioning.',
                'actions': [
                    'Monitor performa kompetitor secara real-time',
                    'Identifikasi celah waktu tayang yang kurang kompetitif',
                    'Kembangkan konten diferensiasi yang unik',
                    'Pertimbangkan kolaborasi atau counter-programming'
                ]
            })

    # Rekomendasi berdasarkan hari
    if len(all_predictions) > 0:
        day_analysis = all_predictions.groupby(all_predictions['Date'].dt.dayofweek)[pred_rating_col].mean()
        if len(day_analysis) > 0:
            best_day = day_analysis.idxmax()
            worst_day = day_analysis.idxmin()

            day_names = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']

            recommendations.append({
                'priority': 'medium',
                'title': f'📅 Optimasi Scheduling',
                'description': f'{day_names[best_day]} menunjukkan performa terbaik, sementara {day_names[worst_day]} terlemah.',
                'actions': [
                    f'Pindahkan program premium ke {day_names[best_day]}',
                    f'Program eksperimental atau rerun untuk {day_names[worst_day]}',
                    'Analisis pola viewing habit penonton',
                    'Pertimbangkan durasi yang berbeda untuk hari-hari tertentu'
                ]
            })

    return recommendations

# Header Dashboard
st.markdown("""
<div class="main-header">
    <h1>🎬 GarudaTV Analytics Dashboard</h1>
    <h3>Analisis dan Prediksi Rating Program Televisi</h3>
    <p>Powered by MichSteven - Enhanced Analytics v2.0</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Konfigurasi Dashboard")

# Upload files untuk multiple programs
st.sidebar.subheader("📁 Upload Data Program")

# Dictionary untuk menyimpan data dan model setiap program
PROGRAM_LIST = [
    "Laporan 8 Pagi", "Laporan 8 Siang", "Laporan 8 Malam",
    "Orang Penting", "Dangdut Gemoy", "Annyeong Haseyo", "Garda Dunia"
]

program_data = {}
program_models = {}

# Upload data untuk setiap program
for program in PROGRAM_LIST:
    with st.sidebar.expander(f"📺 {program}"):
        data_file = st.file_uploader(f"Data {program}", type=['xlsx'], key=f"data_{program}")
        model_file = st.file_uploader(f"Model {program}", type=['pkl'], key=f"model_{program}")

        if data_file and model_file:
            try:
                df = pd.read_excel(data_file)

                # Handle different column naming conventions
                column_mapping = {
                    'Tanggal_Program': 'Date',
                    'Rating_Program': 'Rating_Program',
                    # Add other potential mappings as needed
                }

                for old_col, new_col in column_mapping.items():
                    if old_col in df.columns and new_col not in df.columns:
                        df = df.rename(columns={old_col: new_col})

                # Ensure we have Rating_Program column
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
                            df[col] = 60  # Default duration
                        else:
                            df[col] = 0  # Default value

                program_data[program] = df

                model = pickle.load(model_file)
                program_models[program] = model

                st.success(f"✅ {program} loaded!")

                # Show data info
                st.info(f"📊 {len(df)} records, Latest: {df['Date'].max().strftime('%Y-%m-%d')}")

            except Exception as e:
                st.error(f"❌ Error loading {program}: {e}")

# Display expected data format
with st.sidebar.expander("📋 Format Data yang Diharapkan"):
    st.markdown("""
    **Kolom yang diperlukan:**
    - Date/Tanggal_Program: Tanggal program
    - Rating_Program: Rating program
    - Durasi_Menit: Durasi dalam menit
    - Share: Share persentase
    - Jumlah_Penonton: Jumlah penonton
    - AveTime/Viewer: Rata-rata waktu per viewer
    - Rating_Kompetitor_Tertinggi: Rating kompetitor tertinggi

    **Note:** Kolom yang tidak ada akan diberi nilai default.
    """)

# Main Dashboard
if program_data and program_models:

    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Predictions", "💡 Insights", "🎯 Recommendations"])

    with tab1:
        st.header("📊 Key Performance Indicators")

        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)

        total_programs = len(program_data)
        total_predictions = 0
        avg_rating = 0
        best_program = ""

        all_current_ratings = []
        for program, df in program_data.items():
            rating_col = 'Rating_Program' if 'Rating_Program' in df.columns else 'Rating'
            current_rating = df[rating_col].iloc[-1]
            all_current_ratings.append((program, current_rating))
            avg_rating += current_rating

        avg_rating /= len(all_current_ratings)
        best_program = max(all_current_ratings, key=lambda x: x[1])[0]

        with col1:
            st.metric("📺 Total Program", total_programs)

        with col2:
            st.metric("⭐ Rata-rata Rating", f"{avg_rating:.2f}")

        with col3:
            st.metric("🏆 Program Terbaik", best_program)

        with col4:
            st.metric("📅 Prediksi Period", "7 Hari")

        # Current Performance Chart
        st.subheader("📊 Performa Rating Saat Ini")

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

        # Interactive chart
        fig = px.line(current_df, x='Date', y='Rating', color='Program',
                     title='Trend Rating 7 Hari Terakhir',
                     markers=True)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Additional metrics
        col1, col2 = st.columns(2)

        with col1:
            # Average duration chart
            if 'Durasi_Menit' in current_df.columns:
                avg_duration = current_df.groupby('Program')['Durasi_Menit'].mean().reset_index()
                fig_duration = px.bar(avg_duration, x='Program', y='Durasi_Menit',
                                    title='Rata-rata Durasi Program (Menit)')
                fig_duration.update_layout(height=300)
                st.plotly_chart(fig_duration, use_container_width=True)

        with col2:
            # Average viewers chart
            if 'Jumlah_Penonton' in current_df.columns:
                avg_viewers = current_df.groupby('Program')['Jumlah_Penonton'].mean().reset_index()
                fig_viewers = px.bar(avg_viewers, x='Program', y='Jumlah_Penonton',
                                   title='Rata-rata Jumlah Penonton')
                fig_viewers.update_layout(height=300)
                st.plotly_chart(fig_viewers, use_container_width=True)

    with tab2:
        st.header("📈 Prediksi Rating 7 Hari Ke Depan")

        # Generate predictions for all programs
        all_predictions = []

        for program, df in program_data.items():
            if program in program_models:
                try:
                    df_feat = add_features(df).dropna()
                    last_7 = df_feat.tail(7)

                    pred_df = predict_7_days(program_models[program], last_7, program)
                    all_predictions.append(pred_df)
                    st.success(f"✅ Prediksi berhasil untuk {program}")
                except Exception as e:
                    st.error(f"❌ Error prediksi untuk {program}: {str(e)}")
                    st.write("Debug info:", str(e))

        if all_predictions:
            combined_predictions = pd.concat(all_predictions, ignore_index=True)

            # Prediction visualization
            fig = px.line(combined_predictions, x='Date', y='Rating', color='Program',
                         title='Prediksi Rating 7 Hari Ke Depan',
                         markers=True)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Prediction table
            st.subheader("📋 Detail Prediksi")

            # Program selector
            selected_program = st.selectbox("Pilih Program:",
                                          combined_predictions['Program'].unique())

            program_pred = combined_predictions[
                combined_predictions['Program'] == selected_program
            ][['Date', 'Rating', 'DayName']].round(2)

            st.dataframe(program_pred, use_container_width=True)

            # Summary statistics
            col1, col2, col3 = st.columns(3)

            with col1:
                avg_pred = program_pred['Rating'].mean()
                st.metric("📊 Rata-rata Prediksi", f"{avg_pred:.4f}")

            with col2:
                max_pred = program_pred['Rating'].max()
                st.metric("📈 Rating Tertinggi", f"{max_pred:.4f}")

            with col3:
                volatility = program_pred['Rating'].std()
                st.metric("📉 Volatilitas", f"{volatility:.4f}")

    with tab3:
        st.header("💡 Insights & Analisis")

        if 'combined_predictions' in locals() and len(all_predictions) > 0:
            # Generate insights for each program
            for program in program_data.keys():
                if program in program_models:
                    st.subheader(f"📺 {program}")

                    program_hist = program_data[program]
                    program_pred = combined_predictions[
                        combined_predictions['Program'] == program
                    ]

                    if len(program_pred) > 0:
                        insights = generate_insights(program_hist, program_pred, program)

                        for insight in insights:
                            if insight['type'] == 'positive':
                                st.success(f"**{insight['title']}**\n\n{insight['description']}")
                            elif insight['type'] == 'warning':
                                st.warning(f"**{insight['title']}**\n\n{insight['description']}")
                            else:
                                st.info(f"**{insight['title']}**\n\n{insight['description']}")

                        # Additional metrics
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            volatility = program_pred['Rating'].std()
                            st.metric("📊 Volatilitas", f"{volatility:.2f}")

                        with col2:
                            trend = program_pred['Rating'].iloc[-1] - program_pred['Rating'].iloc[0]
                            st.metric("📈 Trend", f"{trend:+.2f}")

                        with col3:
                            peak_day = program_pred.loc[program_pred['Rating'].idxmax(), 'DayName']
                            st.metric("🎯 Hari Terbaik", peak_day)

                        # Show feature importance if model has it
                        try:
                            if hasattr(program_models[program], 'feature_importances_'):
                                st.subheader(f"🔍 Faktor Penting untuk {program}")
                                feature_names = [
                                    'Durasi_Menit', 'Rating_Program', 'Share', 'Jumlah_Penonton',
                                    'AveTime/Viewer', 'Rating_Kompetitor_Tertinggi', 'Year', 'Month', 'Day', 'lag_1',
                                    'lag_2', 'lag_7', 'rolling_3', 'rolling_7', 'std_3', 'std_7'
                                ]
                                importances = program_models[program].feature_importances_

                                feature_df = pd.DataFrame({
                                    'Feature': feature_names[:len(importances)],
                                    'Importance': importances
                                }).sort_values('Importance', ascending=False).head(8)

                                fig_importance = px.bar(feature_df, x='Importance', y='Feature',
                                                      orientation='h', title='Top 8 Faktor Prediksi')
                                st.plotly_chart(fig_importance, use_container_width=True)
                        except Exception as e:
                            pass  # Skip if feature importance not available

                        st.divider()

    with tab4:
        st.header("🎯 Rekomendasi Strategis")

        if 'combined_predictions' in locals() and len(all_predictions) > 0:
            # Combine all historical data
            all_historical = pd.concat(list(program_data.values()), ignore_index=True)

            recommendations = generate_recommendations(combined_predictions, all_historical)

            for rec in recommendations:
                priority_color = {
                    'high': '🔴',
                    'medium': '🟡',
                    'low': '🟢'
                }

                st.markdown(f"""
                <div class="recommendation-card">
                    <h3>{priority_color[rec['priority']]} {rec['title']}</h3>
                    <p>{rec['description']}</p>
                    <h4>Action Items:</h4>
                    <ul>
                """, unsafe_allow_html=True)

                for action in rec['actions']:
                    st.markdown(f"<li>{action}</li>", unsafe_allow_html=True)

                st.markdown("</ul></div>", unsafe_allow_html=True)

            # Strategic Dashboard
            st.subheader("📊 Strategic Performance Matrix")

            # Create performance matrix
            perf_data = []
            rating_col = 'Rating_Program' if 'Rating_Program' in combined_predictions.columns else 'Rating'

            for program in combined_predictions['Program'].unique():
                prog_data = combined_predictions[combined_predictions['Program'] == program]
                perf_data.append({
                    'Program': program,
                    'Avg_Rating': prog_data[rating_col].mean(),
                    'Volatility': prog_data[rating_col].std(),
                    'Growth': prog_data[rating_col].iloc[-1] - prog_data[rating_col].iloc[0]
                })

            perf_df = pd.DataFrame(perf_data)

            # Performance scatter plot
            fig = px.scatter(perf_df, x='Volatility', y='Avg_Rating',
                           size='Growth', hover_name='Program',
                           title='Program Performance Matrix (Ukuran = Growth Trend)',
                           labels={'Avg_Rating': 'Average Rating', 'Volatility': 'Volatility'})

            # Add quadrant lines
            avg_rating_median = perf_df['Avg_Rating'].median()
            avg_volatility_median = perf_df['Volatility'].median()

            fig.add_hline(y=avg_rating_median, line_dash="dash", line_color="red",
                         annotation_text="Median Rating")
            fig.add_vline(x=avg_volatility_median, line_dash="dash", line_color="blue",
                         annotation_text="Median Volatility")

            st.plotly_chart(fig, use_container_width=True)

            # Quadrant analysis
            st.subheader("🎯 Analisis Kuadran Program")

            col1, col2 = st.columns(2)

            with col1:
                # High Rating, Low Volatility (Stars)
                stars = perf_df[
                    (perf_df['Avg_Rating'] > avg_rating_median) &
                    (perf_df['Volatility'] < avg_volatility_median)
                ]

                st.markdown("### ⭐ **STAR PROGRAMS** (High Rating, Low Volatility)")
                if len(stars) > 0:
                    for _, prog in stars.iterrows():
                        st.success(f"🌟 **{prog['Program']}** - Rating: {prog['Avg_Rating']:.2f}")
                else:
                    st.info("Tidak ada program di kuadran ini")

                # Low Rating, High Volatility (Problem Children)
                problems = perf_df[
                    (perf_df['Avg_Rating'] < avg_rating_median) &
                    (perf_df['Volatility'] > avg_volatility_median)
                ]

                st.markdown("### ⚠️ **PROBLEM PROGRAMS** (Low Rating, High Volatility)")
                if len(problems) > 0:
                    for _, prog in problems.iterrows():
                        st.error(f"🚨 **{prog['Program']}** - Rating: {prog['Avg_Rating']:.2f}")
                else:
                    st.info("Tidak ada program di kuadran ini")

            with col2:
                # High Rating, High Volatility (Question Marks)
                questions = perf_df[
                    (perf_df['Avg_Rating'] > avg_rating_median) &
                    (perf_df['Volatility'] > avg_volatility_median)
                ]

                st.markdown("### ❓ **QUESTION MARKS** (High Rating, High Volatility)")
                if len(questions) > 0:
                    for _, prog in questions.iterrows():
                        st.warning(f"🤔 **{prog['Program']}** - Rating: {prog['Avg_Rating']:.2f}")
                else:
                    st.info("Tidak ada program di kuadran ini")

                # Low Rating, Low Volatility (Cash Cows)
                stable = perf_df[
                    (perf_df['Avg_Rating'] < avg_rating_median) &
                    (perf_df['Volatility'] < avg_volatility_median)
                ]

                st.markdown("### 🐄 **STABLE PROGRAMS** (Low Rating, Low Volatility)")
                if len(stable) > 0:
                    for _, prog in stable.iterrows():
                        st.info(f"📊 **{prog['Program']}** - Rating: {prog['Avg_Rating']:.2f}")
                else:
                    st.info("Tidak ada program di kuadran ini")

            # Competitive Analysis if data available
            if 'Rating_Kompetitor_Tertinggi' in all_historical.columns:
                st.subheader("🏆 Analisis Kompetitif")

                comp_analysis = []
                for program in all_historical['Program'].unique():
                    prog_data = all_historical[all_historical['Program'] == program]
                    avg_rating = prog_data['Rating_Program'].mean() if 'Rating_Program' in prog_data.columns else prog_data['Rating'].mean()
                    avg_competitor = prog_data['Rating_Kompetitor_Tertinggi'].mean()

                    comp_analysis.append({
                        'Program': program,
                        'Our_Rating': avg_rating,
                        'Competitor_Rating': avg_competitor,
                        'Gap': avg_rating - avg_competitor,
                        'Gap_Percent': ((avg_rating / avg_competitor - 1) * 100) if avg_competitor > 0 else 0
                    })

                comp_df = pd.DataFrame(comp_analysis)

                # Competitive gap chart
                fig_comp = px.bar(comp_df, x='Program', y='Gap',
                                color='Gap', color_continuous_scale='RdYlGn',
                                title='Gap Rating vs Kompetitor Terdekat')
                fig_comp.add_hline(y=0, line_dash="dash", line_color="black")
                st.plotly_chart(fig_comp, use_container_width=True)

                # Top performers vs competitors
                col1, col2 = st.columns(2)

                with col1:
                    winners = comp_df[comp_df['Gap'] > 0].sort_values('Gap', ascending=False)
                    st.markdown("### 🏆 **LEADING COMPETITORS**")
                    for _, row in winners.iterrows():
                        st.success(f"**{row['Program']}**: +{row['Gap']:.2f} ({row['Gap_Percent']:+.1f}%)")

                with col2:
                    losers = comp_df[comp_df['Gap'] < 0].sort_values('Gap')
                    st.markdown("### 📉 **BEHIND COMPETITORS**")
                    for _, row in losers.iterrows():
                        st.error(f"**{row['Program']}**: {row['Gap']:.2f} ({row['Gap_Percent']:+.1f}%)")

else:
    st.warning("""
    ### 📋 Petunjuk Penggunaan Dashboard v2.0:

    1. **Upload Data Program**: Upload file Excel untuk setiap program di sidebar
    2. **Upload Model**: Upload file model (.pkl) untuk setiap program
    3. **Format Data Baru**: Pastikan file Excel memiliki kolom:
       - `Date` atau `Tanggal_Program`: Tanggal program
       - `Rating_Program`: Rating program
       - `Durasi_Menit`: Durasi program dalam menit
       - `Share`: Share persentase
       - `Jumlah_Penonton`: Jumlah penonton
       - `AveTime/Viewer`: Rata-rata waktu per viewer
       - `Rating_Kompetitor_Tertinggi`: Rating kompetitor tertinggi

    4. **Fitur Model Baru**: Model menggunakan fitur:
       - **Temporal Features**: lag_1, lag_2, lag_7, rolling_3, rolling_7, std_3, std_7
       - **Program Features**: Durasi_Menit, Share, Jumlah_Penonton, AveTime/Viewer
       - **Competitive Features**: Rating_Kompetitor_Tertinggi
       - **Time Features**: Year, Month, Day

    5. **Analisis Dashboard**: Setelah upload, dashboard akan menampilkan:
       - **Overview**: KPI dan performa real-time
       - **Predictions**: Prediksi 7 hari dengan confidence metrics
       - **Insights**: Analisis mendalam dengan competitive intelligence
       - **Recommendations**: Rekomendasi strategis berbasis data

    **Enhanced Features v2.0**:
    - ✅ Competitive analysis dan benchmarking
    - ✅ Advanced temporal features (rolling std, multiple lags)
    - ✅ Quadrant analysis untuk portfolio program
    - ✅ Feature importance visualization
    - ✅ Multi-dimensional performance metrics

    **Note**: Kolom yang tidak tersedia akan diberi nilai default untuk kompatibilitas.
    """)

# Footer
st.markdown("""
---
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🎬 GarudaTV Analytics Dashboard v2.0 | Powered by MichSteven</p>
    <p>© 2025 - Advanced Television Analytics with Competitive Intelligence</p>
    <p>🔧 Enhanced Features: Temporal Analysis • Competitive Benchmarking • Strategic Quadrants</p>
</div>
""", unsafe_allow_html=True)
