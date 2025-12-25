import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from numerical_methods import *

st.set_page_config(page_title="PROYEK AKHIR METODE NUMERIK", layout="wide")

# ========================================
# DATA PREPROCESSING
# ========================================

@st.cache_data
def load_and_clean_data():
    """
    Load dan clean data konsumsi energi listrik
    Mengambil 10.080 data pertama (1 Minggu)
    """
    try:
        # Load data dengan separator ';'
        df = pd.read_csv('household_power_consumption.txt', sep=';', low_memory=False)
        
        # Gabungkan Date dan Time menjadi Datetime
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
        
        # Tangani Missing Values ('?') dan konversi ke angka
        cols_to_fix = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
        for col in cols_to_fix:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Hapus baris yang mengandung NaN
        df = df.dropna(subset=['Global_active_power'])
        
        # Ambil subset data (1 minggu pertama)
        df_subset = df.head(10080).copy()
        
        return df_subset
    except FileNotFoundError:
        st.error("File 'household_power_consumption.txt' tidak ditemukan!")
        return None

# Load data
df_raw = load_and_clean_data()

if df_raw is not None:
    # # Resample ke hourly data
    # df_hourly = df_raw.set_index('datetime').resample('H').mean(numeric_only=True).reset_index()
    
    # # Data untuk analisis
    # power = df_hourly['Global_active_power'].values
    # h = 1.0  # Step size dalam jam
    


   # 1. DATA MENIT (Resolusi Tinggi)
    # Digunakan untuk: Integrasi (Akurasi), Diferensiasi (Anomali), Interpolasi (Recovery)
    df_minute = df_raw.reset_index(drop=True)
    power_minute = df_minute['Global_active_power'].values
    h_minute = 1.0 / 60.0 # Step size 1 menit = 1/60 jam (Agar output tetap kWh)
    
    # 2. DATA JAM (Resolusi Rendah)
    # Digunakan khusus untuk: Regresi (Agar pola tren terlihat jelas dan tidak noisy)
    df_hourly = df_minute.set_index('datetime').resample('H').mean(numeric_only=True).reset_index()
    df_hourly = df_hourly.dropna(subset=['Global_active_power']) # Hapus NaN sisa resample
    power_hourly = df_hourly['Global_active_power'].values
    

    # ========================================
    # HEADER & INTRODUCTION
    # ========================================
    
    st.title("ANALISIS KONSUMSI ENERGI LISTRIK RUMAH TANGGA MENGGUNAKAN METODE NUMERIK")
    st.markdown("---")
    
    st.markdown("""
    ### Deskripsi Proyek
    
    Proyek ini bertujuan untuk menganalisis konsumsi energi listrik pada satu rumah tangga di **Sceaux, Prancis** (Des 2006 – Nov 2010) menggunakan metode numerik.
    
    **Masalah Utama:**
    Data daya listrik bersifat diskrit (per menit) dengan fluktuasi tajam dan memiliki celah data kosong (*missing values*), sehingga tidak dapat dihitung menggunakan rumus analitik biasa.
    
    **Tujuan & Implementasi Metode Numerik:**
    * **Integrasi Numerik** $\\rightarrow$ Menghitung total energi (kWh) dari data daya sesaat untuk estimasi tagihan.
    * **Diferensiasi Numerik** $\\rightarrow$ Mendeteksi anomali atau lonjakan arus (*spike*) yang tidak wajar.
    * **Interpolasi** $\\rightarrow$ Memulihkan (*recovery*) data yang hilang agar rekaman menjadi kontinu.
    * **Regresi** $\\rightarrow$ Memodelkan tren konsumsi untuk melihat pola jangka panjang.

    ---
    **Metode Spesifik yang Digunakan:**
    1.  **Integrasi:** Trapezoidal, Simpson 1/3, Simpson 3/8, Richardson Extrapolation.
    2.  **Diferensiasi:** Forward, Backward, Central Difference, Second Order Derivative.
    3.  **Interpolasi:** Lagrange & Cubic Spline.
    4.  **Regresi:** Polinomial Least Squares (dengan Analisis SPL).
    
    **Sumber Data:** [UCI Machine Learning Repository - Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
    """)
    
    # Info dataset
    start_date = df_minute['datetime'].min().strftime('%d %b %Y')
    end_date = df_minute['datetime'].max().strftime('%d %b %Y')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Data Points", f"{len(df_minute):,} (Menit)")
    with col2:
        st.metric("Periode Data", "1 Minggu", help=f"Dari {start_date} sampai {end_date}")
    with col3:
        st.metric("Rata-rata Power", f"{np.mean(power_minute):.2f} kW")
    with col4:
        st.metric("Std Dev Power", f"{np.std(power_minute):.2f} kW")
    
    st.info("""
    **Catatan Lingkup Analisis:** Meskipun dataset asli mencakup periode 4 tahun, analisis pada dashboard ini dibatasi pada sampel data **1 minggu pertama**. 
    Hal ini dilakukan untuk menjaga efisiensi komputasi pada algoritma kompleks (seperti Interpolasi & Regresi) serta agar pola fluktuasi harian dapat divisualisasikan dengan jelas.
    """)
    st.markdown("---")
    
    # ========================================
    # TAB NAVIGATION
    # ========================================
    
    tabs = st.tabs([
        "1. Integrasi (Total Energi)",
        "2. Diferensiasi (Deteksi Anomali)",
        "3. Interpolasi (Recovery Data)",
        "4. Regresi (Tren Prediction)",
        "5. Kesimpulan"
    ])
    
    # ========================================
    # TAB 1: INTEGRASI NUMERIK
    # ========================================
    
    with tabs[0]:
        st.header("Integrasi Numerik: Menghitung Total Energi Konsumsi (kWh)")
        
        st.markdown("""
        **Tujuan:** Menghitung total energi yang dikonsumsi selama 1 minggu dengan mengintegrasikan
        kurva power consumption terhadap waktu.
        
        **Formula:** Energy (kWh) = ∫ Power(t) dt
        """)
        
        st.subheader("Pengaturan Parameter Integrasi")
        
        show_comparison = st.checkbox("Tampilkan Perbandingan dengan NumPy", value=True)
        
        st.markdown("---")
        

        val_trap = manual_trapezoidal(power_minute, h_minute)
        val_simp13 = manual_simpson_13(power_minute, h_minute)

        # Simpson 3/8 (Cek Syarat Strict)
        # Syarat: Jumlah interval (N-1) harus kelipatan 3
        n_data = len(power_minute)
        n_interval = n_data - 1
        
        # if n_interval % 3 == 0:
        val_simp38 = manual_simpson_38_smart(power_minute, h_minute)
            # status_38 = "Valid"
        # else:
            # val_simp38 = None # Tandai bahwa metode ini gagal
            # status_38 = f"Invalid (Interval {n_interval} tidak habis dibagi 3)"
        
        # Nilai referensi
        val_numpy = np.trapezoid(power_minute, dx=h_minute)
        
        # Richardson Extrapolation
        power_h2 = power_minute[::2]
        val_trap_h2 = manual_trapezoidal(power_h2, h_minute*2)
        val_richardson = richardson_extrapolation(val_trap_h2,val_trap,2)
        exact_val = val_richardson

        # Display hasil
        st.subheader("Hasil Perhitungan")
        
        # col1, col2, col3, col4 = st.columns(4)
        col1, col2 = st.columns(2)
        # with col1:
            # st.metric("Rectangular (Left)", f"{val_rect_left:.2f} kWh")
            # st.metric("Rectangular (Right)", f"{val_rect_right:.2f} kWh")
        with col1:
            # st.metric("Rectangular (Mid)", f"{val_rect_mid:.2f} kWh")
            st.metric("Trapezoidal", f"{val_trap:.2f} kWh")
            st.metric("Simpson 1/3", f"{val_simp13:.2f} kWh")
        with col2:
            if val_simp38 is not None:
                st.metric("Simpson 3/8", f"{val_simp38:.4f} kWh")
            else:
                st.metric("Simpson 3/8", "N/A (Interval tidak kelipatan 3)")

            st.metric("Richardson Extrap.", f"{val_richardson:.2f} kWh", 
                     help="Menggunakan Richardson Extrapolation untuk akurasi lebih tinggi")
        # with col3:
            
            # st.metric("Adaptive (Segmented)", f"{val_adaptive:.2f} kWh")
            
        if show_comparison:
            st.info(f"NumPy Reference (np.trapezoid): {val_numpy:.2f} kWh")
        
        # Analisis Error
        st.subheader("Analisis Error")
        
        # methods = ['Rect (Left)', 'Rect (Right)', 'Rect (Mid)', 'Trapezoidal', 
        #            'Simpson 1/3', 'Adaptive', 'NumPy']
        methods = ['Richardson (Benchmark)','Trapezoidal', 
                   'Simpson 1/3', 'NumPy']
        # values = [val_rect_left, val_rect_right, val_rect_mid, val_trap, 
        #           val_simp13, val_adaptive, val_numpy]
        values = [val_richardson, val_trap, val_simp13, val_numpy]
        
        if val_simp38:
            methods.insert(-1, 'Simpson 3/8')
            values.insert(-1, val_simp38)
        
        errors_abs = [abs(v - exact_val) for v in values]
        errors_rel = [abs(v - exact_val) / exact_val * 100 for v in values]
        
        df_error = pd.DataFrame({
            'Metode': methods,
            'Hasil (kWh)': [f"{v:.4f}" for v in values],
            'Absolute Error': [f"{e:.4f}" for e in errors_abs],
            'Relative Error (%)': [f"{e:.4f}" for e in errors_rel]
        })
        
        st.dataframe(df_error, use_container_width=True)
        
        # Visualisasi error dengan Plotly
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Absolute Error Comparison', 'Relative Error Comparison')
        )
        
        # Absolute Error
        fig.add_trace(
            go.Bar(x=methods, y=errors_abs, name='Absolute Error',
                   marker_color='steelblue', opacity=0.7,
                   hovertemplate='<b>%{x}</b><br>Error: %{y:.4f} kWh<extra></extra>'),
            row=1, col=1
        )
        
        # Relative Error
        fig.add_trace(
            go.Bar(x=methods, y=errors_rel, name='Relative Error',
                   marker_color='coral', opacity=0.7,
                   hovertemplate='<b>%{x}</b><br>Error: %{y:.4f}%<extra></extra>'),
            row=1, col=2
        )
        
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(title_text="Absolute Error (kWh)", row=1, col=1)
        fig.update_yaxes(title_text="Relative Error (%)", row=1, col=2)
        fig.update_layout(height=500, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Pembahasan
        st.subheader("Analisis Hasil & Kesimpulan:")
        st.markdown(f"""
        1.  **Akurasi Metode:**
            * Metode **Simpson 1/3** menghasilkan nilai total energi sebesar **{val_simp13:.2f} kWh**.
            * Nilai ini sangat mendekati hasil perhitungan **Richardson Extrapolation** ({val_richardson:.2f} kWh) yang kita asumsikan sebagai nilai paling presisi (Exact Value) karena menggabungkan dua step size berbeda untuk meminimalisir error.
            * Hal ini menunjukkan bahwa untuk dataset daya listrik ini, pendekatan parabola pada Simpson 1/3 mampu menangkap kelengkungan fluktuasi daya dengan lebih baik dibandingkan pendekatan linear (garis lurus) pada Trapezoidal.
        2.  **Implikasi Tagihan Listrik:**
            * Total konsumsi energi selama 1 minggu terhitung sangat konsisten di angka ~**{val_richardson:.2f} kWh** berdasarkan estimasi paling presisi (Richardson)..
            * Dengan asumsi tarif listrik rata-rata Rp 1.444 per kWh, estimasi tagihan biaya listrik untuk minggu ini adalah: **Rp {val_richardson * 1444:,.0f}**.

        **Rekomendasi:**
        Untuk analisis data sensor listrik berfrekuensi tinggi (High-Frequency data), metode Simpson 1/3 adalah pilihan paling optimal. Metode ini menawarkan keseimbangan terbaik antara efisiensi komputasi dan akurasi tinggi (orde galat $O(h^4)$) tanpa memerlukan kerumitan komputasi ganda seperti pada metode Richardson.
        """)
    
# Di TAB 2: DIFERENSIASI NUMERIK (Ganti bagian slider dan visualisasi)

with tabs[1]:
    st.header("Diferensiasi Numerik: Deteksi Anomali dan Laju Perubahan")
    
    st.markdown("""
    **Tujuan:** Menghitung laju perubahan power consumption (dP/dt) untuk mendeteksi
    anomali atau lonjakan mendadak dalam penggunaan listrik.
    
    **Formula:** dP/dt ≈ [P(t+h) - P(t-h)] / (2h) (Central Difference)
    
    **Interpretasi Fisik:**
    - **dP/dt > 0**: Lonjakan konsumsi (alat elektronik dinyalakan)
    - **dP/dt < 0**: Penurunan konsumsi (alat elektronik dimatikan)
    - **|dP/dt| besar**: Perubahan mendadak (potensial anomali)
    """)
    
    # 1. Hitung Turunan (Pakai Data MENIT)
    dpdt_forward = manual_diff_forward(power_minute, h_minute)
    dpdt_backward = manual_diff_backward(power_minute, h_minute)
    dpdt_central = manual_diff_central(power_minute, h_minute)
    dpdt_second = manual_diff_second_order(power_minute, h_minute)
    
    # 2. Statistik Otomatis untuk Threshold yang Lebih Relevan
    mean_abs_dpdt = np.mean(np.abs(dpdt_central))
    std_dpdt = np.std(dpdt_central)
    
    # Metode deteksi: 3-sigma rule (99.7% data normal dalam ±3σ)
    suggested_threshold = mean_abs_dpdt + 3 * std_dpdt
    max_dpdt = np.max(np.abs(dpdt_central))
    
    st.subheader("Pengaturan Deteksi Anomali")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        threshold = st.slider(
            "Threshold Anomali (kW/jam)", 
            min_value=10.0,
            max_value=float(max_dpdt), 
            value=float(suggested_threshold),
            step=5.0,
            help=f"Saran sistem: {suggested_threshold:.1f} (berdasarkan 3σ rule)"
        )
    
    with col2:
        st.metric("Mean |dP/dt|", f"{mean_abs_dpdt:.2f}")
    
    with col3:
        st.metric("Std Dev", f"{std_dpdt:.2f}")
    
    # 3. Hitung Anomali SEBELUM Visualisasi
    anomali_indices = np.where(np.abs(dpdt_central) > threshold)[0]
    n_anomali = len(anomali_indices)
    anomali_rate = (n_anomali / len(power_minute)) * 100
    
    # Info Box Anomali
    if n_anomali > 0:
        st.warning(f"""
         **Terdeteksi {n_anomali:,} titik anomali** ({anomali_rate:.2f}% dari total data)
        - **Lonjakan terbesar**: {np.max(np.abs(dpdt_central)):.2f} kW/jam
        - **Waktu pertama**: {df_minute['datetime'].iloc[anomali_indices[0]].strftime('%Y-%m-%d %H:%M')}
        """)
    else:
        st.success(f" Tidak ada anomali terdeteksi dengan threshold {threshold:.1f} kW/jam")
    
    # 4. Visualisasi dengan Highlight Anomali
    st.subheader("Visualisasi Perbandingan Metode")
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '1. Original Power Consumption (dengan Anomali)',    
            '2. Forward Difference (O(h))',
            '3. Backward Difference (O(h))',   
            '4. Central Difference (Best - O(h²)) + Anomali',
            '5. Second Derivative (Acceleration)', ''
        ),
        vertical_spacing=0.2,
        shared_xaxes=False,
        specs=[[{}, {}], [{}, {}], [{"colspan": 2}, None]]
    )
    
    # --- ROW 1 COL 1: Original dengan Highlight Anomali ---
    fig.add_trace(
        go.Scatter(
            x=df_minute['datetime'], 
            y=power_minute, 
            mode='lines', 
            name='Power', 
            line=dict(color='orange', width=1)
        ), 
        row=1, col=1
    )
    
    # Tambahkan marker untuk titik anomali
    if n_anomali > 0:
        fig.add_trace(
            go.Scatter(
                x=df_minute['datetime'].iloc[anomali_indices],
                y=power_minute[anomali_indices],
                mode='markers',
                name='Anomali',
                marker=dict(color='red', size=6, symbol='x'),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # --- ROW 1 COL 2: Forward ---
    fig.add_trace(
        go.Scatter(x=df_minute['datetime'], y=dpdt_forward, mode='lines', 
                   name='Forward', line=dict(color='green', width=1)), 
        row=1, col=2
    )
    fig.add_hline(y=threshold, line_dash="dot", line_color="gray", row=1, col=2)
    fig.add_hline(y=-threshold, line_dash="dot", line_color="gray", row=1, col=2)

    # --- ROW 2 COL 1: Backward ---
    fig.add_trace(
        go.Scatter(x=df_minute['datetime'], y=dpdt_backward, mode='lines', 
                   name='Backward', line=dict(color='red', width=1)), 
        row=2, col=1
    )
    fig.add_hline(y=threshold, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=-threshold, line_dash="dot", line_color="gray", row=2, col=1)
    
    # --- ROW 2 COL 2: Central dengan Highlight Anomali ---
    fig.add_trace(
        go.Scatter(x=df_minute['datetime'], y=dpdt_central, mode='lines', 
                   name='Central', line=dict(color='blue', width=1)), 
        row=2, col=2
    )
    
    # Highlight anomali di Central
    if n_anomali > 0:
        fig.add_trace(
            go.Scatter(
                x=df_minute['datetime'].iloc[anomali_indices],
                y=dpdt_central[anomali_indices],
                mode='markers',
                name='Anomali',
                marker=dict(color='red', size=8, symbol='diamond'),
                showlegend=False
            ),
            row=2, col=2
        )
    
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                  annotation_text=f"Threshold: ±{threshold:.1f}", row=2, col=2)
    fig.add_hline(y=-threshold, line_dash="dash", line_color="red", row=2, col=2)

    # --- ROW 3 COL 1: Second Derivative (Full Width) ---
    fig.add_trace(
        go.Scatter(x=df_minute['datetime'], y=dpdt_second, mode='lines', 
                   name='Acceleration', line=dict(color='purple', width=1)), 
        row=3, col=1
    )

    # Update Layout
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="dP/dt (kW/jam)", row=1, col=2)
    fig.update_yaxes(title_text="dP/dt (kW/jam)", row=2, col=1)
    fig.update_yaxes(title_text="dP/dt (kW/jam)", row=2, col=2)
    fig.update_yaxes(title_text="d²P/dt² (kW/jam²)", row=3, col=1)

    # Tambahkan keterangan sumbu X untuk semua grafik
    fig.update_xaxes(title_text="Waktu", row=1, col=1)
    fig.update_xaxes(title_text="Waktu", row=1, col=2)
    fig.update_xaxes(title_text="Waktu", row=2, col=1)
    fig.update_xaxes(title_text="Waktu", row=2, col=2)
    fig.update_xaxes(title_text="Waktu", row=3, col=1)

    fig.update_layout(
        height=1000, 
        showlegend=False, 
        title_text=f"Analisis Diferensiasi Lengkap ({n_anomali} Anomali Terdeteksi)",
        margin=dict(t=50, b=50) # Memberi ruang agar label sumbu X tidak terpotong
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 5. Tabel Anomali & Error Analysis
    st.markdown("---")
    
    # --- PERHITUNGAN ERROR ---
    dpdt_numpy = np.gradient(power_minute, h_minute)
    
    err_fwd = np.sqrt(np.mean((dpdt_numpy - dpdt_forward)**2))
    err_bwd = np.sqrt(np.mean((dpdt_numpy - dpdt_backward)**2))
    err_cen = np.sqrt(np.mean((dpdt_numpy - dpdt_central)**2))
    
    errors_methods = {
        'Forward': calculate_errors(dpdt_numpy, dpdt_forward),
        'Backward': calculate_errors(dpdt_numpy, dpdt_backward),
        'Central': calculate_errors(dpdt_numpy, dpdt_central)
    }
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(f"Daftar Anomali (Top 10 dari {n_anomali})")
        
        if n_anomali > 0:
            anomali_data = df_minute.iloc[anomali_indices].copy()
            anomali_data['dP/dt (kW/jam)'] = dpdt_central[anomali_indices]
            anomali_data['Severity'] = np.abs(anomali_data['dP/dt (kW/jam)'])
            
            # Sort by severity
            anomali_data = anomali_data.sort_values('Severity', ascending=False)
            
            # Klasifikasi severity
            def classify_severity(val):
                if val > threshold * 3:
                    return "Kritis"
                elif val > threshold * 2:
                    return "*Tinggi"
                else:
                    return "Sedang"
            
            anomali_data['Level'] = anomali_data['Severity'].apply(classify_severity)
            
            display_cols = ['datetime', 'Global_active_power', 'dP/dt (kW/jam)', 'Level']
            st.dataframe(
                anomali_data[display_cols].head(10).style.format({
                    'Global_active_power': '{:.3f}',
                    'dP/dt (kW/jam)': '{:.2f}'
                }), 
                use_container_width=True
            )
            
        
    
    with col2:
        st.subheader("Akurasi Metode (RMSE vs NumPy)")
        
        comparison_df = pd.DataFrame({
            'Metode': ['Forward', 'Backward', 'Central'],
            'RMSE': [f"{err_fwd:.4f}", f"{err_bwd:.4f}", f"{err_cen:.4f}"],
            'MAE': [
                f"{errors_methods['Forward']['MAE']:.4f}",
                f"{errors_methods['Backward']['MAE']:.4f}",
                f"{errors_methods['Central']['MAE']:.4f}"
            ],
            'Rank': ['(3️)', '(2️)', '(1️) Best']
        })
        
        st.dataframe(comparison_df, use_container_width=True)
    
    # 6. Pembahasan yang Lebih Detail
    st.subheader("Pembahasan & Kesimpulan Akhir")
    
    max_accel = np.max(np.abs(dpdt_second))
    rmse_improve = err_fwd / err_cen if err_cen > 0 else 0
    
    # Analisis distribusi anomali
    if n_anomali > 0:
        anomali_positive = np.sum(dpdt_central[anomali_indices] > 0)
        anomali_negative = n_anomali - anomali_positive
        
        anomali_analysis = f"""
        Karakteristik Anomali yang Terdeteksi:
        - Total Anomali: {n_anomali:,} titik ({anomali_rate:.2f}% dari data)
        - Lonjakan Naik (alat dinyalakan): {anomali_positive} kali
        - Lonjakan Turun (alat dimatikan): {anomali_negative} kali
        - Magnitude Tertinggi: {np.max(np.abs(dpdt_central[anomali_indices])):.2f} kW/jam
        - Frekuensi: Sekitar {n_anomali / 7:.1f} anomali per hari
        
        Interpretasi Praktis:
        {'Tingkat anomali cukup tinggi (>1%) - Perlu investigasi pola penggunaan' if anomali_rate > 1 
         else 'Tingkat anomali normal (<1%) - Pola penggunaan wajar'}
        """
    else:
        anomali_analysis = f"""
        **Status**: Tidak ada anomali terdeteksi dengan threshold {threshold:.1f} kW/jam.
        Ini menunjukkan pola konsumsi yang stabil dan predictable.
        """
    
    st.markdown(f"""
    **1. Superioritas Telak Central Difference:**
    * Data pada tabel *Akurasi (RMSE)* menunjukkan bukti empiris yang kuat:
        * **RMSE Forward/Backward:** ~{err_fwd:.4f}
        * **RMSE Central:** {err_cen:.4f}
    * Artinya, metode **Central Difference sekitar {rmse_improve:.1f}x lebih akurat** dibandingkan metode Forward/Backward. 
      Hal ini mengonfirmasi teori bahwa Central Difference ($O(h^2)$) sangat presisi untuk data sensor rapat.

    **2. Deteksi Anomali dengan 3-Sigma Rule:**
    {anomali_analysis}

    **3. Analisis Ekstrem Second Derivative:**
    * Grafik *Acceleration* menunjukkan nilai mencapai kisaran **±{max_accel:,.0f} kW/jam²**.
    * **Mengapa nilainya sangat besar?** Ini menunjukkan bahwa perubahan daya listrik terjadi secara **Instan (Diskrit)**.
    * **Efeknya terhadap Sistem:** Nilai turunan kedua yang "meledak" ini menandakan bahwa data memiliki **Volatilitas Tinggi**. 
      Ini menjadi peringatan bahwa jika ingin melakukan *forecasting*, model linear biasa akan kesulitan menangkap pola "lompatan" ini.

    **Rekomendasi Tindakan:**
    1. Gunakan **Central Difference** untuk sistem monitoring otomatis karena akurasinya yang tinggi meminimalisir *false alarm*.
    2. Untuk rumah tangga ini, threshold optimal adalah **{suggested_threshold:.1f} kW/jam** (3σ dari rata-rata).
    3. Anomali dengan severity "Kritis" (>3x threshold) perlu investigasi lebih lanjut - bisa jadi indikasi:
       - Alat elektronik rusak (short circuit)
       - Penggunaan berlebihan yang tidak efisien
       - Masalah pada instalasi listrik
    """)


    # ========================================
    # TAB 3: INTERPOLASI
    # ========================================
    
    with tabs[2]:
        st.header("Interpolasi: Recovery Data yang Hilang")
        
        st.markdown("""
        **Tujuan:** Melakukan recovery atau prediksi data yang hilang menggunakan
        interpolasi polinomial.
        
        **Metode:**
        - Lagrange Interpolation
        - Cubic Spline
        """)
        
        # Parameter simulasi missing data
        st.subheader("Simulasi Missing Data")
        
        col1, col2 = st.columns(2)
        with col1:
            start_missing = st.slider("Start Index Missing Data", 10, 100, 50)
        with col2:
            n_missing = st.slider("Jumlah Data Hilang", 1, 20, 6)
        
        # Simulasi data hilang
        idx_missing = list(range(start_missing, start_missing + n_missing))
        
        # Sesuaikan window_size dengan slider maximum
        window_size = 150  # Atau buat dynamic
        x_full = np.arange(min(window_size, len(power_minute)))
        y_full = power_minute[:len(x_full)].copy()
        
        # Hapus data pada indeks missing
        x_sample = np.delete(x_full, [i for i in idx_missing if i < window_size])
        y_sample = np.delete(y_full, [i for i in idx_missing if i < window_size])
        
        # Interpolasi
        
        # Evaluasi interpolasi
        x_interp = np.linspace(0, window_size-1, 200)
        y_lagrange = [lagrange_interpolation(x_sample, y_sample, xi) for xi in x_interp]
        
        # Cubic spline
        spline_coef = cubic_spline_coefficients(x_sample, y_sample)
        y_spline = evaluate_cubic_spline(spline_coef, x_interp)
        
        # Recovery data yang hilang
        x_missing_actual = [i for i in idx_missing if i < window_size]
        y_missing_true = [y_full[i] for i in x_missing_actual]
        y_missing_lagrange = [lagrange_interpolation(x_sample, y_sample, xi) for xi in x_missing_actual]
        y_missing_spline = evaluate_cubic_spline(spline_coef, np.array(x_missing_actual))
        
        # Visualisasi dengan Plotly
        st.subheader("Hasil Interpolasi")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Lagrange Interpolation', 
                          'Cubic Spline (No Oscillation!)')
        )
        
        # Plot Lagrange
        fig.add_trace(
            go.Scatter(x=x_sample, y=y_sample, mode='markers',
                      marker=dict(size=8, color='blue'), showlegend=False,
                      hovertemplate='<b>Index:</b> %{x}<br><b>Power:</b> %{y:.2f} kW<extra></extra>'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_missing_actual, y=y_missing_true, mode='markers',
                      marker=dict(size=10, color='red', symbol='x'), showlegend=False,
                      hovertemplate='<b>Index:</b> %{x}<br><b>True:</b> %{y:.2f} kW<extra></extra>'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_interp, y=y_lagrange, mode='lines',
                      line=dict(color='magenta', dash='dash', width=2), showlegend=False,
                      hovertemplate='<b>Index:</b> %{x:.1f}<br><b>Power:</b> %{y:.2f} kW<extra></extra>'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_missing_actual, y=y_missing_lagrange, mode='markers',
                      marker=dict(size=12, color='cyan', symbol='triangle-up',
                      line=dict(width=2, color='black')), showlegend=False,
                      hovertemplate='<b>Index:</b> %{x}<br><b>Pred:</b> %{y:.2f} kW<extra></extra>'),
            row=1, col=1
        )
        
        # Plot Cubic Spline
        fig.add_trace(
            go.Scatter(x=x_sample, y=y_sample, mode='markers',
                      marker=dict(size=8, color='blue'), showlegend=False,
                      hovertemplate='<b>Index:</b> %{x}<br><b>Power:</b> %{y:.2f} kW<extra></extra>'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=x_missing_actual, y=y_missing_true, mode='markers',
                      marker=dict(size=10, color='red', symbol='x'), showlegend=False,
                      hovertemplate='<b>Index:</b> %{x}<br><b>True:</b> %{y:.2f} kW<extra></extra>'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=x_interp, y=y_spline, mode='lines',
                      line=dict(color='purple', width=2), showlegend=False,
                      hovertemplate='<b>Index:</b> %{x:.1f}<br><b>Power:</b> %{y:.2f} kW<extra></extra>'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=x_missing_actual, y=y_missing_spline, mode='markers',
                      marker=dict(size=12, color='yellow', symbol='triangle-up',
                      line=dict(width=2, color='black')), showlegend=False,
                      hovertemplate='<b>Index:</b> %{x}<br><b>Pred:</b> %{y:.2f} kW<extra></extra>'),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Time Index (Hours)")
        fig.update_yaxes(title_text="Power (kW)")
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Analisis Error
        st.subheader("Analisis Error Interpolasi")
        
        if len(y_missing_true) > 0:
            errors_lagrange = calculate_errors(y_missing_true, y_missing_lagrange)
            errors_spline = calculate_errors(y_missing_true, y_missing_spline)
            
            df_interp_error = pd.DataFrame({
                'Lagrange': errors_lagrange,
                'Cubic Spline': errors_spline
            }).T
            
            st.dataframe(df_interp_error, use_container_width=True)
            
            # Detail prediksi vs truth
            st.subheader("Detail Prediksi vs Truth")
            
            df_missing_comparison = pd.DataFrame({
                'Index': x_missing_actual,
                'True Value': y_missing_true,
                'Lagrange Pred': y_missing_lagrange,
                'Spline Pred': y_missing_spline,
                'Lagrange Error': np.abs(np.array(y_missing_true) - np.array(y_missing_lagrange)),
                'Spline Error': np.abs(np.array(y_missing_true) - np.array(y_missing_spline))
            })
            
            st.dataframe(df_missing_comparison, use_container_width=True)
            
            # Visualisasi error per titik dengan Plotly
            fig = go.Figure()
            
            x_pos = np.arange(len(x_missing_actual))
            width = 0.35
            
            
            fig.add_trace(go.Bar(
                x=x_pos,
                y=df_missing_comparison['Lagrange Error'],
                name='Lagrange Error',
                marker_color='magenta',
                opacity=0.7,
                hovertemplate='<b>Point:</b> %{x}<br><b>Error:</b> %{y:.4f} kW<extra></extra>'
            ))
            
            fig.add_trace(go.Bar(
                x=x_pos + width,
                y=df_missing_comparison['Spline Error'],
                name='Cubic Spline Error',
                marker_color='purple',
                opacity=0.7,
                hovertemplate='<b>Point:</b> %{x}<br><b>Error:</b> %{y:.4f} kW<extra></extra>'
            ))
            
            fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=x_pos,
                    ticktext=x_missing_actual,
                    title='Missing Data Point'
                ),
                yaxis_title='Absolute Error (kW)',
                title='Error Comparison per Missing Point',
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Pembahasan
        st.subheader("Pembahasan")
        if len(y_missing_true) > 0:
            best_method = min(
                [('Lagrange', errors_lagrange['MAE']),
                ('Cubic Spline', errors_spline['MAE'])],
                key=lambda x: x[1]
            )
            observasi_text = f"""
        **Observasi:**
        - MAE Lagrange: {errors_lagrange['MAE']:.4f} kW
        - MAE Cubic Spline: {errors_spline['MAE']:.4f} kW
        
        **Perbandingan Metode:**
        1. **Cubic Spline Advantages:**
        - Menghindari Runge's Phenomenon (tidak oscillate)
        - Smooth interpolation (C² continuity)
        - Error rata-rata: {errors_spline['MAE']:.4f} kW
        - Lebih stable untuk banyak data points
        - Natural boundary conditions

        2. **Kapan Gunakan Cubic Spline?**
        - Data dengan banyak titik (>10 points)
        - Membutuhkan smoothness (aplikasi fisika)
        - Interpolasi jangka panjang
        - Menghindari oscillation artifacts

        **Metode Terbaik:** {best_method[0]} dengan MAE = {best_method[1]:.4f} kW

        **Kesimpulan:**
        Cubic Spline memberikan hasil terbaik dengan error rata-rata {errors_spline['MAE']:.4f} kW.
        Untuk recovery data yang hilang, Cubic Spline adalah pilihan optimal karena:
        - Menghindari polynomial oscillation
        - Memberikan curve yang lebih natural dan smooth
        - Lebih robust terhadap noise
        """
        else:
            observasi_text = """
            **Observasi:**
            - Tidak ada data yang hilang dalam window yang dipilih
            - Silakan sesuaikan parameter untuk simulasi missing data
            """
       
        st.markdown(observasi_text)
    
    # ========================================
    # TAB 4: REGRESI POLINOMIAL
    # ========================================
    
    with tabs[3]:
        st.header("Regresi Polinomial: Analisis Tren dan Prediksi")
        
        st.markdown("""
        **Tujuan:** Mengidentifikasi pola dan tren dalam konsumsi energi menggunakan
        regresi polinomial dengan berbagai derajat.
        
        **Metode:** Least Squares Polynomial Regression
        """)
        
        # Parameter regresi
        st.subheader("Pengaturan Parameter Regresi")
        
        col1, col2 = st.columns(2)
        with col1:
            deg = st.slider("Derajat Polinomial", 1, 8, 3)
        with col2:
            train_size = st.slider("Training Data Size (%)", 50, 90, 80)
        
        # Split data
        split_idx = int(len(power_hourly) * train_size / 100)
        x_train = np.arange(split_idx)
        y_train = power_hourly[:split_idx]
        x_test = np.arange(split_idx, len(power_hourly))
        y_test = power_hourly[split_idx:]
        
        # Training regresi
        coeffs = manual_poly_regression(x_train, y_train, deg)
        
        # Prediksi
        y_train_pred = evaluate_polynomial(coeffs, x_train)
        y_test_pred = evaluate_polynomial(coeffs, x_test)
        
        # Visualisasi dengan Plotly
        st.subheader("Hasil Regresi")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'Polynomial Regression (Degree {deg})', 'Residual Plot'),
            vertical_spacing=0.12
        )
        
        # Plot keseluruhan
        fig.add_trace(
            go.Scatter(x=np.arange(len(power_hourly)), y=power_hourly, mode='lines',
                      name='Data Asli', line=dict(color='blue', width=1),
                      opacity=0.5,
                      hovertemplate='<b>Time:</b> %{x}h<br><b>Power:</b> %{y:.2f} kW<extra></extra>'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=x_train, y=y_train_pred, mode='lines',
                      name=f'Regresi Training (Degree {deg})',
                      line=dict(color='red', width=2),
                      hovertemplate='<b>Time:</b> %{x}h<br><b>Pred:</b> %{y:.2f} kW<extra></extra>'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=x_test, y=y_test_pred, mode='lines',
                      name='Prediksi Testing',
                      line=dict(color='green', width=2, dash='dash'),
                      hovertemplate='<b>Time:</b> %{x}h<br><b>Pred:</b> %{y:.2f} kW<extra></extra>'),
            row=1, col=1
        )
        
        fig.add_vline(x=split_idx, line_dash="dot", line_color="black",
                     annotation_text="Train/Test Split", row=1, col=1)
        
        # Plot residual
        residual_train = y_train - y_train_pred
        residual_test = y_test - y_test_pred
        
        fig.add_trace(
            go.Scatter(x=x_train, y=residual_train, mode='markers',
                      name='Residual Training',
                      marker=dict(size=4, color='blue', opacity=0.5),
                      hovertemplate='<b>Time:</b> %{x}h<br><b>Residual:</b> %{y:.2f} kW<extra></extra>'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=x_test, y=residual_test, mode='markers',
                      name='Residual Testing',
                      marker=dict(size=4, color='green', opacity=0.5),
                      hovertemplate='<b>Time:</b> %{x}h<br><b>Residual:</b> %{y:.2f} kW<extra></extra>'),
            row=2, col=1
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2, row=2, col=1)
        
        fig.update_xaxes(title_text="Waktu (Jam ke-n dalam 1 Minggu)", row=1, col=1)
        fig.update_xaxes(title_text="Waktu (Jam ke-n dalam 1 Minggu)", row=2, col=1)
        
        fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
        fig.update_yaxes(title_text="Residual (kW)", row=2, col=1)
        fig.update_layout(height=900)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Analisis Error
        st.subheader("Analisis Error Regresi")
        
        errors_train = calculate_errors(y_train, y_train_pred)
        errors_test = calculate_errors(y_test, y_test_pred)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Error**")
            df_train_error = pd.DataFrame(errors_train, index=['Training']).T
            st.dataframe(df_train_error, use_container_width=True)
        
        with col2:
            st.markdown("**Testing Error**")
            df_test_error = pd.DataFrame(errors_test, index=['Testing']).T
            st.dataframe(df_test_error, use_container_width=True)
        
        # Koefisien regresi
        st.subheader("Koefisien Polinomial")
        
        df_coeffs = pd.DataFrame({
            'Derajat': [f'x^{i}' for i in range(len(coeffs))],
            'Koefisien': coeffs
        })
        st.dataframe(df_coeffs, use_container_width=True)
        
        # Persamaan polinomial
        poly_equation = " + ".join([f"{coeffs[i]:.4f}x^{i}" if i > 0 else f"{coeffs[i]:.4f}" 
                                    for i in range(len(coeffs))])
        st.code(f"P(x) = {poly_equation}", language="text")
        
        # Perbandingan berbagai derajat
        st.subheader("Perbandingan Berbagai Derajat Polinomial")
        
        degrees_to_compare = [1, 2, 3, 4, 5]
        comparison_results = []
        
        for d in degrees_to_compare:
            coeffs_temp = manual_poly_regression(x_train, y_train, d)
            y_test_pred_temp = evaluate_polynomial(coeffs_temp, x_test)
            errors_temp = calculate_errors(y_test, y_test_pred_temp)
            comparison_results.append({
                'Degree': d,
                'R²': errors_temp['R2'],
                'RMSE': errors_temp['RMSE'],
                'MAE': errors_temp['MAE'],
                'MAPE': errors_temp['MAPE']
            })
        
        df_comparison = pd.DataFrame(comparison_results)
        st.dataframe(df_comparison, use_container_width=True)
        
        # Visualisasi perbandingan dengan Plotly
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('R² vs Polynomial Degree', 'Error vs Polynomial Degree')
        )
        
        # R² plot
        fig.add_trace(
            go.Scatter(x=df_comparison['Degree'], y=df_comparison['R²'],
                      mode='lines+markers', name='R²',
                      line=dict(width=2), marker=dict(size=10),
                      hovertemplate='<b>Degree:</b> %{x}<br><b>R²:</b> %{y:.4f}<extra></extra>'),
            row=1, col=1
        )
        
        # Error plot
        fig.add_trace(
            go.Scatter(x=df_comparison['Degree'], y=df_comparison['RMSE'],
                      mode='lines+markers', name='RMSE',
                      line=dict(width=2), marker=dict(size=10),
                      hovertemplate='<b>Degree:</b> %{x}<br><b>RMSE:</b> %{y:.4f}<extra></extra>'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=df_comparison['Degree'], y=df_comparison['MAE'],
                      mode='lines+markers', name='MAE',
                      line=dict(width=2), marker=dict(size=10, symbol='square'),
                      hovertemplate='<b>Degree:</b> %{x}<br><b>MAE:</b> %{y:.4f}<extra></extra>'),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Polynomial Degree")
        fig.update_yaxes(title_text="R² Score", row=1, col=1)
        fig.update_yaxes(title_text="Error", row=1, col=2)
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Pembahasan
        st.subheader("Pembahasan")
        
        best_degree = df_comparison.loc[df_comparison['R²'].idxmax(), 'Degree']
        
        r2_warning = ""
        if errors_test['R2'] < 0:
            r2_warning = f"""> **Catatan Analisis:** Nilai $R^2$ yang negatif ({errors_test['R2']:.4f}) pada data testing menunjukkan bahwa model polinomial derajat {deg} tidak cocok (*poor fit*) untuk data ini. Hal ini mengindikasikan bahwa pola konsumsi listrik sangat fluktuatif sehingga fungsi polinomial sederhana tidak mampu menangkap kompleksitas perubahan data dalam jangka panjang."""

        st.markdown(f"""
        **Observasi:**
        - **Konteks Data**: Data mencakup durasi **168 jam (1 minggu)**. Satuan jam ini krusial karena penggunaan listrik memiliki siklus harian (24 jam).
        - **Derajat optimal** (berdasarkan $R^2$): {int(best_degree)} 
        - **$R^2$ Training**: {errors_train['R2']:.4f} 
        - **$R^2$ Testing**: {errors_test['R2']:.4f} 
        - **RMSE Testing**: {errors_test['RMSE']:.4f} kW 
        
        **Trade-off Bias-Variance:**
        - **Derajat rendah**: Underfitting (bias tinggi) 
        - **Derajat tinggi**: Overfitting (variance tinggi) 
        - **Derajat optimal** menyeimbangkan keduanya untuk mendapatkan error generalisasi terkecil. 
        
        **Interpretasi Koefisien:**
        - Koefisien positif pada $x^n$: Menunjukkan tren kenaikan konsumsi energi seiring waktu. 
        - Koefisien negatif: Menunjukkan tren penurunan konsumsi energi. 
        - Magnitude koefisien menunjukkan seberapa besar pengaruh variabel waktu terhadap perubahan power. 
        
        **Kesimpulan:**
        Regresi polinomial derajat {deg} berusaha menangkap tren umum penggunaan energi. Namun, karena durasi data mencapai 168 jam, model polinomial tunggal sering kali gagal mengikuti pola siklus harian yang tajam.
        Juga bisa dilihat Residual plot menunjukkan distribusi error yang {"bersifat acak (random)" if abs(np.mean(residual_test)) < 0.1 else "masih memiliki pola tertentu"}.
        
        {r2_warning}
        """)
    
    # ========================================
    # TAB 5: SUMMARY & COMPARISON
    # ========================================
    
    with tabs[4]:
        st.header("Summary & Overall Comparison")
        
        st.markdown("""
        ### Ringkasan Analisis Metode Numerik pada Data Konsumsi Energi
        
        Berikut adalah rangkuman lengkap dari semua metode yang telah dianalisis:
        """)
        
        # Summary metrics dari semua metode
        st.subheader("1. Integrasi Numerik - Total Energi")
        
        summary_integration = pd.DataFrame({
            'Metode': ['Simpson 1/3', 'Simpson 3/8', 'Trapezoidal', 'Richardson'],
            'Hasil (kWh)': [val_simp13, val_simp38, val_trap, val_richardson],
            'Relative Error (%)': [
                abs(val_simp13 - exact_val) / exact_val * 100,
                abs(val_simp38 - exact_val) / exact_val * 100,
                abs(val_trap - exact_val) / exact_val * 100,
                0.0,
            ],
            'Kompleksitas': ['O(n)', 'O(n)', 'O(n)', 'O(2n)'],
            'Rekomendasi': [
                'Terbaik (Presisi Tertinggi)', 
                'Hybrid (Setara Trapezoidal)', 
                'Stabil (Standar NumPy)', 
                'Benchmark (Referensi)'
            ]
        })
        # Update Narasi Summary Integrasi (TAB 5)
        st.subheader("Analisis Detail Integrasi")
        st.markdown(f"""
        **Evaluasi Komprehensif Metode Integrasi:**
        * **Optimasi Kuadratik (Simpson 1/3):** Metode ini tetap menjadi yang paling presisi dengan galat terendah yaitu **{abs(val_simp13 - exact_val) / exact_val * 100:.4f}%**. Hal ini membuktikan bahwa pendekatan parabola sangat efektif untuk memodelkan lengkungan fluktuasi konsumsi listrik mingguan rumah tangga.
        * **Analisis Fenomena Kembar (Trapezoidal vs Simpson 3/8):** Ditemukan bahwa metode Trapezoidal dan Simpson 3/8 menghasilkan nilai yang identik sebesar **{val_trap:.4f} kWh**. Fenomena ini secara numerik menunjukkan bahwa penambahan kompleksitas algoritma pada Simpson 3/8 tidak memberikan peningkatan akurasi jika pola data pada titik-titik interval tertentu lebih cenderung bersifat linear atau jika sisa interval pada metode hybrid didominasi oleh pendekatan linear.
        * **Validitas Algoritma:** Konsistensi hasil antara metode Trapezoidal manual dengan **NumPy Reference (np.trapezoid)** sebesar **{val_numpy:.2f} kWh** memvalidasi bahwa seluruh fungsi integrasi yang dibangun dalam proyek ini telah bekerja sesuai standar perhitungan saintifik.
        """)
        
        st.dataframe(summary_integration, use_container_width=True)
        # --- TAMBAHAN: Visualisasi Konsep Optimasi Kuadratik ---
        st.subheader("Visualisasi Konsep: Simpson 1/3 vs Trapezoidal")
        
        # Mengambil sampel kecil (3 titik) untuk menunjukkan kelengkungan
        # Kita cari indeks yang memiliki fluktuasi (misal indeks 50-52)
        sample_x = np.array([0, 1, 2])
        sample_y = power_minute[50:53]
        
        # Membuat titik halus untuk kurva parabola (Simpson)
        x_fine = np.linspace(0, 2, 100)
        # Polinomial derajat 2 (Parabola) yang melewati 3 titik
        poly_coef = np.polyfit(sample_x, sample_y, 2)
        y_fine_simpson = np.polyval(poly_coef, x_fine)

        fig_concept = go.Figure()

        # 1. Plot Titik Data Asli
        fig_concept.add_trace(go.Scatter(x=sample_x, y=sample_y, mode='markers+text', 
                                         name='Data Points', text=["P0", "P1", "P2"],
                                         textposition="top center", marker=dict(size=12, color='black')))

        # 2. Plot Pendekatan Trapezoidal (Garis Lurus)
        fig_concept.add_trace(go.Scatter(x=sample_x, y=sample_y, mode='lines', 
                                         name='Trapezoidal (Linear)', line=dict(color='blue', dash='dash')))

        # 3. Plot Pendekatan Simpson 1/3 (Kurva Parabola)
        fig_concept.add_trace(go.Scatter(x=x_fine, y=y_fine_simpson, mode='lines', 
                                         name='Simpson 1/3 (Quadratic)', line=dict(color='green', width=3)))

        fig_concept.update_layout(
            title="Bagaimana Simpson 1/3 Menangkap 'Lengkungan' Data",
            xaxis_title="Waktu (Menit)",
            yaxis_title="Power (kW)",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_concept, use_container_width=True)
        
        st.info("""
        **Insight Visual:** Garis putus-putus biru (Trapezoidal) memotong area secara lurus, sedangkan garis hijau (Simpson) melengkung mengikuti pola titik. 
        Pada data listrik yang fluktuatif, selisih area di bawah kurva inilah yang membuat Simpson 1/3 lebih akurat.
        """)
        
        st.subheader("2. Diferensiasi Numerik - Deteksi Anomali")
        
        summary_differentiation = pd.DataFrame({
            'Metode': ['Central', 'Forward', 'Backward'],
            'RMSE': [
                errors_methods['Central']['RMSE'],
                errors_methods['Forward']['RMSE'],
                errors_methods['Backward']['RMSE']
            ],
            'Anomali Detected': [len(anomali_data), '-', '-'],
            'Error Order': ['O(h²)', 'O(h)', 'O(h)'],
            'Rekomendasi': ['Terbaik', 'Boundary', 'Boundary']
        })
        
        st.dataframe(summary_differentiation, use_container_width=True)
        
        st.subheader("3. Interpolasi - Data Recovery")
        
        # Hitung ulang interpolasi untuk summary
        missing_indices_summary = list(range(50, 56))
        window_size_summary = 100
        
        x_full_summary = np.arange(min(window_size_summary, len(power_minute)))
        y_full_summary = power_minute[:len(x_full_summary)].copy()
        
        valid_missing_summary = [i for i in missing_indices_summary if i < len(x_full_summary)]
        
        if len(valid_missing_summary) > 0:
            x_sample_summary = np.delete(x_full_summary, valid_missing_summary)
            y_sample_summary = np.delete(y_full_summary, valid_missing_summary)
            
            y_true_summary = [y_full_summary[i] for i in valid_missing_summary]
            
            # Lagrange interpolation
            y_lagrange_summary = [lagrange_interpolation(x_sample_summary, y_sample_summary, i) for i in valid_missing_summary]
            
            # Cubic Spline
            spline_coef_summary = cubic_spline_coefficients(x_sample_summary, y_sample_summary)
            y_spline_summary = evaluate_cubic_spline(spline_coef_summary, np.array(valid_missing_summary))
            
            # Hitung error
            errors_lagrange_summary = calculate_errors(y_true_summary, y_lagrange_summary)
            errors_spline_summary = calculate_errors(y_true_summary, y_spline_summary) 
    
            summary_interpolation = pd.DataFrame({
                'Metode': ['Lagrange', 'Cubic Spline'],
                'MAE (kW)': [errors_lagrange_summary['MAE'], errors_spline_summary['MAE']],
                'RMSE (kW)': [errors_lagrange_summary['RMSE'], errors_spline_summary['RMSE']],
                'MAPE (%)': [errors_lagrange_summary['MAPE'], errors_spline_summary['MAPE']],
                'Komputasi': ['Lebih Lambat', 'Moderate'],
                'Rekomendasi': ['Teoritis', 'Terbaik']
            })
        else:
            summary_interpolation = pd.DataFrame({
                'Metode': ['Lagrange', 'Cubic Spline'],
                'MAE (kW)': ['N/A', 'N/A'],
                'RMSE (kW)': ['N/A', 'N/A'],
                'MAPE (%)': ['N/A', 'N/A'],
                'Komputasi': ['Lebih Lambat', 'Moderate'],
                'Rekomendasi': ['Teoritis', 'Terbaik']
            })
        
        st.dataframe(summary_interpolation, use_container_width=True)
        
        st.subheader("4. Regresi Polinomial - Trend Analysis")
        
        summary_regression = pd.DataFrame({
            'Metode': [f'Polynomial Deg {deg}'],
            'R² Training': [errors_train['R2']],
            'R² Testing': [errors_test['R2']],
            'RMSE Test (kW)': [errors_test['RMSE']],
            'MAE Test (kW)': [errors_test['MAE']],
            'Overfitting Risk': ['Medium' if abs(errors_train['R2'] - errors_test['R2']) < 0.1 else 'High']
        })
        
        st.dataframe(summary_regression, use_container_width=True)
        
        # Overall Comparison Chart dengan Plotly
        st.subheader("Visualisasi Perbandingan Overall")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Integrasi: Error Comparison', 
                          'Diferensiasi: RMSE Comparison',
                          'Interpolasi: Error Distribution', 
                          'Regresi: R² vs Degree'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "box"}, {"type": "scatter"}]]
        )
        
        # 1. Integrasi Methods Comparison (Update List)
        int_methods = ['Trapezoidal', 'Simpson 1/3', 'Simpson 3/8', 'Richardson']
        int_errors = [
            abs(val_trap - exact_val) / exact_val * 100,
            abs(val_simp13 - exact_val) / exact_val * 100,
            abs(val_simp38 - exact_val) / exact_val * 100,
            0.0
        ]
        
        fig.add_trace(
            go.Bar(y=int_methods, x=int_errors, orientation='h',
                  marker_color=['blue', 'green', 'magenta', 'orange'], # Tambah warna magenta untuk 3/8
                  hovertemplate='<b>%{y}</b><br>Error: %{x:.4f}%<extra></extra>',
                  showlegend=False),
            row=1, col=1
        )
        
        # 2. Diferensiasi Methods Comparison
        diff_methods = ['Forward', 'Backward', 'Central']
        diff_rmse = [
            errors_methods['Forward']['RMSE'],
            errors_methods['Backward']['RMSE'],
            errors_methods['Central']['RMSE']
        ]
        
        fig.add_trace(
            go.Bar(x=diff_methods, y=diff_rmse,
                  marker_color=['orange', 'blue', 'green'],
                  hovertemplate='<b>%{x}</b><br>RMSE: %{y:.4f}<extra></extra>',
                  showlegend=False),
            row=1, col=2
        )
        
        # 3. Interpolasi Error Distribution
        if len(valid_missing_summary) > 0:
            lagrange_errors = np.abs(np.array(y_true_summary) - np.array(y_lagrange_summary))
            spline_errors = np.abs(np.array(y_true_summary) - np.array(y_spline_summary))
            
            fig.add_trace(
                go.Box(y=lagrange_errors, name='Lagrange',
                      marker_color='orange',
                      hovertemplate='<b>Lagrange</b><br>Error: %{y:.4f} kW<extra></extra>'),
                row=2, col=1
            )
            fig.add_trace(
                go.Box(y=spline_errors, name='Spline',
                      marker_color='blue',
                      hovertemplate='<b>Spline</b><br>Error: %{y:.4f} kW<extra></extra>'),
                row=2, col=1
            )
        
        # 4. Regresi R² Comparison
        fig.add_trace(
            go.Scatter(x=df_comparison['Degree'], y=df_comparison['R²'],
                      mode='lines+markers', name='R²',
                      line=dict(width=2, color='#66b3ff'),
                      marker=dict(size=10),
                      hovertemplate='<b>Degree:</b> %{x}<br><b>R²:</b> %{y:.4f}<extra></extra>',
                      showlegend=False),
            row=2, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text="Relative Error (%)", row=1, col=1)
        fig.update_yaxes(title_text="RMSE", row=1, col=2)
        fig.update_yaxes(title_text="Absolute Error (kW)", row=2, col=1)
        fig.update_xaxes(title_text="Polynomial Degree", row=2, col=2)
        fig.update_yaxes(title_text="R² Score", row=2, col=2)
        
        fig.update_layout(height=900)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Final Recommendations
        st.subheader("Rekomendasi Akhir")
        
        st.markdown("""
        ### Kesimpulan dan Rekomendasi
        
        Berdasarkan analisis komprehensif terhadap data konsumsi energi listrik:
        
        **1. Integrasi Numerik:**
        - **Rekomendasi:** Simpson 1/3 atau Richardson Extrapolation
        - **Alasan:** Balance antara akurasi dan efisiensi komputasi
        - **Use Case:** Perhitungan total konsumsi energi, billing estimation
        
        **2. Diferensiasi Numerik:**
        - **Rekomendasi:** Central Difference
        - **Alasan:** Error order O(h²), lebih akurat dari forward/backward
        - **Use Case:** Deteksi anomali, monitoring perubahan mendadak
        
        **3. Interpolasi:**
        - **Rekomendasi:** Cubic Spline
        - **Alasan:** Menghindari oscillation, smooth, dan akurat
        - **Use Case:** Recovery missing data, resampling
        
        **4. Regresi Polinomial:**
        - **Rekomendasi:** Degree 3-4 (trade-off bias-variance)
        - **Alasan:** Menangkap tren tanpa overfitting
        - **Use Case:** Forecasting, trend analysis, anomaly baseline
        
        ### Kontribusi terhadap Analisis Energi:
        
        1. **Akurasi Billing:** Metode integrasi meningkatkan akurasi perhitungan konsumsi
        2. **Real-time Monitoring:** Diferensiasi mendeteksi anomali untuk preventive maintenance
        3. **Data Quality:** Interpolasi memulihkan data sensor yang corrupt/missing
        4. **Planning:** Regresi membantu forecasting untuk capacity planning
        
        ### Limitasi dan Future Work:
        
        - Metode numerik sensitif terhadap noise → Perlu pre-processing (filtering)
        - Interpolasi polinomial tinggi → Runge's phenomenon
        - Regresi degree tinggi → Overfitting risk
        - **Saran:** Eksplorasi metode adaptif, wavelet analysis, atau machine learning
        """)
        
        st.success("Analisis selesai! Semua metode telah dievaluasi dengan komprehensif menggunakan visualisasi interaktif.")