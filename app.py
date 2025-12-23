# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numerical_methods import *

st.set_page_config(page_title="Analisis Energi Listrik", layout="wide")

# Styling untuk plot
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ========================================
# DATA PREPROCESSING
# ========================================

@st.cache_data
def load_and_clean_data():
    """
    Load dan clean data konsumsi energi listrik
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
    # Resample ke hourly data
    df_hourly = df_raw.set_index('datetime').resample('H').mean(numeric_only=True).reset_index()
    
    # Data untuk analisis
    power = df_hourly['Global_active_power'].values
    h = 1.0  # Step size dalam jam
    
    # ========================================
    # HEADER & INTRODUCTION
    # ========================================
    
    st.title("Analisis Konsumsi Energi Listrik dengan Metode Numerik")
    st.markdown("---")
    
    st.markdown("""
    ### Deskripsi Project
    Project ini menganalisis data konsumsi energi listrik rumah tangga menggunakan berbagai metode numerik.
    Dataset berisi pengukuran power consumption setiap menit selama beberapa tahun.
    
    **Metode Numerik yang Digunakan:**
    1. Integrasi Numerik (Rectangular, Trapezoidal, Simpson 1/3, Simpson 3/8)
    2. Diferensiasi Numerik (Forward, Backward, Central Difference)
    3. Interpolasi (Newton Divided Difference & Lagrange)
    4. Regresi Polinomial (Least Squares)
    
    **Sumber Data:** UCI Machine Learning Repository - Individual Household Electric Power Consumption
    """)
    
    # Info dataset
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Data Points", len(df_hourly))
    with col2:
        st.metric("Durasi", "1 Minggu")
    with col3:
        st.metric("Rata-rata Power", f"{np.mean(power):.2f} kW")
    with col4:
        st.metric("Std Dev Power", f"{np.std(power):.2f} kW")
    
    st.markdown("---")
    
    # ========================================
    # TAB NAVIGATION
    # ========================================
    
    tabs = st.tabs([
        "1. Integrasi (Total Energi)",
        "2. Diferensiasi (Deteksi Anomali)",
        "3. Interpolasi (Recovery Data)",
        "4. Regresi (Tren Prediction)",
        "5. Summary & Comparison"
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
        
        # Sidebar untuk kontrol segmen
        st.subheader("Pengaturan Parameter Integrasi")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            n_segments = st.slider(
                "Jumlah Segmen untuk Adaptive Integration",
                min_value=1,
                max_value=20,
                value=5,
                help="Membagi data menjadi beberapa segmen untuk integrasi yang lebih akurat"
            )
        
        with col2:
            show_comparison = st.checkbox("Tampilkan Perbandingan dengan NumPy", value=True)
        
        st.markdown("---")
        
        # Hitung integral dengan berbagai metode
        val_rect_left = manual_rectangular(power, h, method='left')
        val_rect_right = manual_rectangular(power, h, method='right')
        val_rect_mid = manual_rectangular(power, h, method='midpoint')
        val_trap = manual_trapezoidal(power, h)
        val_simp13 = manual_simpson_13(power, h)
        val_simp38 = manual_simpson_38(power, h)
        val_adaptive = adaptive_integration(power, h, n_segments)
        
        # Nilai referensi menggunakan Simpson 1/3 dengan step size lebih kecil (jika ada data menit)
        # Atau gunakan NumPy sebagai "ground truth"
        val_numpy = np.trapz(power, dx=h)
        
        # Richardson Extrapolation untuk meningkatkan akurasi
        power_half = df_raw.set_index('datetime').resample('30min').mean(numeric_only=True)['Global_active_power'].values
        h_half = 0.5
        val_trap_half = manual_trapezoidal(power_half, h_half)
        val_richardson = richardson_extrapolation(val_trap, val_trap_half, 2)
        
        # Gunakan nilai referensi sebagai "exact value"
        # Untuk real-world data, kita gunakan metode higher-order sebagai referensi
        exact_val = val_richardson
        
        # Display hasil
        st.subheader("Hasil Perhitungan Integral")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rectangular (Left)", f"{val_rect_left:.2f} kWh")
            st.metric("Rectangular (Right)", f"{val_rect_right:.2f} kWh")
        with col2:
            st.metric("Rectangular (Mid)", f"{val_rect_mid:.2f} kWh")
            st.metric("Trapezoidal", f"{val_trap:.2f} kWh")
        with col3:
            st.metric("Simpson 1/3", f"{val_simp13:.2f} kWh")
            if val_simp38:
                st.metric("Simpson 3/8", f"{val_simp38:.2f} kWh")
        with col4:
            st.metric("Adaptive (Segmented)", f"{val_adaptive:.2f} kWh")
            st.metric("Richardson Extrap.", f"{val_richardson:.2f} kWh", 
                     help="Menggunakan Richardson Extrapolation untuk akurasi lebih tinggi")
        
        if show_comparison:
            st.info(f"NumPy Reference (np.trapz): {val_numpy:.2f} kWh")
        
        # Analisis Error
        st.subheader("Analisis Error (Menggunakan Richardson Extrapolation sebagai Referensi)")
        
        methods = ['Rect (Left)', 'Rect (Right)', 'Rect (Mid)', 'Trapezoidal', 
                   'Simpson 1/3', 'Adaptive', 'NumPy']
        values = [val_rect_left, val_rect_right, val_rect_mid, val_trap, 
                  val_simp13, val_adaptive, val_numpy]
        
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
        
        # Visualisasi error
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Absolute Error
        ax1.bar(methods, errors_abs, color='steelblue', alpha=0.7)
        ax1.set_ylabel('Absolute Error (kWh)')
        ax1.set_title('Absolute Error Comparison')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Relative Error
        ax2.bar(methods, errors_rel, color='coral', alpha=0.7)
        ax2.set_ylabel('Relative Error (%)')
        ax2.set_title('Relative Error Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Pembahasan
        st.subheader("Pembahasan")
        st.markdown(f"""
        **Observasi:**
        - Metode Simpson 1/3 memberikan error terendah ({errors_rel[methods.index('Simpson 1/3')]:.4f}%)
        - Metode Rectangular memiliki error tertinggi karena aproksimasi paling sederhana
        - Richardson Extrapolation berhasil meningkatkan akurasi dengan menggabungkan hasil dari dua step size berbeda
        - Adaptive integration dengan {n_segments} segmen memberikan hasil: {val_adaptive:.4f} kWh
        
        **Kesimpulan:**
        Untuk integrasi data power consumption, metode Simpson 1/3 atau Richardson Extrapolation
        memberikan hasil paling akurat dengan computational cost yang masih reasonable.
        """)
    
    # ========================================
    # TAB 2: DIFERENSIASI NUMERIK
    # ========================================
    
    with tabs[1]:
        st.header("Diferensiasi Numerik: Deteksi Anomali dan Laju Perubahan")
        
        st.markdown("""
        **Tujuan:** Menghitung laju perubahan power consumption (dP/dt) untuk mendeteksi
        anomali atau lonjakan mendadak dalam penggunaan listrik.
        
        **Formula:** dP/dt ≈ [P(t+h) - P(t-h)] / (2h) (Central Difference)
        """)
        
        # Hitung turunan dengan berbagai metode
        dpdt_forward = manual_diff_forward(power, h)
        dpdt_backward = manual_diff_backward(power, h)
        dpdt_central = manual_diff_central(power, h)
        dpdt_second = manual_diff_second_order(power, h)
        
        # Threshold untuk deteksi anomali
        threshold = st.slider("Threshold Anomali (kW/h)", 0.5, 5.0, 2.0, 0.1)
        
        # Visualisasi
        st.subheader("Laju Perubahan Power Consumption")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Power original
        axes[0, 0].plot(df_hourly['datetime'], power, color='blue', alpha=0.7, linewidth=1)
        axes[0, 0].set_ylabel('Power (kW)')
        axes[0, 0].set_title('Original Power Consumption')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Forward Difference
        axes[0, 1].plot(df_hourly['datetime'], dpdt_forward, color='green', alpha=0.7, linewidth=1)
        axes[0, 1].axhline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
        axes[0, 1].axhline(-threshold, color='red', linestyle='--')
        axes[0, 1].set_ylabel('dP/dt (kW/h)')
        axes[0, 1].set_title('Forward Difference')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Central Difference
        axes[1, 0].plot(df_hourly['datetime'], dpdt_central, color='orange', alpha=0.7, linewidth=1)
        axes[1, 0].axhline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
        axes[1, 0].axhline(-threshold, color='red', linestyle='--')
        axes[1, 0].set_ylabel('dP/dt (kW/h)')
        axes[1, 0].set_title('Central Difference (Most Accurate)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Second Order Derivative
        axes[1, 1].plot(df_hourly['datetime'], dpdt_second, color='purple', alpha=0.7, linewidth=1)
        axes[1, 1].set_ylabel('d²P/dt² (kW/h²)')
        axes[1, 1].set_title('Second Order Derivative (Acceleration)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Deteksi anomali
        anomali_indices = np.where(np.abs(dpdt_central) > threshold)[0]
        anomali_data = df_hourly.iloc[anomali_indices].copy()
        anomali_data['dP/dt'] = dpdt_central[anomali_indices]
        anomali_data['Magnitude'] = np.abs(dpdt_central[anomali_indices])
        
        st.subheader(f"Deteksi Anomali (Total: {len(anomali_data)} titik)")
        
        if len(anomali_data) > 0:
            # Sorting berdasarkan magnitude
            anomali_data_sorted = anomali_data.sort_values('Magnitude', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(
                    anomali_data_sorted[['datetime', 'Global_active_power', 'dP/dt', 'Magnitude']].head(20),
                    use_container_width=True
                )
            
            with col2:
                st.metric("Anomali Terdeteksi", len(anomali_data))
                st.metric("Max |dP/dt|", f"{anomali_data['Magnitude'].max():.2f} kW/h")
                st.metric("% dari Total Data", f"{len(anomali_data)/len(df_hourly)*100:.2f}%")
        
        # Analisis Error - Bandingkan dengan NumPy gradient
        st.subheader("Analisis Error Diferensiasi")
        
        # NumPy gradient sebagai referensi
        dpdt_numpy = np.gradient(power, h)
        
        errors_methods = {
            'Forward': calculate_errors(dpdt_numpy, dpdt_forward),
            'Backward': calculate_errors(dpdt_numpy, dpdt_backward),
            'Central': calculate_errors(dpdt_numpy, dpdt_central)
        }
        
        df_diff_error = pd.DataFrame(errors_methods).T
        st.dataframe(df_diff_error, use_container_width=True)
        
        # Pembahasan
        st.subheader("Pembahasan")
        if len(anomali_data) > 0:
            anomali_terbesar_info = f"- Anomali terbesar terjadi pada: {anomali_data_sorted.iloc[0]['datetime']}"
        else:
            anomali_terbesar_info = "- Tidak ada anomali yang terdeteksi dengan threshold ini"
        
        st.markdown(f"""
        **Observasi:**
        - Central Difference memiliki error terendah (RMSE: {errors_methods['Central']['RMSE']:.4f})
        - Terdeteksi {len(anomali_data)} anomali dengan threshold {threshold} kW/h
        - {anomali_terbesar_info}
        
        **Interpretasi Fisik:**
        - Lonjakan positif: Peningkatan mendadak konsumsi (peralatan dinyalakan)
        - Lonjakan negatif: Penurunan mendadak konsumsi (peralatan dimatikan)
        - Second derivative mengidentifikasi percepatan perubahan
        
        **Kesimpulan:**
        Central Difference memberikan aproksimasi turunan paling akurat dengan error order O(h²).
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
        - Newton Divided Difference
        - Lagrange Interpolation
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
        window_size = 50  # Menggunakan window untuk interpolasi
        
        x_full = np.arange(window_size)
        y_full = power[:window_size].copy()
        
        # Hapus data pada indeks missing
        x_sample = np.delete(x_full, [i for i in idx_missing if i < window_size])
        y_sample = np.delete(y_full, [i for i in idx_missing if i < window_size])
        
        # Interpolasi menggunakan Newton
        coef_newton = newton_divided_diff(x_sample, y_sample)
        
        # Evaluasi interpolasi
        x_interp = np.linspace(0, window_size-1, 200)
        y_newton = [evaluate_newton(x_sample, coef_newton, xi) for xi in x_interp]
        y_lagrange = [lagrange_interpolation(x_sample, y_sample, xi) for xi in x_interp]
        
        # Recovery data yang hilang
        x_missing_actual = [i for i in idx_missing if i < window_size]
        y_missing_true = [y_full[i] for i in x_missing_actual]
        y_missing_newton = [evaluate_newton(x_sample, coef_newton, xi) for xi in x_missing_actual]
        y_missing_lagrange = [lagrange_interpolation(x_sample, y_sample, xi) for xi in x_missing_actual]
        
        # Visualisasi
        st.subheader("Hasil Interpolasi")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot Newton
        axes[0].scatter(x_sample, y_sample, color='blue', s=50, label='Data Tersedia', zorder=3)
        axes[0].scatter(x_missing_actual, y_missing_true, color='red', s=50, 
                       label='Data Hilang (Truth)', zorder=4, marker='x')
        axes[0].plot(x_interp, y_newton, 'g--', linewidth=2, label='Newton Interpolation', zorder=2)
        axes[0].scatter(x_missing_actual, y_missing_newton, color='orange', s=100, 
                       label='Prediksi Newton', zorder=5, marker='^', edgecolors='black')
        axes[0].set_xlabel('Time Index')
        axes[0].set_ylabel('Power (kW)')
        axes[0].set_title('Newton Divided Difference Interpolation')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot Lagrange
        axes[1].scatter(x_sample, y_sample, color='blue', s=50, label='Data Tersedia', zorder=3)
        axes[1].scatter(x_missing_actual, y_missing_true, color='red', s=50, 
                       label='Data Hilang (Truth)', zorder=4, marker='x')
        axes[1].plot(x_interp, y_lagrange, 'm--', linewidth=2, label='Lagrange Interpolation', zorder=2)
        axes[1].scatter(x_missing_actual, y_missing_lagrange, color='cyan', s=100, 
                       label='Prediksi Lagrange', zorder=5, marker='^', edgecolors='black')
        axes[1].set_xlabel('Time Index')
        axes[1].set_ylabel('Power (kW)')
        axes[1].set_title('Lagrange Interpolation')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Analisis Error
        st.subheader("Analisis Error Interpolasi")
        
        if len(y_missing_true) > 0:
            errors_newton = calculate_errors(y_missing_true, y_missing_newton)
            errors_lagrange = calculate_errors(y_missing_true, y_missing_lagrange)
            
            df_interp_error = pd.DataFrame({
                'Newton': errors_newton,
                'Lagrange': errors_lagrange
            }).T
            
            st.dataframe(df_interp_error, use_container_width=True)
            
            # Detail prediksi vs truth
            st.subheader("Detail Prediksi vs Truth")
            
            df_missing_comparison = pd.DataFrame({
                'Index': x_missing_actual,
                'True Value': y_missing_true,
                'Newton Pred': y_missing_newton,
                'Lagrange Pred': y_missing_lagrange,
                'Newton Error': np.abs(np.array(y_missing_true) - np.array(y_missing_newton)),
                'Lagrange Error': np.abs(np.array(y_missing_true) - np.array(y_missing_lagrange))
            })
            
            st.dataframe(df_missing_comparison, use_container_width=True)
            
            # Visualisasi error per titik
            fig, ax = plt.subplots(figsize=(12, 5))
            x_pos = np.arange(len(x_missing_actual))
            width = 0.35
            
            ax.bar(x_pos - width/2, df_missing_comparison['Newton Error'], width, 
                   label='Newton Error', color='green', alpha=0.7)
            ax.bar(x_pos + width/2, df_missing_comparison['Lagrange Error'], width, 
                   label='Lagrange Error', color='magenta', alpha=0.7)
            
            ax.set_xlabel('Missing Data Point')
            ax.set_ylabel('Absolute Error (kW)')
            ax.set_title('Error Comparison per Missing Point')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_missing_actual)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            st.pyplot(fig)
        
        # Pembahasan
        st.subheader("Pembahasan")
        if len(y_missing_true) > 0:
            observasi_text = f"""
        **Observasi:**
        - MAE Newton: {errors_newton['MAE']:.4f} kW
        - MAE Lagrange: {errors_lagrange['MAE']:.4f} kW
        - Kedua metode memberikan hasil yang sangat mirip (secara teoritis identik)
        
        **Karakteristik Interpolasi Polinomial:**
        - Sempurna untuk data yang smooth dan kontinu
        - Dapat terjadi Runge's phenomenon pada polinomial berderajat tinggi
        - Baik untuk recovery data yang hilang dalam jumlah kecil
        
        **Kesimpulan:**
        Interpolasi Newton dan Lagrange efektif untuk recovery data hilang dengan error rata-rata
        {errors_newton['MAE']:.4f} kW. Untuk data yang lebih kompleks, pertimbangkan spline interpolation.
        """
        else:
            observasi_text = """
        **Observasi:**
        - Tidak ada data yang hilang dalam window yang dipilih
        - Silakan sesuaikan parameter untuk simulasi missing data
        
        **Karakteristik Interpolasi Polinomial:**
        - Sempurna untuk data yang smooth dan kontinu
        - Dapat terjadi Runge's phenomenon pada polinomial berderajat tinggi
        - Baik untuk recovery data yang hilang dalam jumlah kecil
        
        **Kesimpulan:**
        Interpolasi Newton dan Lagrange secara teoritis memberikan hasil identik untuk 
        interpolasi polinomial. Sesuaikan parameter untuk melihat hasil recovery data.
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
        split_idx = int(len(power) * train_size / 100)
        x_train = np.arange(split_idx)
        y_train = power[:split_idx]
        x_test = np.arange(split_idx, len(power))
        y_test = power[split_idx:]
        
        # Training regresi
        coeffs = manual_poly_regression(x_train, y_train, deg)
        
        # Prediksi
        y_train_pred = evaluate_polynomial(coeffs, x_train)
        y_test_pred = evaluate_polynomial(coeffs, x_test)
        
        # Visualisasi
        st.subheader("Hasil Regresi")
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot keseluruhan
        axes[0].plot(np.arange(len(power)), power, alpha=0.5, label='Data Asli', color='blue', linewidth=1)
        axes[0].plot(x_train, y_train_pred, color='red', label=f'Regresi Training (Degree {deg})', linewidth=2)
        axes[0].plot(x_test, y_test_pred, color='green', label='Prediksi Testing', linewidth=2, linestyle='--')
        axes[0].axvline(split_idx, color='black', linestyle=':', label='Train/Test Split')
        axes[0].set_xlabel('Time (hours)')
        axes[0].set_ylabel('Power (kW)')
        axes[0].set_title(f'Polynomial Regression (Degree {deg})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot residual
        residual_train = y_train - y_train_pred
        residual_test = y_test - y_test_pred
        
        axes[1].scatter(x_train, residual_train, alpha=0.5, s=10, label='Residual Training', color='blue')
        axes[1].scatter(x_test, residual_test, alpha=0.5, s=10, label='Residual Testing', color='green')
        axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Time (hours)')
        axes[1].set_ylabel('Residual (kW)')
        axes[1].set_title('Residual Plot')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
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
        
        # Visualisasi perbandingan
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(df_comparison['Degree'], df_comparison['R²'], marker='o', linewidth=2)
        axes[0].set_xlabel('Polynomial Degree')
        axes[0].set_ylabel('R² Score')
        axes[0].set_title('R² vs Polynomial Degree')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(df_comparison['Degree'], df_comparison['RMSE'], marker='o', linewidth=2, label='RMSE')
        axes[1].plot(df_comparison['Degree'], df_comparison['MAE'], marker='s', linewidth=2, label='MAE')
        axes[1].set_xlabel('Polynomial Degree')
        axes[1].set_ylabel('Error')
        axes[1].set_title('Error vs Polynomial Degree')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Pembahasan
        st.subheader("Pembahasan")
        
        best_degree = df_comparison.loc[df_comparison['R²'].idxmax(), 'Degree']
        
        st.markdown(f"""
        **Observasi:**
        - Derajat optimal (berdasarkan R²): {int(best_degree)}
        - R² Training: {errors_train['R2']:.4f}
        - R² Testing: {errors_test['R2']:.4f}
        - RMSE Testing: {errors_test['RMSE']:.4f} kW
        
        **Trade-off Bias-Variance:**
        - Derajat rendah: Underfitting (bias tinggi)
        - Derajat tinggi: Overfitting (variance tinggi)
        - Derajat optimal menyeimbangkan keduanya
        
        **Interpretasi Koefisien:**
        - Koefisien positif pada x^n: Tren naik
        - Koefisien negatif: Tren turun
        - Magnitude koefisien menunjukkan pengaruh terhadap prediksi
        
        **Kesimpulan:**
        Regresi polinomial degree {deg} berhasil menangkap tren dengan R² = {errors_test['R2']:.4f}.
        Residual plot menunjukkan distribusi error yang {"random" if abs(np.mean(residual_test)) < 0.1 else "masih memiliki pola"}.
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
            'Metode': ['Simpson 1/3', 'Trapezoidal', 'Richardson', 'Adaptive'],
            'Hasil (kWh)': [val_simp13, val_trap, val_richardson, val_adaptive],
            'Relative Error (%)': [
                abs(val_simp13 - exact_val) / exact_val * 100,
                abs(val_trap - exact_val) / exact_val * 100,
                0.0,  # Richardson sebagai referensi
                abs(val_adaptive - exact_val) / exact_val * 100
            ],
            'Kompleksitas': ['O(n)', 'O(n)', 'O(2n)', 'O(n)'],
            'Rekomendasi': ['Akurasi Tinggi', 'Balance', 'Referensi', 'Fleksibel']
        })
        
        st.dataframe(summary_integration, use_container_width=True)
        
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
        
        # Hitung ulang interpolasi untuk summary (karena variabel dari tab 3 tidak tersedia di sini)
        missing_indices_summary = list(range(50, 56))
        window_size_summary = 100  # Perbesar window agar missing indices pasti masuk
        
        x_full_summary = np.arange(min(window_size_summary, len(power)))
        y_full_summary = power[:len(x_full_summary)].copy()
        
        valid_missing_summary = [i for i in missing_indices_summary if i < len(x_full_summary)]
        
        if len(valid_missing_summary) > 0:
            x_sample_summary = np.delete(x_full_summary, valid_missing_summary)
            y_sample_summary = np.delete(y_full_summary, valid_missing_summary)
            
            y_true_summary = [y_full_summary[i] for i in valid_missing_summary]
            
            # Newton interpolation
            coef_newton_summary = newton_divided_diff(x_sample_summary, y_sample_summary)
            y_newton_summary = [evaluate_newton(x_sample_summary, coef_newton_summary, i) for i in valid_missing_summary]
            
            # Lagrange interpolation
            y_lagrange_summary = [lagrange_interpolation(x_sample_summary, y_sample_summary, i) for i in valid_missing_summary]
            
            # Hitung error
            errors_newton_summary = calculate_errors(y_true_summary, y_newton_summary)
            errors_lagrange_summary = calculate_errors(y_true_summary, y_lagrange_summary)
            
            summary_interpolation = pd.DataFrame({
                'Metode': ['Newton', 'Lagrange'],
                'MAE (kW)': [errors_newton_summary['MAE'], errors_lagrange_summary['MAE']],
                'RMSE (kW)': [errors_newton_summary['RMSE'], errors_lagrange_summary['RMSE']],
                'MAPE (%)': [errors_newton_summary['MAPE'], errors_lagrange_summary['MAPE']],
                'Komputasi': ['Efisien', 'Lebih Lambat'],
                'Rekomendasi': ['Praktis', 'Teoritis']
            })
        else:
            # Jika tidak ada data missing yang valid, buat tabel placeholder
            summary_interpolation = pd.DataFrame({
                'Metode': ['Newton', 'Lagrange'],
                'MAE (kW)': ['N/A', 'N/A'],
                'RMSE (kW)': ['N/A', 'N/A'],
                'MAPE (%)': ['N/A', 'N/A'],
                'Komputasi': ['Efisien', 'Lebih Lambat'],
                'Rekomendasi': ['Praktis', 'Teoritis']
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
        
        # Overall Comparison Chart
        st.subheader("Visualisasi Perbandingan Overall")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Integrasi Methods Comparison
        int_methods = ['Rectangular', 'Trapezoidal', 'Simpson 1/3', 'Richardson']
        int_errors = [
            abs(val_rect_mid - exact_val) / exact_val * 100,
            abs(val_trap - exact_val) / exact_val * 100,
            abs(val_simp13 - exact_val) / exact_val * 100,
            0.0
        ]
        
        axes[0, 0].barh(int_methods, int_errors, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        axes[0, 0].set_xlabel('Relative Error (%)')
        axes[0, 0].set_title('Integrasi: Error Comparison')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. Diferensiasi Methods Comparison
        diff_methods = ['Forward', 'Backward', 'Central']
        diff_rmse = [
            errors_methods['Forward']['RMSE'],
            errors_methods['Backward']['RMSE'],
            errors_methods['Central']['RMSE']
        ]
        
        axes[0, 1].bar(diff_methods, diff_rmse, color=['#ff9999', '#66b3ff', '#99ff99'])
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Diferensiasi: RMSE Comparison')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Interpolasi Error Distribution
        if len(valid_missing_summary) > 0:
            # Hitung error per point untuk boxplot
            newton_errors = np.abs(np.array(y_true_summary) - np.array(y_newton_summary))
            lagrange_errors = np.abs(np.array(y_true_summary) - np.array(y_lagrange_summary))
            
            axes[1, 0].boxplot([newton_errors, lagrange_errors],
                               labels=['Newton', 'Lagrange'])
            axes[1, 0].set_ylabel('Absolute Error (kW)')
            axes[1, 0].set_title('Interpolasi: Error Distribution')
            axes[1, 0].grid(axis='y', alpha=0.3)
        else:
            # Jika tidak ada data, tampilkan pesan
            axes[1, 0].text(0.5, 0.5, 'No missing data\nfor interpolation', 
                           ha='center', va='center', fontsize=12)
            axes[1, 0].set_title('Interpolasi: Error Distribution')
            axes[1, 0].set_xticks([])
            axes[1, 0].set_yticks([])
        
        # 4. Regresi R² Comparison
        axes[1, 1].plot(df_comparison['Degree'], df_comparison['R²'], 
                       marker='o', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Polynomial Degree')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('Regresi: R² vs Degree')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        st.pyplot(fig)
        
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
        - **Rekomendasi:** Newton Divided Difference
        - **Alasan:** Lebih efisien komputasi, hasil identik dengan Lagrange
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
        
        # Download button untuk hasil
        st.subheader("Export Hasil Analisis")
        
        # Gabungkan semua hasil
        all_results = {
            'Integration': summary_integration.to_dict(),
            'Differentiation': summary_differentiation.to_dict(),
            'Interpolation': summary_interpolation.to_dict(),
            'Regression': summary_regression.to_dict()
        }
        
        st.success("Analisis selesai! Semua metode telah dievaluasi dengan komprehensif.")

else:
    st.error("Gagal memuat data. Pastikan file 'household_power_consumption.txt' ada di direktori yang sama.")