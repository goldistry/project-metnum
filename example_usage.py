# example_usage.py
"""
Contoh penggunaan lengkap untuk analisis data real dengan metode numerik
Dapat dijalankan secara standalone tanpa Streamlit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numerical_methods import *
from error_analysis import ErrorAnalyzer

def load_sample_data():
    """
    Load dan preprocess data konsumsi energi
    """
    print("Loading data...")
    try:
        df = pd.read_csv('household_power_consumption.txt', sep=';', low_memory=False)
        
        # Preprocessing
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
        df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
        df = df.dropna(subset=['Global_active_power'])
        
        # Ambil 1 minggu data
        df_subset = df.head(10080).copy()
        
        # Resample ke hourly
        df_hourly = df_subset.set_index('datetime').resample('H').mean(numeric_only=True).reset_index()
        
        print(f"Data loaded: {len(df_hourly)} hourly data points")
        return df_hourly
        
    except FileNotFoundError:
        print("File not found. Generating synthetic data for demonstration...")
        # Generate synthetic data jika file tidak ada
        hours = 168  # 1 minggu
        time = pd.date_range('2024-01-01', periods=hours, freq='H')
        
        # Simulate realistic power consumption pattern
        base = 1.5
        daily_pattern = 0.8 * np.sin(2 * np.pi * np.arange(hours) / 24)
        weekly_trend = 0.3 * np.sin(2 * np.pi * np.arange(hours) / 168)
        noise = np.random.normal(0, 0.2, hours)
        
        power = base + daily_pattern + weekly_trend + noise
        power = np.maximum(power, 0.5)  # Minimum power
        
        df = pd.DataFrame({
            'datetime': time,
            'Global_active_power': power
        })
        
        print(f"Synthetic data generated: {len(df)} hourly data points")
        return df

def example_1_integration_analysis():
    """
    Contoh 1: Analisis Integrasi - Menghitung Total Energi
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: INTEGRATION ANALYSIS - Total Energy Calculation")
    print("=" * 80)
    
    df = load_sample_data()
    power = df['Global_active_power'].values
    h = 1.0  # hourly data
    
    print("\n1. Computing integral using different methods...")
    
    # Hitung dengan berbagai metode
    methods = {
        'Rectangular (Left)': manual_rectangular(power, h, method='left'),
        'Rectangular (Right)': manual_rectangular(power, h, method='right'),
        'Rectangular (Mid)': manual_rectangular(power, h, method='midpoint'),
        'Trapezoidal': manual_trapezoidal(power, h),
        'Simpson 1/3': manual_simpson_13(power, h),
    }
    
    # Adaptive dengan berbagai segmen
    for n_seg in [1, 5, 10]:
        methods[f'Adaptive ({n_seg} seg)'] = adaptive_integration(power, h, n_seg)
    
    # Richardson Extrapolation sebagai referensi
    power_half = df.set_index('datetime').resample('30min').interpolate().reset_index()['Global_active_power'].values
    trap_h = manual_trapezoidal(power, 1.0)
    trap_h2 = manual_trapezoidal(power_half, 0.5)
    reference = richardson_extrapolation(trap_h, trap_h2, 2)
    
    # Display hasil
    print("\nResults:")
    print("-" * 80)
    print(f"{'Method':<25} {'Energy (kWh)':<15} {'Error':<15} {'Rel Error (%)'}")
    print("-" * 80)
    
    for method, value in methods.items():
        error = abs(value - reference)
        rel_error = error / reference * 100
        print(f"{method:<25} {value:>12.4f}   {error:>12.6f}   {rel_error:>10.4f}")
    
    print("-" * 80)
    print(f"{'Richardson (Reference)':<25} {reference:>12.4f}   {'---':>12}   {'---':>10}")
    print("-" * 80)
    
    # Find best method
    errors = {m: abs(v - reference) for m, v in methods.items()}
    best_method = min(errors, key=errors.get)
    print(f"\nBest Method: {best_method} with error {errors[best_method]:.6f} kWh")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Energy values
    axes[0].barh(list(methods.keys()), list(methods.values()), color='skyblue')
    axes[0].axvline(reference, color='red', linestyle='--', linewidth=2, label='Reference')
    axes[0].set_xlabel('Total Energy (kWh)')
    axes[0].set_title('Integration Results Comparison')
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot 2: Errors
    axes[1].bar(range(len(errors)), list(errors.values()), color='coral')
    axes[1].set_xticks(range(len(errors)))
    axes[1].set_xticklabels(list(errors.keys()), rotation=45, ha='right')
    axes[1].set_ylabel('Absolute Error (kWh)')
    axes[1].set_title('Integration Error Comparison')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example1_integration.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved: example1_integration.png")
    
    return df, power

def example_2_differentiation_analysis(power):
    """
    Contoh 2: Analisis Diferensiasi - Deteksi Anomali
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: DIFFERENTIATION ANALYSIS - Anomaly Detection")
    print("=" * 80)
    
    h = 1.0
    
    print("\n1. Computing derivatives using different methods...")
    
    # Hitung turunan
    dpdt_forward = manual_diff_forward(power, h)
    dpdt_backward = manual_diff_backward(power, h)
    dpdt_central = manual_diff_central(power, h)
    dpdt_numpy = np.gradient(power, h)
    
    # Error analysis
    print("\nError Analysis (compared to NumPy gradient):")
    print("-" * 80)
    
    methods = {
        'Forward': dpdt_forward,
        'Backward': dpdt_backward,
        'Central': dpdt_central
    }
    
    print(f"{'Method':<15} {'MAE':<12} {'RMSE':<12} {'Max Error'}")
    print("-" * 80)
    
    for name, dpdt in methods.items():
        errors = calculate_errors(dpdt_numpy, dpdt)
        print(f"{name:<15} {errors['MAE']:>10.6f}  {errors['RMSE']:>10.6f}  {errors['Max_Error']:>10.6f}")
    
    # Deteksi anomali
    threshold = 2.0
    anomaly_indices = np.where(np.abs(dpdt_central) > threshold)[0]
    
    print(f"\n2. Anomaly Detection (threshold = {threshold} kW/h):")
    print(f"   Total anomalies detected: {len(anomaly_indices)}")
    print(f"   Percentage: {len(anomaly_indices)/len(power)*100:.2f}%")
    
    if len(anomaly_indices) > 0:
        print(f"   Max rate of change: {np.max(np.abs(dpdt_central)):.4f} kW/h")
        print(f"   First anomaly at index: {anomaly_indices[0]}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Original power
    axes[0, 0].plot(power, color='blue', alpha=0.7, linewidth=1)
    axes[0, 0].set_ylabel('Power (kW)')
    axes[0, 0].set_title('Original Power Consumption')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Derivatives comparison
    axes[0, 1].plot(dpdt_forward, label='Forward', alpha=0.7, linewidth=1)
    axes[0, 1].plot(dpdt_central, label='Central', alpha=0.7, linewidth=1)
    axes[0, 1].plot(dpdt_numpy, label='NumPy', alpha=0.7, linewidth=1, linestyle='--')
    axes[0, 1].set_ylabel('dP/dt (kW/h)')
    axes[0, 1].set_title('Derivative Methods Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Anomaly detection
    axes[1, 0].plot(dpdt_central, color='orange', alpha=0.7, linewidth=1)
    axes[1, 0].axhline(threshold, color='red', linestyle='--', label=f'Threshold = ±{threshold}')
    axes[1, 0].axhline(-threshold, color='red', linestyle='--')
    axes[1, 0].scatter(anomaly_indices, dpdt_central[anomaly_indices], 
                      color='red', s=50, zorder=5, label='Anomalies')
    axes[1, 0].set_ylabel('dP/dt (kW/h)')
    axes[1, 0].set_title('Anomaly Detection')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Error distribution
    error_central = dpdt_central - dpdt_numpy
    axes[1, 1].hist(error_central, bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Central Difference Error Distribution')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example2_differentiation.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved: example2_differentiation.png")

def example_3_interpolation_analysis(power):
    """
    Contoh 3: Analisis Interpolasi - Data Recovery
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: INTERPOLATION ANALYSIS - Missing Data Recovery")
    print("=" * 80)
    
    # Simulasi missing data
    window_size = 50
    missing_indices = list(range(20, 30))  # 10 consecutive missing points
    
    print(f"\n1. Simulating missing data...")
    print(f"   Window size: {window_size}")
    print(f"   Missing indices: {missing_indices[0]} to {missing_indices[-1]}")
    print(f"   Number of missing points: {len(missing_indices)}")
    
    # Prepare data
    x_full = np.arange(window_size)
    y_full = power[:window_size].copy()
    
    x_sample = np.delete(x_full, missing_indices)
    y_sample = np.delete(y_full, missing_indices)
    
    # Interpolation
    print("\n2. Performing interpolation...")
    
    # Newton
    coef_newton = newton_divided_diff(x_sample, y_sample)
    y_true_missing = [y_full[i] for i in missing_indices]
    y_newton_missing = [evaluate_newton(x_sample, coef_newton, i) for i in missing_indices]
    
    # Lagrange
    y_lagrange_missing = [lagrange_interpolation(x_sample, y_sample, i) for i in missing_indices]
    
    # Error analysis
    errors_newton = calculate_errors(y_true_missing, y_newton_missing)
    errors_lagrange = calculate_errors(y_true_missing, y_lagrange_missing)
    
    print("\nError Analysis:")
    print("-" * 80)
    print(f"{'Method':<15} {'MAE':<12} {'RMSE':<12} {'MAPE (%)':<12} {'Max Error'}")
    print("-" * 80)
    print(f"{'Newton':<15} {errors_newton['MAE']:>10.6f}  {errors_newton['RMSE']:>10.6f}  "
          f"{errors_newton['MAPE']:>10.4f}  {errors_newton['Max_Error']:>10.6f}")
    print(f"{'Lagrange':<15} {errors_lagrange['MAE']:>10.6f}  {errors_lagrange['RMSE']:>10.6f}  "
          f"{errors_lagrange['MAPE']:>10.4f}  {errors_lagrange['Max_Error']:>10.6f}")
    print("-" * 80)
    
    # Detailed comparison
    print("\n3. Detailed point-by-point comparison:")
    print("-" * 80)
    print(f"{'Index':<8} {'True':<10} {'Newton':<10} {'Lagrange':<10} {'Error N':<10} {'Error L'}")
    print("-" * 80)
    for i, idx in enumerate(missing_indices[:5]):  # Show first 5
        print(f"{idx:<8} {y_true_missing[i]:>8.4f}  {y_newton_missing[i]:>8.4f}  "
              f"{y_lagrange_missing[i]:>8.4f}  {abs(y_true_missing[i]-y_newton_missing[i]):>8.4f}  "
              f"{abs(y_true_missing[i]-y_lagrange_missing[i]):>8.4f}")
    if len(missing_indices) > 5:
        print(f"... ({len(missing_indices)-5} more points)")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Newton interpolation
    x_interp = np.linspace(0, window_size-1, 200)
    y_interp = [evaluate_newton(x_sample, coef_newton, xi) for xi in x_interp]
    
    axes[0, 0].scatter(x_sample, y_sample, color='blue', s=50, label='Available Data', zorder=3)
    axes[0, 0].scatter(missing_indices, y_true_missing, color='red', s=100, 
                      marker='x', label='True (Missing)', zorder=4, linewidth=2)
    axes[0, 0].plot(x_interp, y_interp, 'g--', linewidth=2, label='Newton Interp', zorder=2)
    axes[0, 0].scatter(missing_indices, y_newton_missing, color='orange', s=80, 
                      marker='^', label='Predicted', zorder=5, edgecolors='black')
    axes[0, 0].set_xlabel('Index')
    axes[0, 0].set_ylabel('Power (kW)')
    axes[0, 0].set_title('Newton Interpolation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Error per point
    errors_per_point = np.abs(np.array(y_true_missing) - np.array(y_newton_missing))
    axes[0, 1].bar(missing_indices, errors_per_point, color='coral', alpha=0.7)
    axes[0, 1].set_xlabel('Index')
    axes[0, 1].set_ylabel('Absolute Error (kW)')
    axes[0, 1].set_title('Error per Missing Point')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: True vs Predicted
    axes[1, 0].scatter(y_true_missing, y_newton_missing, alpha=0.7, s=50)
    min_val = min(min(y_true_missing), min(y_newton_missing))
    max_val = max(max(y_true_missing), max(y_newton_missing))
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 0].set_xlabel('True Value (kW)')
    axes[1, 0].set_ylabel('Predicted Value (kW)')
    axes[1, 0].set_title('True vs Predicted')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Methods comparison
    methods_names = ['Newton', 'Lagrange']
    metrics = ['MAE', 'RMSE', 'Max_Error']
    newton_vals = [errors_newton[m] for m in metrics]
    lagrange_vals = [errors_lagrange[m] for m in metrics]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    axes[1, 1].bar(x_pos - width/2, newton_vals, width, label='Newton', color='green', alpha=0.7)
    axes[1, 1].bar(x_pos + width/2, lagrange_vals, width, label='Lagrange', color='magenta', alpha=0.7)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].set_ylabel('Error Value')
    axes[1, 1].set_title('Methods Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example3_interpolation.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved: example3_interpolation.png")

def example_4_regression_analysis(power):
    """
    Contoh 4: Analisis Regresi - Trend Prediction
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: REGRESSION ANALYSIS - Trend Prediction")
    print("=" * 80)
    
    # Split data
    train_ratio = 0.8
    split_idx = int(len(power) * train_ratio)
    
    x_train = np.arange(split_idx)
    y_train = power[:split_idx]
    x_test = np.arange(split_idx, len(power))
    y_test = power[split_idx:]
    
    print(f"\n1. Data split:")
    print(f"   Training: {len(x_train)} points ({train_ratio*100}%)")
    print(f"   Testing:  {len(x_test)} points ({(1-train_ratio)*100}%)")
    
    # Test berbagai derajat
    degrees = [1, 2, 3, 4, 5]
    print(f"\n2. Testing polynomial degrees: {degrees}")
    
    results = []
    
    print("\nResults:")
    print("-" * 80)
    print(f"{'Degree':<8} {'Train R²':<12} {'Test R²':<12} {'Test RMSE':<12} {'Overfit Score'}")
    print("-" * 80)
    
    for deg in degrees:
        coeffs = manual_poly_regression(x_train, y_train, deg)
        
        y_train_pred = evaluate_polynomial(coeffs, x_train)
        y_test_pred = evaluate_polynomial(coeffs, x_test)
        
        errors_train = calculate_errors(y_train, y_train_pred)
        errors_test = calculate_errors(y_test, y_test_pred)
        
        overfit_score = abs(errors_train['R2'] - errors_test['R2'])
        
        results.append({
            'degree': deg,
            'train_r2': errors_train['R2'],
            'test_r2': errors_test['R2'],
            'test_rmse': errors_test['RMSE'],
            'overfit': overfit_score,
            'coeffs': coeffs,
            'y_test_pred': y_test_pred
        })
        
        print(f"{deg:<8} {errors_train['R2']:>10.6f}  {errors_test['R2']:>10.6f}  "
              f"{errors_test['RMSE']:>10.6f}  {overfit_score:>12.6f}")
    
    # Find best degree
    best_idx = max(range(len(results)), key=lambda i: results[i]['test_r2'])
    best_deg = results[best_idx]['degree']
    
    print("-" * 80)
    print(f"\nBest Degree: {best_deg} (R² = {results[best_idx]['test_r2']:.6f})")
    print(f"Coefficients: {[f'{c:.6f}' for c in results[best_idx]['coeffs'][:4]]}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot regressions for degrees 1, 2, 3
    for i, deg in enumerate([1, 2, 3]):
        ax = axes[0, i]
        idx = degrees.index(deg)
        
        all_x = np.concatenate([x_train, x_test])
        all_y = np.concatenate([y_train, y_test])
        
        y_train_pred = evaluate_polynomial(results[idx]['coeffs'], x_train)
        
        ax.plot(all_x, all_y, alpha=0.3, color='blue', linewidth=1, label='Original')
        ax.plot(x_train, y_train_pred, color='red', linewidth=2, label='Train Pred')
        ax.plot(x_test, results[idx]['y_test_pred'], color='green', 
               linewidth=2, linestyle='--', label='Test Pred')
        ax.axvline(split_idx, color='black', linestyle=':', label='Split')
        ax.set_xlabel('Index')
        ax.set_ylabel('Power (kW)')
        ax.set_title(f'Degree {deg} (R²={results[idx]["test_r2"]:.4f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Plot 4: R² comparison
    axes[1, 0].plot(degrees, [r['train_r2'] for r in results], 
                   marker='o', label='Train R²', linewidth=2)
    axes[1, 0].plot(degrees, [r['test_r2'] for r in results], 
                   marker='s', label='Test R²', linewidth=2)
    axes[1, 0].set_xlabel('Polynomial Degree')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title('R² vs Degree')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Plot 5: RMSE comparison
    axes[1, 1].plot(degrees, [r['test_rmse'] for r in results], 
                   marker='o', linewidth=2, color='coral')
    axes[1, 1].set_xlabel('Polynomial Degree')
    axes[1, 1].set_ylabel('Test RMSE')
    axes[1, 1].set_title('Test RMSE vs Degree')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Overfitting score
    axes[1, 2].bar(degrees, [r['overfit'] for r in results], color='skyblue', alpha=0.7)
    axes[1, 2].set_xlabel('Polynomial Degree')
    axes[1, 2].set_ylabel('Overfitting Score')
    axes[1, 2].set_title('Overfitting Indicator')
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example4_regression.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved: example4_regression.png")

def example_5_comprehensive_analysis():
    """
    Contoh 5: Analisis Komprehensif menggunakan ErrorAnalyzer
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: COMPREHENSIVE ERROR ANALYSIS")
    print("=" * 80)
    
    df = load_sample_data()
    power = df['Global_active_power'].values
    
    print("\n1. Initializing ErrorAnalyzer...")
    analyzer = ErrorAnalyzer(power, h=1.0)
    
    print("\n2. Running all analyses...")
    
    # Integration
    print("\n   - Integration analysis...")
    df_int = analyzer.analyze_integration_error()
    
    # Differentiation
    print("   - Differentiation analysis...")
    df_diff = analyzer.analyze_differentiation_error()
    
    # Interpolation
    print("   - Interpolation analysis...")
    missing_indices = list(range(50, 56))
    df_interp = analyzer.analyze_interpolation_error(missing_indices, window_size=100)
    
    # Regression
    print("   - Regression analysis...")
    df_reg = analyzer.analyze_regression_error(degrees=[1,2,3,4,5], train_ratio=0.8)
    
    print("\n3. Generating visualizations...")
    
    # Generate all plots
    fig1 = analyzer.plot_integration_comparison()
    fig1.savefig('comprehensive_integration.png', dpi=150, bbox_inches='tight')
    
    fig2 = analyzer.plot_differentiation_comparison()
    fig2.savefig('comprehensive_differentiation.png', dpi=150, bbox_inches='tight')
    
    fig3 = analyzer.plot_interpolation_comparison()
    fig3.savefig('comprehensive_interpolation.png', dpi=150, bbox_inches='tight')
    
    fig4 = analyzer.plot_regression_comparison()
    fig4.savefig('comprehensive_regression.png', dpi=150, bbox_inches='tight')
    
    fig5 = analyzer.plot_overall_summary()
    fig5.savefig('comprehensive_summary.png', dpi=150, bbox_inches='tight')
    
    print("\n   All plots saved:")
    print("   - comprehensive_integration.png")
    print("   - comprehensive_differentiation.png")
    print("   - comprehensive_interpolation.png")
    print("   - comprehensive_regression.png")
    print("   - comprehensive_summary.png")
    
    # Generate text report
    print("\n4. Generating text report...")
    analyzer.generate_report('comprehensive_report.txt')
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS COMPLETED")
    print("=" * 80)

def main():
    """
    Main function untuk menjalankan semua contoh
    """
    print("\n")
    print("*" * 80)
    print("NUMERICAL METHODS - COMPREHENSIVE EXAMPLES")
    print("*" * 80)
    print("\n")
    
    # Run examples
    df, power = example_1_integration_analysis()
    example_2_differentiation_analysis(power)
    example_3_interpolation_analysis(power)
    example_4_regression_analysis(power)
    example_5_comprehensive_analysis()
    
    print("\n")
    print("*" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("Check the generated PNG files and comprehensive_report.txt")
    print("*" * 80)
    print("\n")

if __name__ == "__main__":
    main()