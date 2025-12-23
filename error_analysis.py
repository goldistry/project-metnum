# error_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numerical_methods import *

class ErrorAnalyzer:
    """
    Class untuk melakukan analisis error mendalam pada metode numerik
    """
    
    def __init__(self, data, h=1.0):
        """
        Initialize Error Analyzer
        
        Parameters:
        - data: array data untuk dianalisis
        - h: step size
        """
        self.data = data
        self.h = h
        self.results = {}
    
    # ========================================
    # INTEGRASI ERROR ANALYSIS
    # ========================================
    
    def analyze_integration_error(self, reference_method='richardson'):
        """
        Analisis error untuk berbagai metode integrasi
        
        Parameters:
        - reference_method: metode yang digunakan sebagai referensi ('richardson' atau 'numpy')
        
        Returns:
        - DataFrame berisi hasil analisis error
        """
        print("Menganalisis error integrasi...")
        
        # Hitung dengan berbagai metode
        methods = {}
        methods['Rectangular_Left'] = manual_rectangular(self.data, self.h, method='left')
        methods['Rectangular_Right'] = manual_rectangular(self.data, self.h, method='right')
        methods['Rectangular_Mid'] = manual_rectangular(self.data, self.h, method='midpoint')
        methods['Trapezoidal'] = manual_trapezoidal(self.data, self.h)
        methods['Simpson_13'] = manual_simpson_13(self.data, self.h)
        
        simpson_38 = manual_simpson_38(self.data, self.h)
        if simpson_38 is not None:
            methods['Simpson_38'] = simpson_38
        
        # Adaptive dengan berbagai segmen
        for n_seg in [1, 5, 10]:
            methods[f'Adaptive_{n_seg}seg'] = adaptive_integration(self.data, self.h, n_seg)
        
        # Hitung nilai referensi
        if reference_method == 'richardson':
            # Gunakan data dengan step size setengah untuk Richardson
            data_half = self.data[::2] if len(self.data) > 100 else self.data
            h_half = self.h * 2 if len(self.data) > 100 else self.h
            
            trap_h = manual_trapezoidal(self.data, self.h)
            trap_h2 = manual_trapezoidal(data_half, h_half)
            exact_val = richardson_extrapolation(trap_h2, trap_h, 2)
        else:
            exact_val = np.trapz(self.data, dx=self.h)
        
        # Hitung error metrics
        results = []
        for name, value in methods.items():
            abs_error = abs(value - exact_val)
            rel_error = abs_error / abs(exact_val) * 100
            
            results.append({
                'Method': name,
                'Result': value,
                'Absolute_Error': abs_error,
                'Relative_Error_%': rel_error,
                'Reference': exact_val
            })
        
        df_results = pd.DataFrame(results)
        self.results['integration'] = df_results
        
        print("Analisis integrasi selesai.")
        return df_results
    
    # ========================================
    # DIFERENSIASI ERROR ANALYSIS
    # ========================================
    
    def analyze_differentiation_error(self):
        """
        Analisis error untuk berbagai metode diferensiasi
        
        Returns:
        - DataFrame berisi hasil analisis error
        """
        print("Menganalisis error diferensiasi...")
        
        # Hitung dengan berbagai metode
        dpdt_forward = manual_diff_forward(self.data, self.h)
        dpdt_backward = manual_diff_backward(self.data, self.h)
        dpdt_central = manual_diff_central(self.data, self.h)
        
        # Gunakan NumPy sebagai referensi
        dpdt_numpy = np.gradient(self.data, self.h)
        
        # Hitung error metrics untuk setiap metode
        methods = {
            'Forward': dpdt_forward,
            'Backward': dpdt_backward,
            'Central': dpdt_central
        }
        
        results = []
        for name, dpdt in methods.items():
            errors = calculate_errors(dpdt_numpy, dpdt)
            errors['Method'] = name
            results.append(errors)
        
        df_results = pd.DataFrame(results)
        df_results = df_results[['Method', 'MAE', 'RMSE', 'MAPE', 'Max_Error']]
        
        self.results['differentiation'] = df_results
        self.results['differentiation_arrays'] = {
            'Forward': dpdt_forward,
            'Backward': dpdt_backward,
            'Central': dpdt_central,
            'NumPy': dpdt_numpy
        }
        
        print("Analisis diferensiasi selesai.")
        return df_results
    
    # ========================================
    # INTERPOLASI ERROR ANALYSIS
    # ========================================
    
    def analyze_interpolation_error(self, missing_indices, window_size=50):
        """
        Analisis error untuk metode interpolasi
        
        Parameters:
        - missing_indices: list indeks data yang dianggap hilang
        - window_size: ukuran window data untuk interpolasi
        
        Returns:
        - DataFrame berisi hasil analisis error
        """
        print("Menganalisis error interpolasi...")
        
        # Ambil window data
        x_full = np.arange(window_size)
        y_full = self.data[:window_size].copy()
        
        # Hapus data pada indeks missing
        valid_missing = [i for i in missing_indices if i < window_size]
        x_sample = np.delete(x_full, valid_missing)
        y_sample = np.delete(y_full, valid_missing)
        
        # True values pada missing indices
        y_true = [y_full[i] for i in valid_missing]
        
        # Interpolasi Newton
        coef_newton = newton_divided_diff(x_sample, y_sample)
        y_newton = [evaluate_newton(x_sample, coef_newton, i) for i in valid_missing]
        
        # Interpolasi Lagrange
        y_lagrange = [lagrange_interpolation(x_sample, y_sample, i) for i in valid_missing]
        
        # Hitung error
        errors_newton = calculate_errors(y_true, y_newton)
        errors_lagrange = calculate_errors(y_true, y_lagrange)
        
        df_results = pd.DataFrame({
            'Method': ['Newton', 'Lagrange'],
            'MAE': [errors_newton['MAE'], errors_lagrange['MAE']],
            'RMSE': [errors_newton['RMSE'], errors_lagrange['RMSE']],
            'MAPE': [errors_newton['MAPE'], errors_lagrange['MAPE']],
            'Max_Error': [errors_newton['Max_Error'], errors_lagrange['Max_Error']]
        })
        
        self.results['interpolation'] = df_results
        self.results['interpolation_details'] = {
            'x_sample': x_sample,
            'y_sample': y_sample,
            'x_missing': valid_missing,
            'y_true': y_true,
            'y_newton': y_newton,
            'y_lagrange': y_lagrange
        }
        
        print("Analisis interpolasi selesai.")
        return df_results
    
    # ========================================
    # REGRESI ERROR ANALYSIS
    # ========================================
    
    def analyze_regression_error(self, degrees=[1, 2, 3, 4, 5], train_ratio=0.8):
        """
        Analisis error untuk berbagai derajat regresi polinomial
        
        Parameters:
        - degrees: list derajat polinomial yang akan dianalisis
        - train_ratio: rasio data training (0-1)
        
        Returns:
        - DataFrame berisi hasil analisis error
        """
        print("Menganalisis error regresi...")
        
        # Split data
        split_idx = int(len(self.data) * train_ratio)
        x_train = np.arange(split_idx)
        y_train = self.data[:split_idx]
        x_test = np.arange(split_idx, len(self.data))
        y_test = self.data[split_idx:]
        
        results = []
        predictions = {}
        
        for deg in degrees:
            # Training
            coeffs = manual_poly_regression(x_train, y_train, deg)
            
            # Prediksi
            y_train_pred = evaluate_polynomial(coeffs, x_train)
            y_test_pred = evaluate_polynomial(coeffs, x_test)
            
            # Error metrics
            errors_train = calculate_errors(y_train, y_train_pred)
            errors_test = calculate_errors(y_test, y_test_pred)
            
            # Overfitting indicator
            overfit_score = abs(errors_train['R2'] - errors_test['R2'])
            
            results.append({
                'Degree': deg,
                'Train_R2': errors_train['R2'],
                'Test_R2': errors_test['R2'],
                'Train_RMSE': errors_train['RMSE'],
                'Test_RMSE': errors_test['RMSE'],
                'Train_MAE': errors_train['MAE'],
                'Test_MAE': errors_test['MAE'],
                'Overfit_Score': overfit_score
            })
            
            predictions[deg] = {
                'train': y_train_pred,
                'test': y_test_pred,
                'coeffs': coeffs
            }
        
        df_results = pd.DataFrame(results)
        
        self.results['regression'] = df_results
        self.results['regression_predictions'] = predictions
        self.results['regression_split'] = {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test
        }
        
        print("Analisis regresi selesai.")
        return df_results
    
    # ========================================
    # VISUALISASI
    # ========================================
    
    def plot_integration_comparison(self, figsize=(15, 10)):
        """
        Plot perbandingan hasil dan error integrasi
        """
        if 'integration' not in self.results:
            print("Jalankan analyze_integration_error() terlebih dahulu!")
            return
        
        df = self.results['integration']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Hasil perhitungan
        axes[0, 0].barh(df['Method'], df['Result'], color='skyblue')
        axes[0, 0].axvline(df['Reference'].iloc[0], color='red', linestyle='--', 
                          linewidth=2, label='Reference')
        axes[0, 0].set_xlabel('Integral Value')
        axes[0, 0].set_title('Integration Results Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # Plot 2: Absolute Error
        axes[0, 1].bar(range(len(df)), df['Absolute_Error'], color='coral')
        axes[0, 1].set_xticks(range(len(df)))
        axes[0, 1].set_xticklabels(df['Method'], rotation=45, ha='right')
        axes[0, 1].set_ylabel('Absolute Error')
        axes[0, 1].set_title('Absolute Error Comparison')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Plot 3: Relative Error
        axes[1, 0].bar(range(len(df)), df['Relative_Error_%'], color='lightgreen')
        axes[1, 0].set_xticks(range(len(df)))
        axes[1, 0].set_xticklabels(df['Method'], rotation=45, ha='right')
        axes[1, 0].set_ylabel('Relative Error (%)')
        axes[1, 0].set_title('Relative Error Comparison')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Plot 4: Error heatmap
        error_matrix = df[['Absolute_Error', 'Relative_Error_%']].T
        im = axes[1, 1].imshow(error_matrix, cmap='YlOrRd', aspect='auto')
        axes[1, 1].set_xticks(range(len(df)))
        axes[1, 1].set_xticklabels(df['Method'], rotation=45, ha='right')
        axes[1, 1].set_yticks([0, 1])
        axes[1, 1].set_yticklabels(['Abs Error', 'Rel Error (%)'])
        axes[1, 1].set_title('Error Heatmap')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        return fig
    
    def plot_differentiation_comparison(self, figsize=(15, 10)):
        """
        Plot perbandingan hasil diferensiasi
        """
        if 'differentiation_arrays' not in self.results:
            print("Jalankan analyze_differentiation_error() terlebih dahulu!")
            return
        
        arrays = self.results['differentiation_arrays']
        df = self.results['differentiation']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Hasil diferensiasi
        x = np.arange(len(arrays['NumPy']))
        axes[0, 0].plot(x, arrays['NumPy'], label='NumPy (Reference)', linewidth=2, alpha=0.7)
        axes[0, 0].plot(x, arrays['Forward'], label='Forward', linewidth=1.5, alpha=0.7)
        axes[0, 0].plot(x, arrays['Central'], label='Central', linewidth=1.5, alpha=0.7)
        axes[0, 0].set_xlabel('Index')
        axes[0, 0].set_ylabel('Derivative')
        axes[0, 0].set_title('Differentiation Results Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Error per titik
        axes[0, 1].plot(x, np.abs(arrays['Forward'] - arrays['NumPy']), 
                       label='Forward Error', alpha=0.7)
        axes[0, 1].plot(x, np.abs(arrays['Central'] - arrays['NumPy']), 
                       label='Central Error', alpha=0.7)
        axes[0, 1].set_xlabel('Index')
        axes[0, 1].set_ylabel('Absolute Error')
        axes[0, 1].set_title('Point-wise Absolute Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Error metrics comparison
        metrics = ['MAE', 'RMSE', 'Max_Error']
        x_pos = np.arange(len(metrics))
        width = 0.25
        
        for i, method in enumerate(['Forward', 'Backward', 'Central']):
            values = [df.loc[df['Method'] == method, metric].values[0] for metric in metrics]
            axes[1, 0].bar(x_pos + i*width, values, width, label=method)
        
        axes[1, 0].set_xticks(x_pos + width)
        axes[1, 0].set_xticklabels(metrics)
        axes[1, 0].set_ylabel('Error Value')
        axes[1, 0].set_title('Error Metrics Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Plot 4: Error distribution
        error_forward = arrays['Forward'] - arrays['NumPy']
        error_central = arrays['Central'] - arrays['NumPy']
        
        axes[1, 1].hist(error_forward, bins=30, alpha=0.5, label='Forward', color='blue')
        axes[1, 1].hist(error_central, bins=30, alpha=0.5, label='Central', color='green')
        axes[1, 1].set_xlabel('Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_interpolation_comparison(self, figsize=(15, 8)):
        """
        Plot perbandingan hasil interpolasi
        """
        if 'interpolation_details' not in self.results:
            print("Jalankan analyze_interpolation_error() terlebih dahulu!")
            return
        
        details = self.results['interpolation_details']
        df = self.results['interpolation']
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Interpolation results
        x_plot = np.linspace(min(details['x_sample']), max(details['x_sample']), 200)
        
        # Evaluate Newton on plot points
        coef = newton_divided_diff(details['x_sample'], details['y_sample'])
        y_plot_newton = [evaluate_newton(details['x_sample'], coef, xi) for xi in x_plot]
        
        axes[0].scatter(details['x_sample'], details['y_sample'], 
                       label='Available Data', color='blue', s=50, zorder=3)
        axes[0].scatter(details['x_missing'], details['y_true'], 
                       label='True (Missing)', color='red', s=100, marker='x', zorder=4)
        axes[0].plot(x_plot, y_plot_newton, 'g--', label='Newton Interp', linewidth=2, zorder=2)
        axes[0].scatter(details['x_missing'], details['y_newton'], 
                       label='Newton Pred', color='orange', s=80, marker='^', 
                       edgecolors='black', zorder=5)
        axes[0].set_xlabel('Index')
        axes[0].set_ylabel('Value')
        axes[0].set_title('Interpolation Results')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Error comparison
        x_missing_range = range(len(details['x_missing']))
        error_newton = np.abs(np.array(details['y_true']) - np.array(details['y_newton']))
        error_lagrange = np.abs(np.array(details['y_true']) - np.array(details['y_lagrange']))
        
        width = 0.35
        axes[1].bar(np.array(x_missing_range) - width/2, error_newton, width, 
                   label='Newton', color='green', alpha=0.7)
        axes[1].bar(np.array(x_missing_range) + width/2, error_lagrange, width, 
                   label='Lagrange', color='magenta', alpha=0.7)
        axes[1].set_xlabel('Missing Point Index')
        axes[1].set_ylabel('Absolute Error')
        axes[1].set_title('Error per Missing Point')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        # Plot 3: Metrics comparison
        metrics = ['MAE', 'RMSE', 'Max_Error']
        newton_vals = [df.loc[df['Method'] == 'Newton', m].values[0] for m in metrics]
        lagrange_vals = [df.loc[df['Method'] == 'Lagrange', m].values[0] for m in metrics]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        axes[2].bar(x_pos - width/2, newton_vals, width, label='Newton', color='green', alpha=0.7)
        axes[2].bar(x_pos + width/2, lagrange_vals, width, label='Lagrange', color='magenta', alpha=0.7)
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(metrics)
        axes[2].set_ylabel('Error Value')
        axes[2].set_title('Error Metrics Comparison')
        axes[2].legend()
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_regression_comparison(self, figsize=(15, 12)):
        """
        Plot perbandingan hasil regresi
        """
        if 'regression' not in self.results:
            print("Jalankan analyze_regression_error() terlebih dahulu!")
            return
        
        df = self.results['regression']
        preds = self.results['regression_predictions']
        split = self.results['regression_split']
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: R² comparison
        axes[0, 0].plot(df['Degree'], df['Train_R2'], marker='o', label='Train R²', linewidth=2)
        axes[0, 0].plot(df['Degree'], df['Test_R2'], marker='s', label='Test R²', linewidth=2)
        axes[0, 0].set_xlabel('Polynomial Degree')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('R² vs Degree')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: RMSE comparison
        axes[0, 1].plot(df['Degree'], df['Train_RMSE'], marker='o', label='Train RMSE', linewidth=2)
        axes[0, 1].plot(df['Degree'], df['Test_RMSE'], marker='s', label='Test RMSE', linewidth=2)
        axes[0, 1].set_xlabel('Polynomial Degree')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('RMSE vs Degree')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Overfitting score
        axes[0, 2].bar(df['Degree'], df['Overfit_Score'], color='coral', alpha=0.7)
        axes[0, 2].set_xlabel('Polynomial Degree')
        axes[0, 2].set_ylabel('Overfitting Score')
        axes[0, 2].set_title('Overfitting Indicator (|Train R² - Test R²|)')
        axes[0, 2].grid(axis='y', alpha=0.3)
        
        # Plot 4-6: Regression results for different degrees
        degrees_to_show = [1, 3, 5] if 5 in df['Degree'].values else df['Degree'].values[:3].tolist()
        
        for idx, deg in enumerate(degrees_to_show):
            ax = axes[1, idx]
            
            # Plot data
            all_x = np.concatenate([split['x_train'], split['x_test']])
            all_y = np.concatenate([split['y_train'], split['y_test']])
            
            ax.plot(all_x, all_y, alpha=0.3, label='Original Data', color='blue', linewidth=1)
            ax.plot(split['x_train'], preds[deg]['train'], 
                   color='red', label=f'Train Pred (D={deg})', linewidth=2)
            ax.plot(split['x_test'], preds[deg]['test'], 
                   color='green', label='Test Pred', linewidth=2, linestyle='--')
            ax.axvline(split['x_train'][-1], color='black', linestyle=':', label='Train/Test Split')
            
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')
            ax.set_title(f'Regression Degree {deg} (R²={df.loc[df["Degree"]==deg, "Test_R2"].values[0]:.3f})')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_overall_summary(self, figsize=(16, 10)):
        """
        Plot ringkasan keseluruhan semua metode
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Integration summary
        if 'integration' in self.results:
            ax1 = fig.add_subplot(gs[0, :2])
            df_int = self.results['integration']
            ax1.barh(df_int['Method'], df_int['Relative_Error_%'], color='skyblue')
            ax1.set_xlabel('Relative Error (%)')
            ax1.set_title('Integration Methods - Relative Error', fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)
        
        # Differentiation summary
        if 'differentiation' in self.results:
            ax2 = fig.add_subplot(gs[0, 2])
            df_diff = self.results['differentiation']
            ax2.bar(df_diff['Method'], df_diff['RMSE'], color='coral')
            ax2.set_ylabel('RMSE')
            ax2.set_title('Differentiation - RMSE', fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(axis='y', alpha=0.3)
        
        # Interpolation summary
        if 'interpolation' in self.results:
            ax3 = fig.add_subplot(gs[1, :2])
            df_interp = self.results['interpolation']
            x = np.arange(len(df_interp))
            width = 0.2
            
            metrics = ['MAE', 'RMSE', 'Max_Error']
            for i, metric in enumerate(metrics):
                offset = (i - 1) * width
                ax3.bar(x + offset, df_interp[metric], width, label=metric)
            
            ax3.set_xticks(x)
            ax3.set_xticklabels(df_interp['Method'])
            ax3.set_ylabel('Error Value')
            ax3.set_title('Interpolation Methods - Error Metrics', fontweight='bold')
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
        
        # Regression summary
        if 'regression' in self.results:
            ax4 = fig.add_subplot(gs[1, 2])
            df_reg = self.results['regression']
            ax4.plot(df_reg['Degree'], df_reg['Test_R2'], marker='o', linewidth=2, color='green')
            ax4.set_xlabel('Polynomial Degree')
            ax4.set_ylabel('R² Score')
            ax4.set_title('Regression - Test R²', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim([0, 1])
        
        # Overall best methods
        ax5 = fig.add_subplot(gs[2, :])
        
        best_methods = []
        if 'integration' in self.results:
            best_int = self.results['integration'].loc[
                self.results['integration']['Relative_Error_%'].idxmin(), 'Method']
            best_methods.append(f"Integration: {best_int}")
        
        if 'differentiation' in self.results:
            best_diff = self.results['differentiation'].loc[
                self.results['differentiation']['RMSE'].idxmin(), 'Method']
            best_methods.append(f"Differentiation: {best_diff}")
        
        if 'interpolation' in self.results:
            best_interp = self.results['interpolation'].loc[
                self.results['interpolation']['MAE'].idxmin(), 'Method']
            best_methods.append(f"Interpolation: {best_interp}")
        
        if 'regression' in self.results:
            best_reg_deg = self.results['regression'].loc[
                self.results['regression']['Test_R2'].idxmax(), 'Degree']
            best_methods.append(f"Regression: Degree {int(best_reg_deg)}")
        
        ax5.text(0.5, 0.5, 'BEST METHODS SUMMARY\n\n' + '\n'.join(best_methods),
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax5.axis('off')
        
        return fig
    
    def generate_report(self, filename='error_analysis_report.txt'):
        """
        Generate text report dari semua analisis
        """
        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ERROR ANALYSIS REPORT - NUMERICAL METHODS\n")
            f.write("=" * 80 + "\n\n")
            
            if 'integration' in self.results:
                f.write("\n" + "-" * 80 + "\n")
                f.write("1. INTEGRATION ERROR ANALYSIS\n")
                f.write("-" * 80 + "\n")
                f.write(self.results['integration'].to_string(index=False))
                f.write("\n\nBest Method: " + 
                       self.results['integration'].loc[
                           self.results['integration']['Relative_Error_%'].idxmin(), 'Method'])
                f.write("\n")
            
            if 'differentiation' in self.results:
                f.write("\n" + "-" * 80 + "\n")
                f.write("2. DIFFERENTIATION ERROR ANALYSIS\n")
                f.write("-" * 80 + "\n")
                f.write(self.results['differentiation'].to_string(index=False))
                f.write("\n\nBest Method: " + 
                       self.results['differentiation'].loc[
                           self.results['differentiation']['RMSE'].idxmin(), 'Method'])
                f.write("\n")
            
            if 'interpolation' in self.results:
                f.write("\n" + "-" * 80 + "\n")
                f.write("3. INTERPOLATION ERROR ANALYSIS\n")
                f.write("-" * 80 + "\n")
                f.write(self.results['interpolation'].to_string(index=False))
                f.write("\n\nBest Method: " + 
                       self.results['interpolation'].loc[
                           self.results['interpolation']['MAE'].idxmin(), 'Method'])
                f.write("\n")
            
            if 'regression' in self.results:
                f.write("\n" + "-" * 80 + "\n")
                f.write("4. REGRESSION ERROR ANALYSIS\n")
                f.write("-" * 80 + "\n")
                f.write(self.results['regression'].to_string(index=False))
                f.write("\n\nBest Degree: " + 
                       str(int(self.results['regression'].loc[
                           self.results['regression']['Test_R2'].idxmax(), 'Degree'])))
                f.write("\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Report generated: {filename}")


# Contoh penggunaan
if __name__ == "__main__":
    print("Error Analysis Module - Numerical Methods")
    print("Import this module and use ErrorAnalyzer class for comprehensive error analysis")
    print("\nExample usage:")
    print("  from error_analysis import ErrorAnalyzer")
    print("  analyzer = ErrorAnalyzer(data, h=1.0)")
    print("  analyzer.analyze_integration_error()")
    print("  analyzer.plot_integration_comparison()")