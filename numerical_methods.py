    # numerical_methods.py
    import numpy as np

    # ========================================
    # 1. INTEGRASI NUMERIK (MANUAL)
    # ========================================

    def manual_rectangular(y, h, method='left'):
        """
        Integrasi numerik menggunakan metode Rectangular (Riemann Sum)
        
        Parameters:
        - y: array nilai fungsi
        - h: step size
        - method: 'left', 'right', atau 'midpoint'
        
        Returns:
        - nilai integral
        """
        if method == 'left':
            return h * sum(y[:-1])
        elif method == 'right':
            return h * sum(y[1:])
        elif method == 'midpoint':
            n = len(y) - 1
            midpoints = [(y[i] + y[i+1]) / 2 for i in range(n)]
            return h * sum(midpoints)
        else:
            return h * sum(y[:-1])

    def manual_trapezoidal(y, h):
        """
        Integrasi numerik menggunakan metode Trapezoidal
        Formula: I = (h/2) * [y0 + 2*y1 + 2*y2 + ... + 2*yn-1 + yn]
        
        Parameters:
        - y: array nilai fungsi
        - h: step size
        
        Returns:
        - nilai integral
        """
        n = len(y)
        total = y[0] + y[-1]
        for i in range(1, n-1):
            total += 2 * y[i]
        return (h / 2) * total

    def manual_simpson_13(y, h):
        """
        Integrasi numerik menggunakan metode Simpson 1/3
        Formula: I = (h/3) * [y0 + 4*y1 + 2*y2 + 4*y3 + ... + yn]
        Memerlukan jumlah interval genap (n-1 genap)
        
        Parameters:
        - y: array nilai fungsi
        - h: step size
        
        Returns:
        - nilai integral
        """
        n = len(y) - 1
        if n % 2 != 0:
            # Jika ganjil, gunakan simpson untuk n-1 titik, trapezoid untuk sisa
            return manual_simpson_13(y[:-1], h) + (h/2)*(y[-2] + y[-1])
        
        total = y[0] + y[-1]
        for i in range(1, n):
            if i % 2 == 1:
                total += 4 * y[i]
            else:
                total += 2 * y[i]
        return (h / 3) * total

    def manual_simpson_38(y, h):
        """
        Integrasi numerik menggunakan metode Simpson 3/8
        Formula: I = (3h/8) * [y0 + 3*y1 + 3*y2 + 2*y3 + 3*y4 + ... + yn]
        Memerlukan jumlah interval kelipatan 3 (n-1 kelipatan 3)
        
        Parameters:
        - y: array nilai fungsi
        - h: step size
        
        Returns:
        - nilai integral atau None jika tidak memenuhi syarat
        """
        n = len(y) - 1
        if n % 3 != 0:
            return None
        
        total = y[0] + y[-1]
        for i in range(1, n):
            if i % 3 == 0:
                total += 2 * y[i]
            else:
                total += 3 * y[i]
        return (3 * h / 8) * total

    def adaptive_integration(y, h, n_segments):
        """
        Integrasi dengan segmen yang dapat diatur
        Membagi data menjadi n_segments dan mengintegrasikan dengan Simpson 1/3
        
        Parameters:
        - y: array nilai fungsi
        - h: step size original
        - n_segments: jumlah segmen pembagian
        
        Returns:
        - nilai integral
        """
        n_total = len(y)
        points_per_segment = n_total // n_segments
        
        if points_per_segment < 3:
            # Jika terlalu sedikit, gunakan trapezoidal
            return manual_trapezoidal(y, h)
        
        total_integral = 0
        for i in range(n_segments):
            start_idx = i * points_per_segment
            if i == n_segments - 1:
                end_idx = n_total
            else:
                end_idx = (i + 1) * points_per_segment + 1
            
            segment = y[start_idx:end_idx]
            
            # Gunakan Simpson 1/3 untuk setiap segmen
            if len(segment) > 2:
                total_integral += manual_simpson_13(segment, h)
            else:
                total_integral += manual_trapezoidal(segment, h)
        
        return total_integral

    # ========================================
    # 2. DIFERENSIASI NUMERIK (MANUAL)
    # ========================================

    def manual_diff_forward(y, h):
        """
        Diferensiasi numerik menggunakan Forward Difference
        Formula: f'(x) = [f(x+h) - f(x)] / h
        
        Parameters:
        - y: array nilai fungsi
        - h: step size
        
        Returns:
        - array turunan pertama
        """
        dy = np.zeros(len(y))
        for i in range(len(y) - 1):
            dy[i] = (y[i+1] - y[i]) / h
        # Untuk titik terakhir, gunakan backward
        dy[-1] = (y[-1] - y[-2]) / h
        return dy

    def manual_diff_backward(y, h):
        """
        Diferensiasi numerik menggunakan Backward Difference
        Formula: f'(x) = [f(x) - f(x-h)] / h
        
        Parameters:
        - y: array nilai fungsi
        - h: step size
        
        Returns:
        - array turunan pertama
        """
        dy = np.zeros(len(y))
        # Untuk titik pertama, gunakan forward
        dy[0] = (y[1] - y[0]) / h
        for i in range(1, len(y)):
            dy[i] = (y[i] - y[i-1]) / h
        return dy

    def manual_diff_central(y, h):
        """
        Diferensiasi numerik menggunakan Central Difference
        Formula: f'(x) = [f(x+h) - f(x-h)] / (2h)
        Lebih akurat daripada forward/backward (error O(h^2))
        
        Parameters:
        - y: array nilai fungsi
        - h: step size
        
        Returns:
        - array turunan pertama
        """
        dy = np.zeros(len(y))
        # Central difference untuk titik tengah
        for i in range(1, len(y) - 1):
            dy[i] = (y[i+1] - y[i-1]) / (2 * h)
        # Forward difference untuk titik pertama
        dy[0] = (-3*y[0] + 4*y[1] - y[2]) / (2*h)
        # Backward difference untuk titik terakhir
        dy[-1] = (3*y[-1] - 4*y[-2] + y[-3]) / (2*h)
        return dy

    def manual_diff_second_order(y, h):
        """
        Diferensiasi numerik orde kedua
        Formula: f''(x) = [f(x+h) - 2*f(x) + f(x-h)] / h^2
        
        Parameters:
        - y: array nilai fungsi
        - h: step size
        
        Returns:
        - array turunan kedua
        """
        d2y = np.zeros(len(y))
        for i in range(1, len(y) - 1):
            d2y[i] = (y[i+1] - 2*y[i] + y[i-1]) / (h**2)
        # Boundary menggunakan forward/backward difference
        d2y[0] = (2*y[0] - 5*y[1] + 4*y[2] - y[3]) / (h**2)
        d2y[-1] = (2*y[-1] - 5*y[-2] + 4*y[-3] - y[-4]) / (h**2)
        return d2y

    # ========================================
    # 3. INTERPOLASI NEWTON (MANUAL)
    # ========================================

    def newton_divided_diff(x, y):
        """
        Menghitung koefisien untuk interpolasi Newton menggunakan divided difference
        
        Parameters:
        - x: array titik x
        - y: array nilai fungsi di titik x
        
        Returns:
        - array koefisien divided difference
        """
        n = len(y)
        coef = np.zeros([n, n])
        coef[:,0] = y
        
        for j in range(1, n):
            for i in range(n - j):
                coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
        
        return coef[0, :]

    def evaluate_newton(x_data, coef, x_target):
        """
        Evaluasi polinomial Newton di titik x_target
        
        Parameters:
        - x_data: array titik x yang digunakan untuk interpolasi
        - coef: koefisien dari newton_divided_diff
        - x_target: titik yang ingin dievaluasi
        
        Returns:
        - nilai interpolasi di x_target
        """
        n = len(x_data) - 1
        p = coef[n]
        for k in range(1, n + 1):
            p = coef[n-k] + (x_target - x_data[n-k]) * p
        return p

    def lagrange_interpolation(x_data, y_data, x_target):
        """
        Interpolasi menggunakan metode Lagrange
        
        Parameters:
        - x_data: array titik x
        - y_data: array nilai fungsi di titik x
        - x_target: titik yang ingin dievaluasi
        
        Returns:
        - nilai interpolasi di x_target
        """
        n = len(x_data)
        result = 0.0
        
        for i in range(n):
            term = y_data[i]
            for j in range(n):
                if i != j:
                    term *= (x_target - x_data[j]) / (x_data[i] - x_data[j])
            result += term
        
        return result

    # ========================================
    # 4. REGRESI POLINOMIAL (MANUAL)
    # ========================================

    def manual_poly_regression(x, y, degree):
        """
        Regresi polinomial menggunakan metode Least Squares (OLS)
        Membangun dan menyelesaikan Normal Equation: (X^T * X) * beta = X^T * y
        
        Parameters:
        - x: array variabel independen
        - y: array variabel dependen
        - degree: derajat polinomial
        
        Returns:
        - array koefisien [beta0, beta1, beta2, ..., beta_degree]
        """
        n = len(x)
        # Matriks Vandermonde
        X = np.zeros((n, degree + 1))
        for i in range(degree + 1):
            X[:, i] = x**i
        
        # Perhitungan Manual (X^T * X) * beta = X^T * y
        XT = X.T
        A = XT @ X
        B = XT @ y
        
        # Solve linear system
        coeffs = np.linalg.solve(A, B)
        return coeffs

    def evaluate_polynomial(coeffs, x):
        """
        Evaluasi polinomial dengan koefisien yang diberikan
        
        Parameters:
        - coeffs: array koefisien [beta0, beta1, ..., beta_n]
        - x: nilai atau array nilai x
        
        Returns:
        - nilai polinomial di x
        """
        result = 0
        for i, coeff in enumerate(coeffs):
            result += coeff * (x ** i)
        return result

    # ========================================
    # 5. FUNGSI ANALISIS ERROR
    # ========================================

    def calculate_errors(y_true, y_pred):
        """
        Menghitung berbagai metrik error
        
        Parameters:
        - y_true: nilai sebenarnya
        - y_pred: nilai prediksi
        
        Returns:
        - dictionary berisi berbagai metrik error
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Root Mean Square Error
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
        # Mean Absolute Percentage Error (hindari pembagian dengan nol)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        # Relative Error (rata-rata)
        relative_error = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        # R-squared
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        # Max Absolute Error
        max_error = np.max(np.abs(y_true - y_pred))
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Relative_Error': relative_error,
            'R2': r2,
            'Max_Error': max_error
        }

    def richardson_extrapolation(f_h, f_h2, p):
        """
        Richardson Extrapolation untuk meningkatkan akurasi
        Formula: f_improved = (2^p * f(h/2) - f(h)) / (2^p - 1)
        
        Parameters:
        - f_h: hasil dengan step size h
        - f_h2: hasil dengan step size h/2
        - p: order of error (1 untuk rectangular, 2 untuk trapezoidal, 4 untuk simpson)
        
        Returns:
        - nilai yang lebih akurat
        """
        return (2**p * f_h2 - f_h) / (2**p - 1)