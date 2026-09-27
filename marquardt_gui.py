#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Графическая программа для метода Марквардта (Левенберга-Марквардта)
Реализация алгоритма нелинейной оптимизации для подгонки параметров
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class MarquardtOptimizer:
    """
    Класс для реализации метода Левенберга-Марквардта
    """
    
    def __init__(self):
        self.iteration_log = []
        
    def compute_residuals(self, y_observed, y_predicted):
        """Вычисление вектора невязок"""
        return y_observed - y_predicted
        
    def compute_cost_function(self, residuals):
        """Вычисление целевой функции (сумма квадратов невязок)"""
        return 0.5 * np.dot(residuals, residuals)
    
    def calculate_jacobian_numerical(self, x_values, params, model_func, epsilon=1e-7):
        """
        Численное вычисление матрицы Якоби
        """
        n_points = len(x_values)
        n_params = len(params)
        jacobi_matrix = np.zeros((n_points, n_params))
        
        for param_idx in range(n_params):
            params_perturbed = params.copy()
            params_perturbed[param_idx] += epsilon
            
            y_plus = model_func(x_values, params_perturbed)
            y_original = model_func(x_values, params)
            
            jacobi_matrix[:, param_idx] = (y_plus - y_original) / epsilon
            
        return jacobi_matrix
    
    def perform_optimization(self, x_values, y_values, initial_params, 
                           model_func, max_iterations=100, 
                           tolerance=1e-8, damping_initial=0.01):
        """
        Основной алгоритм метода Левенберга-Марквардта
        """
        current_params = np.array(initial_params, dtype=float).copy()
        damping_factor = damping_initial
        self.iteration_log = []
        
        num_params = len(current_params)
        identity_matrix = np.eye(num_params)
        
        for iter_num in range(max_iterations):
            # Вычисление предсказанных значений
            y_predicted = model_func(x_values, current_params)
            
            # Вычисление невязок
            residual_vector = self.compute_residuals(y_values, y_predicted)
            
            # Вычисление текущей стоимости
            current_cost = self.compute_cost_function(residual_vector)
            
            # Логирование итерации
            self.iteration_log.append({
                'iteration': iter_num,
                'parameters': current_params.copy(),
                'cost': current_cost,
                'damping': damping_factor
            })
            
            # Вычисление якобиана
            jacobi_matrix = self.calculate_jacobian_numerical(
                x_values, current_params, model_func
            )
            
            # Вычисление градиента: J^T * r
            gradient_vector = jacobi_matrix.T @ residual_vector
            
            # Проверка условия остановки по градиенту
            if np.linalg.norm(gradient_vector) < tolerance:
                return {
                    'success': True,
                    'parameters': current_params,
                    'iterations': iter_num + 1,
                    'message': f'Достигнута сходимость по градиенту на итерации {iter_num + 1}'
                }
            
            # Вычисление приближения гессиана: J^T * J
            hessian_approx = jacobi_matrix.T @ jacobi_matrix
            
            # Решение системы уравнений: (H + λI) * δ = J^T * r
            try:
                augmented_matrix = hessian_approx + damping_factor * identity_matrix
                parameter_update = np.linalg.solve(augmented_matrix, gradient_vector)
            except np.linalg.LinAlgError:
                return {
                    'success': False,
                    'parameters': current_params,
                    'iterations': iter_num + 1,
                    'message': 'Матрица системы вырождена'
                }
            
            # Вычисление новых параметров
            new_params = current_params + parameter_update
            y_new_predicted = model_func(x_values, new_params)
            new_residuals = self.compute_residuals(y_values, y_new_predicted)
            new_cost = self.compute_cost_function(new_residuals)
            
            # Адаптация параметра затухания
            if new_cost < current_cost:
                # Принимаем новые параметры
                current_params = new_params
                damping_factor = max(damping_factor * 0.1, 1e-10)
                
                # Проверка сходимости по изменению функции стоимости
                if abs(current_cost - new_cost) < tolerance:
                    return {
                        'success': True,
                        'parameters': current_params,
                        'iterations': iter_num + 1,
                        'message': f'Достигнута сходимость на итерации {iter_num + 1}'
                    }
            else:
                # Отклоняем новые параметры и увеличиваем затухание
                damping_factor = min(damping_factor * 10.0, 1e10)
        
        return {
            'success': False,
            'parameters': current_params,
            'iterations': max_iterations,
            'message': f'Достигнуто максимальное число итераций ({max_iterations})'
        }


class MarquardtApp:
    """
    Главный класс приложения с графическим интерфейсом
    """
    
    def __init__(self, window):
        self.window = window
        self.window.title("Метод Марквардта - Оптимизация параметров")
        self.window.geometry("1250x850")
        
        self.optimizer = MarquardtOptimizer()
        self.setup_ui()
        self.load_example_data()
        
    def setup_ui(self):
        """Создание пользовательского интерфейса"""
        # Основной фрейм
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(1, weight=1)
        
        # Левая панель - настройки
        self.build_settings_panel(main_frame)
        
        # Правая панель - визуализация
        self.build_visualization_panel(main_frame)
        
    def build_settings_panel(self, parent):
        """Построение панели настроек"""
        settings_frame = ttk.LabelFrame(parent, text="Настройки оптимизации", padding="10")
        settings_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        
        row_counter = 0
        
        # Выбор модели
        ttk.Label(settings_frame, text="Модель функции:").grid(
            row=row_counter, column=0, sticky="w", pady=5
        )
        row_counter += 1
        
        self.model_selector = ttk.Combobox(settings_frame, state="readonly", width=32)
        self.model_selector['values'] = [
            'Линейная: a·x + b',
            'Полином 2-й степени: a·x² + b·x + c',
            'Экспоненциальная: a·exp(b·x)',
            'Синусоида: a·sin(b·x + c)'
        ]
        self.model_selector.current(2)
        self.model_selector.grid(row=row_counter, column=0, pady=5)
        row_counter += 1
        
        # Входные данные X
        ttk.Label(settings_frame, text="Значения X (разделитель: запятая):").grid(
            row=row_counter, column=0, sticky="w", pady=5
        )
        row_counter += 1
        self.x_input = ttk.Entry(settings_frame, width=35)
        self.x_input.grid(row=row_counter, column=0, pady=5)
        row_counter += 1
        
        # Входные данные Y
        ttk.Label(settings_frame, text="Значения Y (разделитель: запятая):").grid(
            row=row_counter, column=0, sticky="w", pady=5
        )
        row_counter += 1
        self.y_input = ttk.Entry(settings_frame, width=35)
        self.y_input.grid(row=row_counter, column=0, pady=5)
        row_counter += 1
        
        # Начальные параметры
        ttk.Label(settings_frame, text="Начальные параметры:").grid(
            row=row_counter, column=0, sticky="w", pady=5
        )
        row_counter += 1
        self.params_input = ttk.Entry(settings_frame, width=35)
        self.params_input.grid(row=row_counter, column=0, pady=5)
        row_counter += 1
        
        # Параметры алгоритма
        ttk.Label(settings_frame, text="Максимум итераций:").grid(
            row=row_counter, column=0, sticky="w", pady=5
        )
        row_counter += 1
        self.max_iter_input = ttk.Entry(settings_frame, width=35)
        self.max_iter_input.insert(0, "100")
        self.max_iter_input.grid(row=row_counter, column=0, pady=5)
        row_counter += 1
        
        ttk.Label(settings_frame, text="Точность (tolerance):").grid(
            row=row_counter, column=0, sticky="w", pady=5
        )
        row_counter += 1
        self.tolerance_input = ttk.Entry(settings_frame, width=35)
        self.tolerance_input.insert(0, "1e-8")
        self.tolerance_input.grid(row=row_counter, column=0, pady=5)
        row_counter += 1
        
        ttk.Label(settings_frame, text="Начальное затухание λ:").grid(
            row=row_counter, column=0, sticky="w", pady=5
        )
        row_counter += 1
        self.damping_input = ttk.Entry(settings_frame, width=35)
        self.damping_input.insert(0, "0.01")
        self.damping_input.grid(row=row_counter, column=0, pady=5)
        row_counter += 1
        
        # Кнопки действий
        button_container = ttk.Frame(settings_frame)
        button_container.grid(row=row_counter, column=0, pady=15)
        
        ttk.Button(button_container, text="Оптимизировать", 
                   command=self.run_optimization_task).pack(pady=5, fill='x')
        ttk.Button(button_container, text="Загрузить пример", 
                   command=self.load_example_data).pack(pady=5, fill='x')
        ttk.Button(button_container, text="Очистить всё", 
                   command=self.clear_interface).pack(pady=5, fill='x')
        
    def build_visualization_panel(self, parent):
        """Построение панели визуализации"""
        # Панель результатов
        results_frame = ttk.LabelFrame(parent, text="Результаты оптимизации", padding="10")
        results_frame.grid(row=0, column=1, sticky="nsew", pady=(0, 10))
        
        self.results_display = scrolledtext.ScrolledText(
            results_frame, width=60, height=12, wrap=tk.WORD
        )
        self.results_display.pack(fill=tk.BOTH, expand=True)
        
        # Панель графиков
        graph_frame = ttk.LabelFrame(parent, text="Графики", padding="10")
        graph_frame.grid(row=1, column=1, sticky="nsew")
        
        self.figure = plt.Figure(figsize=(10, 7), dpi=90)
        self.canvas_widget = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def get_model_by_index(self, index):
        """Получение функции модели по индексу"""
        if index == 0:  # Линейная
            return lambda x, p: p[0] * x + p[1], 2
        elif index == 1:  # Квадратичная
            return lambda x, p: p[0] * x**2 + p[1] * x + p[2], 3
        elif index == 2:  # Экспоненциальная
            return lambda x, p: p[0] * np.exp(p[1] * x), 2
        elif index == 3:  # Синусоидальная
            return lambda x, p: p[0] * np.sin(p[1] * x + p[2]), 3
        
    def run_optimization_task(self):
        """Запуск процесса оптимизации"""
        try:
            # Парсинг входных данных
            x_data = np.array([float(val.strip()) for val in self.x_input.get().split(',')])
            y_data = np.array([float(val.strip()) for val in self.y_input.get().split(',')])
            init_params = [float(val.strip()) for val in self.params_input.get().split(',')]
            
            max_iters = int(self.max_iter_input.get())
            tol = float(self.tolerance_input.get())
            damp = float(self.damping_input.get())
            
            # Валидация
            if len(x_data) != len(y_data):
                self.show_error("Массивы X и Y должны быть одинаковой длины!")
                return
                
            # Получение модели
            model_idx = self.model_selector.current()
            model_func, expected_params = self.get_model_by_index(model_idx)
            
            if len(init_params) != expected_params:
                self.show_error(f"Для выбранной модели нужно {expected_params} параметров!")
                return
            
            # Выполнение оптимизации
            result = self.optimizer.perform_optimization(
                x_data, y_data, init_params, model_func,
                max_iterations=max_iters, tolerance=tol, damping_initial=damp
            )
            
            # Отображение результатов
            self.display_optimization_results(result, x_data, y_data, model_func)
            self.visualize_results(x_data, y_data, result['parameters'], model_func)
            
        except ValueError as err:
            self.show_error(f"Ошибка в данных: {str(err)}")
        except Exception as err:
            self.show_error(f"Непредвиденная ошибка: {str(err)}")
    
    def display_optimization_results(self, result, x_data, y_data, model_func):
        """Отображение текстовых результатов"""
        self.results_display.delete('1.0', tk.END)
        
        output = "=" * 55 + "\n"
        output += "РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ - МЕТОД МАРКВАРДТА\n"
        output += "=" * 55 + "\n\n"
        
        output += f"Статус: {'УСПЕХ' if result['success'] else 'НЕ СОШЛОСЬ'}\n"
        output += f"Информация: {result['message']}\n"
        output += f"Итераций выполнено: {result['iterations']}\n\n"
        
        output += "Оптимальные параметры:\n"
        for idx, param_value in enumerate(result['parameters']):
            output += f"  p[{idx}] = {param_value:.8f}\n"
        
        # Вычисление статистик
        y_fitted = model_func(x_data, result['parameters'])
        residuals = y_data - y_fitted
        rmse = np.sqrt(np.mean(residuals**2))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        output += f"\nСреднеквадратичная ошибка (RMSE): {rmse:.6e}\n"
        output += f"Коэффициент детерминации (R²): {r_squared:.6f}\n"
        
        self.results_display.insert('1.0', output)
    
    def visualize_results(self, x_data, y_data, optimal_params, model_func):
        """Построение графиков"""
        self.figure.clear()
        
        # График 1: Данные и аппроксимация
        subplot1 = self.figure.add_subplot(2, 1, 1)
        subplot1.scatter(x_data, y_data, color='navy', s=60, 
                        label='Исходные данные', alpha=0.7, edgecolors='black')
        
        x_smooth = np.linspace(np.min(x_data), np.max(x_data), 250)
        y_smooth = model_func(x_smooth, optimal_params)
        subplot1.plot(x_smooth, y_smooth, 'r-', linewidth=2.5, 
                     label='Аппроксимация')
        
        subplot1.set_xlabel('X', fontsize=11)
        subplot1.set_ylabel('Y', fontsize=11)
        subplot1.set_title('Результат аппроксимации данных', fontsize=12, fontweight='bold')
        subplot1.legend(loc='best')
        subplot1.grid(True, alpha=0.4, linestyle='--')
        
        # График 2: Сходимость
        subplot2 = self.figure.add_subplot(2, 1, 2)
        iterations = [entry['iteration'] for entry in self.optimizer.iteration_log]
        costs = [entry['cost'] for entry in self.optimizer.iteration_log]
        
        subplot2.semilogy(iterations, costs, 'b-', linewidth=2.5, marker='o', markersize=4)
        subplot2.set_xlabel('Номер итерации', fontsize=11)
        subplot2.set_ylabel('Функция стоимости (логарифм)', fontsize=11)
        subplot2.set_title('График сходимости алгоритма', fontsize=12, fontweight='bold')
        subplot2.grid(True, alpha=0.4, linestyle='--')
        
        self.figure.tight_layout()
        self.canvas_widget.draw()
    
    def load_example_data(self):
        """Загрузка примера данных"""
        np.random.seed(123)
        x_example = np.linspace(0.0, 2.0, 25)
        y_example = 2.8 * np.exp(1.4 * x_example) + np.random.normal(0, 3.0, 25)
        
        self.x_input.delete(0, tk.END)
        self.x_input.insert(0, ', '.join([f'{val:.3f}' for val in x_example]))
        
        self.y_input.delete(0, tk.END)
        self.y_input.insert(0, ', '.join([f'{val:.3f}' for val in y_example]))
        
        self.params_input.delete(0, tk.END)
        self.params_input.insert(0, "1.5, 1.0")
        
        self.model_selector.current(2)
    
    def clear_interface(self):
        """Очистка интерфейса"""
        self.x_input.delete(0, tk.END)
        self.y_input.delete(0, tk.END)
        self.params_input.delete(0, tk.END)
        self.results_display.delete('1.0', tk.END)
        self.figure.clear()
        self.canvas_widget.draw()
    
    def show_error(self, message):
        """Показать сообщение об ошибке"""
        tk.messagebox.showerror("Ошибка", message)


def main():
    """Точка входа в программу"""
    root_window = tk.Tk()
    application = MarquardtApp(root_window)
    root_window.mainloop()


if __name__ == "__main__":
    main()
