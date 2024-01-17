import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ExperimentApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Experiment App")

        self.notebook = ttk.Notebook(self.master)
        self.tab_experiment = ttk.Frame(self.notebook)
        self.tab_data_generation = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_experiment, text="Эксперименты")
        self.notebook.add(self.tab_data_generation, text="Генерация данных")
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Переменные для вкладки "Эксперименты"
        self.num_experiments_var = tk.IntVar(value=10)
        self.a0_var = tk.DoubleVar()
        self.a1_var = tk.DoubleVar()
        self.r_squared_var = tk.DoubleVar()
        self.correlation_coefficient_var = tk.DoubleVar()
        self.elasticity_var = tk.DoubleVar()
        self.beta_coefficient_var = tk.DoubleVar()

        # Переменные для вкладки "Генерация данных"
        self.data_size_var = tk.IntVar(value=100)
        self.loc_var = tk.DoubleVar(value=0)  # Новая переменная для сдвига
        self.scale_var = tk.DoubleVar(value=1)  # Новая переменная для масштаба

        self.create_widgets_experiment()
        self.create_widgets_data_generation()

    def create_widgets_experiment(self):
        tk.Label(self.tab_experiment, text="Количество экспериментов:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        tk.Entry(self.tab_experiment, textvariable=self.num_experiments_var).grid(row=0, column=1, padx=10, pady=5)
        tk.Button(self.tab_experiment, text="Выполнить эксперименты", command=self.run_experiments).grid(row=1, column=0, columnspan=2, pady=10)

        tk.Label(self.tab_experiment, text="a0:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        tk.Entry(self.tab_experiment, textvariable=self.a0_var, state="readonly").grid(row=2, column=1, padx=10, pady=5)

        tk.Label(self.tab_experiment, text="a1:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        tk.Entry(self.tab_experiment, textvariable=self.a1_var, state="readonly").grid(row=3, column=1, padx=10, pady=5)

        tk.Label(self.tab_experiment, text="R^2:").grid(row=4, column=0, sticky="w", padx=10, pady=5)
        tk.Entry(self.tab_experiment, textvariable=self.r_squared_var, state="readonly").grid(row=4, column=1, padx=10, pady=5)

        tk.Label(self.tab_experiment, text="Коэффициент корреляции:").grid(row=5, column=0, sticky="w", padx=10, pady=5)
        tk.Entry(self.tab_experiment, textvariable=self.correlation_coefficient_var, state="readonly").grid(row=5, column=1, padx=10, pady=5)

        tk.Label(self.tab_experiment, text="Коэффициент эластичности:").grid(row=6, column=0, sticky="w", padx=10, pady=5)
        tk.Entry(self.tab_experiment, textvariable=self.elasticity_var, state="readonly").grid(row=6, column=1, padx=10, pady=5)

        tk.Label(self.tab_experiment, text="Бета-коэффициент:").grid(row=7, column=0, sticky="w", padx=10, pady=5)
        tk.Entry(self.tab_experiment, textvariable=self.beta_coefficient_var, state="readonly").grid(row=7, column=1, padx=10, pady=5)

        # Таблица для отображения результатов экспериментов
        columns = ("Номер эксперимента", "Время выполнения (мс)", "Размер данных", "Размер данных в квадрате", "Время * Размер данных")
        self.tree = ttk.Treeview(self.tab_experiment, columns=columns, show="headings")

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center")

        self.tree.grid(row=8, column=0, columnspan=2, pady=10, sticky="w")

        # Поле для отображения системы нормальных уравнений
        tk.Label(self.tab_experiment, text="Система нормальных уравнений:").grid(row=9, column=0, columnspan=2, pady=5)
        self.equations_text = scrolledtext.ScrolledText(self.tab_experiment, width=50, height=10, wrap=tk.WORD)
        self.equations_text.grid(row=10, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        self.figure, self.ax = plt.subplots(figsize=(5, 4), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.tab_experiment)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=2, rowspan=11, padx=10, pady=10)




    def create_widgets_data_generation(self):
        tk.Label(self.tab_data_generation, text="Размер данных:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        tk.Entry(self.tab_data_generation, textvariable=self.data_size_var).grid(row=0, column=1, padx=10, pady=5)

        # Новые поля для указания параметров сдвига и масштаба
        tk.Label(self.tab_data_generation, text="Сдвиг (loc):").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        tk.Entry(self.tab_data_generation, textvariable=self.loc_var).grid(row=1, column=1, padx=10, pady=5)

        tk.Label(self.tab_data_generation, text="Масштаб (scale):").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        tk.Entry(self.tab_data_generation, textvariable=self.scale_var).grid(row=2, column=1, padx=10, pady=5)

        tk.Button(self.tab_data_generation, text="Генерировать и сортировать данные", command=self.generate_and_sort_data).grid(row=3, column=0, columnspan=2, pady=10)

        tk.Label(self.tab_data_generation, text="Сгенерированные данные:").grid(row=4, column=0, columnspan=2, pady=5)
        self.generated_data_text = scrolledtext.ScrolledText(self.tab_data_generation, width=50, height=10, wrap=tk.WORD)
        self.generated_data_text.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        tk.Label(self.tab_data_generation, text="Отсортированные данные:").grid(row=6, column=0, columnspan=2, pady=5)
        self.sorted_data_text = scrolledtext.ScrolledText(self.tab_data_generation, width=50, height=10, wrap=tk.WORD)
        self.sorted_data_text.grid(row=7, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        
    def run_experiments(self):
        try:
            num_experiments = self.num_experiments_var.get()
            all_results = []

            for experiment in range(1, num_experiments + 1):
                size = np.random.randint(100, 501)  # Генерация случайного размера данных от 100 до 500
                data = logistic_distribution(size)

                start_time = time.perf_counter()
                writes = cycle_sort(data.copy())
                end_time = time.perf_counter()

                result = {
                    'Номер эксперимента': experiment,
                    'Время выполнения (мс)': round((end_time - start_time) * 1000),
                    'Размер данных': size,
                    'Размер данных в квадрате': size ** 2,
                    'Время * Размер данных': round((end_time - start_time) * 1000 * size)
                }

                all_results.append(result)

            results_df = pd.DataFrame(all_results)

            # Оценивание коэффициентов уравнения связи
            a0, a1 = estimate_coefficients(results_df['Размер данных'], results_df['Время выполнения (мс)'])

            # Расчет коэффициента детерминации
            y_pred = a0 + a1 * results_df['Размер данных']
            r_squared = calculate_r_squared(results_df['Время выполнения (мс)'], y_pred)

            # Расчет коэффициента корреляции
            correlation_coefficient = np.corrcoef(results_df['Время выполнения (мс)'], results_df['Размер данных'])[0, 1]

            # Расчет коэффициента эластичности
            elasticity = calculate_elasticity(results_df['Размер данных'], results_df['Время выполнения (мс)'], a0, a1)

            # Расчет бета-коэффициента
            beta_coefficient = calculate_beta_coefficient(results_df['Размер данных'], results_df['Время выполнения (мс)'], a0, a1)

            # Вывод результатов в поля ввода
            self.a0_var.set(a0)
            self.a1_var.set(a1)
            self.r_squared_var.set(r_squared)
            self.correlation_coefficient_var.set(correlation_coefficient)
            self.elasticity_var.set(elasticity)
            self.beta_coefficient_var.set(beta_coefficient)

            # Отображение таблицы
            self.display_results_table(results_df)

            # Построение графика и оцененной линии регрессии
            self.ax.clear()
            self.ax.scatter(results_df['Размер данных'], results_df['Время выполнения (мс)'], label='Наблюдения')
            self.ax.plot(results_df['Размер данных'], a0 + a1 * results_df['Размер данных'], color='red', label='Линия регрессии')
            self.ax.set_title('Зависимость времени выполнения от размера данных')
            self.ax.set_xlabel('Размер данных')
            self.ax.set_ylabel('Время выполнения (мс)')
            self.ax.legend()
            self.canvas.draw()

            # Отображение системы нормальных уравнений
            self.display_normal_equations(results_df)

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))


    
    

    def generate_and_sort_data(self):
        try:
            size = self.data_size_var.get()
            loc = self.loc_var.get()
            scale = self.scale_var.get()

            # Генерация данных
            generated_data = logistic_distribution(size, loc, scale)

            # Создаем копию данных для сортировки
            data_to_sort = generated_data.copy()

            # Замер времени сортировки для неотсортированных данных
            start_time = time.perf_counter()
            writes, comparisons = cycle_sort(data_to_sort)
            end_time = time.perf_counter()
            sorting_time_ms = round((end_time - start_time) * 1000)

            # Вывод результатов
            self.generated_data_text.delete(1.0, tk.END)
            self.generated_data_text.insert(tk.END, f"Сгенерированные данные: {generated_data}\n"
                                                    f"Время сортировки: {sorting_time_ms} мс\n"
                                                    f"Количество перестановок: {writes}\n"
                                                    f"Количество сравнений: {comparisons}")

            # Вывод отсортированных данных
            self.sorted_data_text.delete(1.0, tk.END)
            self.sorted_data_text.insert(tk.END, f"Отсортированные данные: {data_to_sort}\n"
                                                f"Количество перестановок: {writes}\n"
                                                f"Количество сравнений: {comparisons}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))




    def display_results_table(self, results_df):
        # Очистка таблицы
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Заполнение таблицы
        for i, row in results_df.iterrows():
            self.tree.insert("", i, values=(row['Номер эксперимента'], row['Время выполнения (мс)'], row['Размер данных'],
                                           row['Размер данных в квадрате'], row['Время * Размер данных']))

        # Отображение суммы значений
        sum_values = results_df.sum(numeric_only=True)
        self.tree.insert("", len(results_df), values=('Сумма', int(sum_values['Время выполнения (мс)']), int(sum_values['Размер данных']),
                                                      int(sum_values['Размер данных в квадрате']), int(sum_values['Время * Размер данных'])))

    def display_normal_equations(self, results_df):
        # Подставленные значения в систему нормальных уравнений
        num_experiments_sum = len(results_df)
        size_sum = results_df['Размер данных'].sum()
        size_squared_sum = (results_df['Размер данных'] ** 2).sum()
        time_sum = results_df['Время выполнения (мс)'].sum()
        nnn_sum = results_df['Время * Размер данных'].sum()
        equations_text = f"{num_experiments_sum} * a0 + {size_sum} * a1 = {time_sum}\n{size_sum} * a0 + {size_squared_sum} * a1 = {nnn_sum}"
        self.equations_text.delete(1.0, tk.END)
        self.equations_text.insert(tk.END, equations_text)

        
def logistic_distribution(size, loc=0, scale=1):
    u = np.random.rand(size)
    data = loc + scale * np.log(u / (1 - u))
    return data


def cycle_sort(arr):
    n = len(arr)
    writes = 0
    comparisons = 0

    for cycle_start in range(n - 1):
        item = arr[cycle_start]
        pos = cycle_start
        for i in range(cycle_start + 1, n):
            comparisons += 1
            if arr[i] < item:
                pos += 1

        if pos == cycle_start:
            continue

        while item == arr[pos]:
            pos += 1

        arr[pos], item = item, arr[pos]
        writes += 1

        while pos != cycle_start:
            pos = cycle_start
            for i in range(cycle_start + 1, n):
                comparisons += 1
                if arr[i] < item:
                    pos += 1

            while item == arr[pos]:
                pos += 1

            arr[pos], item = item, arr[pos]
            writes += 1

    return writes, comparisons

def gaussian_elimination(A, b):
    n = len(A)
    # Прямой ход метода Гаусса
    for i in range(n):
        # Делаем диагональный элемент ненулевым
        if A[i][i] == 0:
            for j in range(i+1, n):
                if A[j][i] != 0:
                    A[i], A[j] = A[j], A[i]
                    b[i], b[j] = b[j], b[i]
                    break
        # Обнуляем элементы в столбце под диагональным элементом
        for j in range(i+1, n):
            ratio = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= ratio * A[i][k]
            b[j] -= ratio * b[i]
    # Обратный ход метода Гаусса
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] -= A[i][j] * x[j]
        x[i] = x[i] / A[i][i]
    return x

def estimate_coefficients(x, y):
    # Подготовка матрицы A и вектора b для системы уравнений
    n = len(x)
    A = np.array([[n, np.sum(x)], [np.sum(x), np.sum(x**2)]])
    b = np.array([np.sum(y), np.sum(x*y)])
    # Решение системы методом Гаусса
    a0, a1 = gaussian_elimination(A, b)
    return a0, a1

def calculate_r_squared(y_true, y_pred):
    # Расчет коэффициента детерминации
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared

def calculate_elasticity(x, y, a0, a1):
    # Расчет коэффициента эластичности
    elasticity = a1 * (np.mean(x) / np.mean(y))
    return elasticity

def calculate_beta_coefficient(x, y, a0, a1):
    # Расчет бета-коэффициента
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    beta_numerator = np.sum((x - x_mean) * (y - y_mean))
    beta_denominator = np.sum((x - x_mean)**2)
    beta_coefficient = beta_numerator / beta_denominator
    return beta_coefficient


if __name__ == "__main__":
    root = tk.Tk()
    app = ExperimentApp(root)
    root.mainloop()
