import numpy as np
import pandas as pd
from scipy.integrate import ode
import matplotlib.pyplot as plt


# Функция правой части системы дифференциальных уравнений
def f(t, y):
    return [y[1], 6 * y[0] / t ** 2]


# Задаем начальные условия
y0 = [1, 3]
t0 = 1
tmax = 2
eps = 0.00001

# функция вывода одного графика
def print_graph(x, y, title, id, count_graphs):
    plt.subplot(1, count_graphs, id)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.title(title)
    plt.plot(x, y, '-o')


# Решение системы дифференциальных уравнений методом RKF45
def solve_rkf45(h):
    t = np.arange(t0, tmax + h, h)
    rk_integ = ode(f).set_integrator("dopri5", atol=eps).set_initial_value(y0, t[0])
    X = np.array([y0, *[rk_integ.integrate(t[i]) for i in range(1, len(t))]])
    return t, X[:, 0]


# Решение системы дифференциальных уравнений методом Эйлера
def solve_euler(h):
    t = np.arange(t0, tmax + h, h)
    y = np.zeros_like(t)
    y[0] = y0[0]
    z = np.zeros_like(t)
    z[0] = y0[1]
    for i in range(1, len(t)):
        y[i] = y[i - 1] + h * z[i - 1]
        z[i] = z[i - 1] + h * 6 * y[i - 1] / t[i - 1] ** 2
    return t, y

# список перебираемых h_int
h_list = [0.1, 0.05, 0.025, 0.0125]

for h_int in h_list:
    print("h_int =", h_int)
    x_values = np.arange(t0, tmax + h_int, h_int)

    # вычислим значения функций
    rkf45_values = solve_rkf45(h_int)
    euler_values = solve_euler(h_int)
    exact_values = [t ** 3 for t in x_values]

    # вычислим погрешности
    rkf45_3_errors = np.abs(rkf45_values[1] - rkf45_values[0] ** 3)
    euler_errors = np.abs(euler_values[1] - euler_values[0] ** 3)

    # вывод таблицы
    results = pd.DataFrame({
        'h': euler_values[0],
        'Value RKF45': rkf45_values[1],
        'Value Euler': euler_values[1],
        'Exact': exact_values,
        'Error RKF45': rkf45_3_errors,
        'Error Euler': euler_errors,
    })

    print(results.to_string(index=False))
    print("RKF45 global err: ",np.sum(rkf45_3_errors))
    print("Euler global err: ", np.sum(euler_errors))
    # выводим график значений
    plt.figure(figsize=(15, 4))

    y = rkf45_values[1]
    t = euler_values[0]
    print_graph(t, y, 'RKF45, step=' + str(h_int), 1, 3)

    t, y = euler_values
    print_graph(t, y, 'Эйлер', 2, 3)

    print_graph(x_values, exact_values, 'Exact solution', 3, 3)

    plt.savefig("Graphs_h_" + str(h_int) + ".jpg")
    plt.show()

    # выводим график погрешности
    plt.figure(figsize=(15, 4))

    y = rkf45_3_errors
    t = rkf45_values[0]
    print_graph(t, y, 'RKF45, step=' + str(h_int), 1, 2)

    y = euler_errors
    t = euler_values[0]
    print_graph(t, y, 'Эйлер', 2, 2)

    plt.savefig("error_h_" + str(h_int) + ".jpg")
    plt.show()