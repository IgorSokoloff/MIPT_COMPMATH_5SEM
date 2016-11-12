import numpy as np
import math
from numpy import linalg
from numpy.linalg import norm

la = 0.01  # лямбда
N = 100
h = 2 * math.pi / N


# Определим f(x)
def f(x):
    return (1 + 2 * la) * (math.cos(x / 2.) ** 2) - la * (x ** 2 + math.pi ** 2) / 2.


xn = []
fn = []

# Посчитаем значения столбца f
for i in range(N):
    xn.append(-math.pi + (i - 0.5) * h)
    fn.append(f(xn[i]))


# Определим \delta_{nm}
def delta(n, m):
    if (n == m):
        return 1
    else:
        return 0


# Введем матрицы L,D,U
L = np.zeros((N, N))
D = np.zeros((N, N))
U = np.zeros((N, N))

# Заполним L
for i in range(N):
    for j in range(i):
        L[i, j] = delta(i, j) - la * h ** 2 * (i - j)

# Заполним U
for i in range(N):
    for j in range(i):
        U[j, i] = delta(i, j) - la * h ** 2 * (i - j)

# Заполним D
for i in range(N):
    D[i, i] = 1

A = U + D + L

# ограничение на число итераций
maxiter = 1000
# ограничение на величину невязки
eps = 1e-6

i=0
x0 = np.ones(N)
x = x0

# Метод Зейделя
while i < maxiter:
    i += 1
    for j in range(N):
        # Посчитаем Lx
        Lx = 0
        for k in range(j):
            Lx += L[j, k] * x[k]
        # Посчитаем Ux
        Ux = 0
        for k in range(j+1, N):
            Ux += U[j,k] * x[k]
        # Находим x_k+1
        x[j] = (fn[j] - Lx - Ux) / D[j,j]
    # Вычисляем невязку
    r = fn - A.dot(x)
    if norm(r) < eps:
        break

print (x)
