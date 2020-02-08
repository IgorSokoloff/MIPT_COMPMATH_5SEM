import numpy as np
import scipy as sp
import math
from scipy.linalg import solve_banded
from scipy.spatial.distance import euclidean


N = 21
h = math.pi/20.
h2 = h**2

# Функция, которая вычисляет x_{k-1} = P_k*x_k + Q_k
def calc_xk(x, k, P, Q):
    x[k-1] = P[k] * x[k] + Q[k]

# Функция, которая осуществляет прямой ход
def find_PQ(P, Q, a, b, c):
    # Зададим P_1 и  Q_1
    P[1] = -c[0] / b[0]
    Q[1] = f[0] / b[0]
    for i in range(1, N):
        P[i] = -c[i-1] / (a[i - 1] * P[i - 1] + b[i - 1])
        Q[i] = (f[i - 1] - a[i - 1] * Q[i - 1]) / (a[i - 1] * P[i - 1] + b[i - 1])


#Массивы для коэффициентов (по умолчанию из нулей)
P = np.zeros(N)
Q = np.zeros(N)

#Списки для коэффициентов
a = []
b = []
c = []
f = []

#Заполним списки коэффициентов
a.append(0)
b.append(1)
c.append(0)
f.append(0)


for i in range(1, N - 1):
    a.append(-1)
    b.append(2 + h2)
    c.append(-1)
    f.append(2 * h2 * math.sin(i * h))

a.append(0)
b.append(1)
c.append(0)
f.append(0)

answer = np.zeros(N)


# Прямой ход
find_PQ(P, Q, a, b, c)


# Обратный ход
for i in reversed(range(1, N)):
    calc_xk(answer, i, P, Q)

# Сравним с sin(nh)
y = []
for i in range(0, N):
    y.append(math.sin(i * h))

print ('My solution (Прогонка): x =', answer)
print ('sin(nh): ', y)

print (len(y))
print (len(answer))

# Ошибка между моим ответом и эталонным в евклидовой метрике
print ("error =",euclidean(y, answer) )