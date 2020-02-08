import numpy as np
import math
import scipy as sp
from scipy.special import chebyt
from scipy.integrate import quad
import pylab

# Приближаемая функция f(x) = sin x
def f(x):
    return np.sin(2 * math.pi * x)


a = -1
b = 1
n = 9

c = np.zeros(n)


# Подинтегральная функция
def func(k, x):
    return chebyt(k)(x) * f(x) / math.sqrt(1 - x ** 2)


# Функция, которая возвращяет коэффициент 1/pi или 2/pi перед c_k
def coef(k):
    if k == 0:
        return 1. / math.pi
    else:
        return 2. / math.pi


# Найдем коэффициенты с_k
for i in range(n):
    c[i] = coef(i) * quad(lambda x: func(i, x), a, b)[0]


# Функция, которая считает значение искомого многочлена в точке x
def P(x):
    sum = 0
    for i in range(n):
        sum += c[i] * chebyt(i)(x)
    return sum


# Построим на графике sin(x) и многочлен P(x) на отрезке [-1,1]
X = np.linspace(a, b, 100)
pylab.plot(X, f(X), 'r', lw=2)
pylab.plot(X, P(X), 'b', lw=2)

# Построим график ошибки на отрезке [-1,1]
pylab.plot(X, f(X) - np.array([c[i]*chebyt(i)(X) for i in range(n)]).sum(axis=0), 'g', lw=3)
pylab.show()

