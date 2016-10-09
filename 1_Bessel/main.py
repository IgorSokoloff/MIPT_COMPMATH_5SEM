import math
import os


"""
S - current value of summ
S_0 - previous value of summ
a - constant before (x/2)**(2k)

"""
def bessel_sum_count(x, eps):
    S = 1
    S_0 = S + 1 #S_0 must be more than S in order to pass first iteration of while
    k = 1
    a = 1
    while math.fabs(S - S_0) > eps:
        a *= ((-1) / (k**2))
        S_0 = S

        S += a * (x/2)**(2*k)
        k = k + 1

    # calculating new error of method
    #k = k + 1
    a *= ((-1) / (k ** 2))
    eps_method = a * (x/2)**(2*k)
    return k - 1

def bessel_funk(x, n):
    S = 1
    a = 1
    for i in range(1, n + 1):
        a *= ((-1) / (i ** 2))
        S += a * (x / 2) ** (2 * i)
    return S

def bessel_derivative(x, k, h):
    return (bessel_funk(x + h, k) - bessel_funk(x - h, k))/(2 * h)


if __name__ == '__main__':
    eps = 1e-6

    h = 6.69e-6 #from slides

    k = bessel_sum_count(1, eps)

    print (bessel_derivative(1, k, h))



