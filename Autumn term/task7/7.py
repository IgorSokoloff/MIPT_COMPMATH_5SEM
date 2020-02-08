import numpy as np
import matplotlib.pyplot as plt



r1 = np.array([1,1,1])
r2 = np.array([-1,-1,1])
r3 = np.array([-1,1,-1])
r4 = np.array([1,-1,-1])
t = np.array([2,2.25, 1.7, 1.5])

A = np.vstack((r1, r2, r3, r4))
B = np.ndarray((4,4))
B[:,:-1] = A
B[:,-1] = t
A = B


def jf(x, t):
    return np.transpose(2 * np.array([x[0] - A[:,0], x[1] - A[:,1], x[2] - A[:,2], - (t - A[:,3]) ]))

def f(x, t):
    B = np.array([x[0] - A[:,0], x[1] - A[:,1], x[2] - A[:,2],  (t - A[:,3]) ])**2
    B[-1,:] *=-1
    return np.sum(B, axis = 0)

f([0.204,0.0518, -0.469], 0.078333)

x = np.array([0.5, 0.5, 0.5, 0.0], dtype='float64')
error = []
err = 1
while( err > 10e-15):
    dx = np.linalg.solve(jf(x[:-1], x[-1]), f(x[:-1], x[-1]))
    x -= dx
    err = np.linalg.norm(dx,2)
    error.append(np.log(err))
fig = plt.figure()


ax = fig.add_subplot(111)
ax.set_xlabel('n, iterations')
ax.set_ylabel('$\log(||dx||_2)$')


plt.title('Newton''s method')
plt.plot(error)
plt.show()