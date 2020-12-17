import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def readData(addr):
    _data = pd.read_csv(addr, sep=",", header=None)
    _data.columns = ["x0", "x1", "y"]
    _data[["x0", "x1"]] -= _data[["x0", "x1"]].mean()
    _data[["x0", "x1"]] /= np.sqrt(_data[["x0", "x1"]].var())
    _data = _data.values
    return _data

def Prob(_x, _y, c):
    mul = np.matmul
    m = []
    s = np.zeros((c, c))
    _nx = _x.copy()
    for i in range(c):
        m += [_x[_y[:, 0]==i].mean(axis=0)]
    for j in range(len(_x)):
        mj = m[_y[j][0].astype(int)]
        s += mul(np.transpose([_x[j]-mj]), [_x[j]-mj])
        _nx[j] -= mj
    s = s/len(_x)
    res = np.zeros((len(_x), c))
    for i in range(len(_x)):
        for j in range(c):
            res[i][j] = 1 / np.sqrt(2*np.pi*np.linalg.det(s)) * np.exp(-1/2 * mul(mul(_x[i]-m[j], np.linalg.inv(s)), np.transpose(_x[i]-m[j])))
    return res

# ------------------------------ Read Datas ------------------------------
_data = readData('Datas/BC-Train1.csv')
_x = _data[:, 0:2]
_y = _data[:, [2]]

_dataTest = readData('Datas/BC-Test1.csv')
_xt = _data[:, 0:2]
_yt = _data[:, [2]]

# ------------------------------ Bayesian -------------------------------

p = Prob(_x, _y, 2)
for i in range(len(p)):
    if(p[i][0]>p[i][1]):
        print(0)
    else:
        print(1)

# -------------------------------- Plots --------------------------------
plt.figure(1)

# plt.subplot(2, 2, 1)
plt.plot(_x[ _y[:, 0] == 0][:, [0]], _x[ _y[:, 0] == 0][:, [1]], '.', label='Train datasets')
plt.plot(_x[ _y[:, 0] == 1][:, [0]], _x[ _y[:, 0] == 1][:, [1]], '.', label='Train datasets')
plt.title('Train datasets and regression lines')
plt.ylabel('x1')
plt.xlabel('x0')
plt.legend()

# plt.subplot(2, 2, 2)
# plt.plot(_xt, _yt, '.', label='Test datasets')
# plt.plot(_xt, np.matmul(_1xt, _theta1), '-', label='closed form solution')
# plt.plot(_xt, np.matmul(_1xt, _theta2), '-', label='Gradient Descent algorithm')
# plt.title('Test datasets and regression lines')
# plt.ylabel('y')
# plt.xlabel('x')
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(np.arange(len(J)), J, '-')
# plt.title('Cost function')
# plt.ylabel('J')
# plt.xlabel('Iterations')
# plt.legend()

# plt.show()
