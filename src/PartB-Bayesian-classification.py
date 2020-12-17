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

# ------------------------------ Read Datas ------------------------------
_data = readData('Datas/BC-Train1.csv')
_x = _data[:, 0:2]
_y = _data[:, [2]]
m = len(_x)
# _1x = np.insert(_x, 0, np.ones(m), axis=1)

_dataTest = readData('Datas/BC-Test1.csv')
_xt = _data[:, 0:2]
_yt = _data[:, [2]]
mt = len(_xt)
# _1xt = np.insert(_xt, 0, np.ones(mt), axis=1)

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

plt.show()