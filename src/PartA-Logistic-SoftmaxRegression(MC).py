import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import e

def readData(addr):
    _data = pd.read_csv(addr, sep=",", header=None)
    _data.columns = ["x1", "x2", "x3", "x4", "class"]
    _data[["x1", "x2", "x3", "x4"]] -= _data[["x1", "x2", "x3", "x4"]].mean()
    _data[["x1", "x2", "x3", "x4"]] /= np.sqrt(_data[["x1", "x2", "x3", "x4"]].var())
    _data = _data.values
    return _data

def g(z): # Sigmoid
    return 1.0 / ( 1 + e**(-z))

def one_vs_one(_x, _y, alpha, iterations, nclass):
    mul = np.matmul
    all_theta=[]
    _x = np.insert(_x, 0, 1, axis=1)
    for i in range(nclass-1):
        for j in range(i+1, nclass):
            _nx = _x[np.logical_or(_y[:, 0]==i, _y[:, 0]==j)]
            _ny = _y[np.logical_or(_y[:, 0]==i, _y[:, 0]==j)]
            _ny[_ny[:, 0]==i], _ny[_ny[:, 0]==j] = 0, 1
            _theta = np.zeros((len(_nx[0]), 1))
            for _ in range(iterations):
                _h = g(mul(_nx, _theta))
                _theta = _theta - (alpha/len(_nx)) * mul( (_h-_ny).transpose(), _nx).transpose()
            all_theta += [_theta]
    return all_theta

def accuracy_one_vs_one(_x, _y, all_theta, nclass):
    mul = np.matmul
    _x = np.insert(_x, 0, 1, axis=1)
    k=0
    _a = []
    for _ in range(nclass):
        _a += [np.zeros((len(_x), 1))]
    for i in range(nclass-1):
        for j in range(i+1, nclass):
            _h = g(mul(_x, all_theta[k]))
            k+=1
            _a[i] = _a[i] + (1-_h)/3
            _a[j] = _a[j] + _h/3
    _yp = np.zeros((len(_y), 1))
    for i in range(len(_x)):
        c = 0
        for j in range(nclass):
            if(_a[j][i]>_a[c][i]):
                c = j
        _yp[i] = c
    return (_yp==_y).astype(float).sum()/len(_y)

def one_vs_all(_x, _y, alpha, iterations, nclass):
    mul = np.matmul
    all_theta=[]
    J = []
    _x = np.insert(_x, 0, 1, axis=1)
    for i in range(nclass):
        J += [np.arange(0)]
        _nx = _x
        _ny = _y.copy()
        _ny[_y[:, 0]!=i], _ny[_y[:, 0]==i] = 0, 1
        _theta = np.zeros((len(_nx[0]), 1))
        for _ in range(iterations):
            _h = g(mul(_nx, _theta))
            _theta = _theta - (alpha/len(_nx)) * mul( (_h-_ny).transpose(), _nx).transpose()
            J[i] = np.append(J[i], np.sum(1.0 / len(_nx) * ( -_ny * np.log(_h.astype(float)) - (1-_ny) * np.log(1-_h.astype(float)))))
        all_theta += [_theta]
    return all_theta, J

def accuracy_one_vs_all(_x, _y, all_theta, nclass):
    mul = np.matmul
    _x = np.insert(_x, 0, 1, axis=1)
    _a = []
    for i in range(nclass):
        _h = g(mul(_x, all_theta[i]))
        _a += [_h]
    _yp = np.zeros((len(_y), 1))
    for i in range(len(_x)):
        c = 0
        for j in range(nclass):
            if(_a[j][i]>_a[c][i]):
                c = j
        _yp[i] = c
    return (_yp==_y).astype(float).sum()/len(_y)

# ------------------------------ Read Datas ------------------------------
_data = readData('Datas/iris.data')

_data0 = _data[ _data[:, 4] == 'Iris-setosa']
_data1 = _data[ _data[:, 4] == 'Iris-versicolor']
_data2 = _data[ _data[:, 4] == 'Iris-virginica']

m0 = int(len(_data0)*0.8)
m1 = int(len(_data1)*0.8)
m2 = int(len(_data2)*0.8)
mt0 = int(len(_data1)*0.2)
mt1 = int(len(_data1)*0.2)
mt2 = int(len(_data2)*0.2)

_x = np.concatenate((_data0[0:m0][:, 0:4], _data1[0:m1][:, 0:4], _data2[0:m2][:, 0:4]))
_y = np.concatenate((np.zeros((m0, 1)), np.ones((m1, 1)), 2*np.ones((m2, 1))))
_1x = np.insert(_x, 0, np.ones(m0+m1+m2), axis=1)

_xt = np.concatenate((_data0[m0:][:, 0:4], _data1[m1:][:, 0:4], _data2[m2:][:, 0:4]))
_yt = np.concatenate((np.zeros((mt0, 1)), np.ones((mt1, 1)), 2*np.ones((mt2, 1))))
_1xt = np.insert(_xt, 0, np.ones(mt0+mt1+mt2), axis=1)

# ---------------------------- one-vs-one LR ----------------------------

iterations = 2000
alpha = 0.01
nclass = 3
all_theta = one_vs_one(_x, _y, alpha, iterations, nclass)
print("train accuracy for one-vs-one logistic regression:")
print(accuracy_one_vs_one(_x, _y, all_theta, nclass))
print()
print("test accuracy for one-vs-one logistic regression:")
print(accuracy_one_vs_one(_xt, _yt, all_theta, nclass))
print()

# ---------------------------- one-vs-all LR ----------------------------

iterations = 2000
alpha = 0.01
nclass = 3
all_theta, J = one_vs_all(_x, _y, alpha, iterations, nclass)
print("train accuracy for one-vs-all logistic regression:")
print(accuracy_one_vs_all(_x, _y, all_theta, nclass))
print()
print("test accuracy for one-vs-all logistic regression:")
print(accuracy_one_vs_all(_xt, _yt, all_theta, nclass))
print()

# -------------------------------- Plots --------------------------------
plt.figure(1)

classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
for i in range(nclass):
    plt.plot(np.arange(len(J[i])), J[i], '-', label=classes[i]+"-vs-all")
    plt.title('Cost function for one-vs-all')
    plt.ylabel('J')
    plt.xlabel('Iterations')
    plt.legend()

plt.show()
