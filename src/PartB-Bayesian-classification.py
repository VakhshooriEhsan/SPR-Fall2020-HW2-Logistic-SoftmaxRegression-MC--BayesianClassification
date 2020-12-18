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

def params(_x, _y, c):
    mul = np.matmul
    Mu = []
    Sigma = np.zeros((c, c))
    _nx = _x.copy()
    for i in range(c):
        Mu += [_x[_y[:, 0]==i].mean(axis=0)]
    for j in range(len(_x)):
        Mu_j = Mu[_y[j][0].astype(int)]
        Sigma += mul(np.transpose([_x[j]-Mu_j]), [_x[j]-Mu_j])
        _nx[j] -= Mu_j
    Sigma = Sigma/len(_x)
    return Mu, Sigma


def Bayesian_accuracies(_x, _y, Mu, Sigma, c):
    mul = np.matmul
    res = np.zeros((len(_x), c))
    for i in range(len(_x)):
        for j in range(c):
            res[i][j] = 1 / np.sqrt(2*np.pi*np.linalg.det(Sigma)) * np.exp(-1/2 * mul(mul(_x[i]-Mu[j], np.linalg.inv(Sigma)), np.transpose(_x[i]-Mu[j])))
    _yp = np.zeros((len(_y), 1))
    for i in range(len(_x)):
        a = 0
        for j in range(c):
            if(res[i][j] > res[i][a]):
                a = j
        _yp[i] = a
    return (_yp==_y).astype(float).sum()/len(_y), _yp

def estimated(_x, _y, Mu, Sigma, c):
    mul = np.matmul
    res = np.zeros((len(_x), c))
    for i in range(len(_x)):
        for j in range(c):
            res[i][j] = 1 / np.sqrt(2*np.pi*np.linalg.det(Sigma)) * np.exp(-1/2 * mul(mul(_x[i]-Mu[j], np.linalg.inv(Sigma)), np.transpose(_x[i]-Mu[j])))
    return res

# ------------------------------ Read Datas ------------------------------
_data = readData('Datas/BC-Train1.csv')
_x1 = _data[:, 0:2]
_y1 = _data[:, [2]]

_data = readData('Datas/BC-Test1.csv')
_xt1 = _data[:, 0:2]
_yt1 = _data[:, [2]]

_data = readData('Datas/BC-Train2.csv')
_x2 = _data[:, 0:2]
_y2 = _data[:, [2]]

_data = readData('Datas/BC-Test2.csv')
_xt2 = _data[:, 0:2]
_yt2 = _data[:, [2]]

nclass = 2

# ------------------------------ Bayesian -------------------------------

Mu1, Sigma1 = params(_x1, _y1, nclass)
p, _yp1 = Bayesian_accuracies(_x1, _y1, Mu1, Sigma1, nclass)
print("train_dataset_1 accuracy for Bayesian classification:")
print(p)
p, _ytp1 = Bayesian_accuracies(_xt1, _yt1, Mu1, Sigma1, nclass)
print("test_dataset_1 accuracy for Bayesian classification:")
print(p)
print()

Mu2, Sigma2 = params(_x2, _y2, nclass)
p, _yp2 = Bayesian_accuracies(_x2, _y2, Mu2, Sigma2, nclass)
print("train_dataset_2 accuracy for Bayesian classification:")
print(p)
p, _ytp2 = Bayesian_accuracies(_xt2, _yt2, Mu2, Sigma2, nclass)
print("test_dataset_2 accuracy for Bayesian classification:")
print(p)

# -------------------------------- Plots --------------------------------
mul = np.matmul
xx0 = np.arange(-2.5, 2.5, 0.1)

bb = 1/2*mul(mul(Mu1[0], np.linalg.inv(Sigma1)), np.transpose(Mu1[0])) - 1/2*mul(mul(Mu1[1], np.linalg.inv(Sigma1)), np.transpose(Mu1[1]))
aa = mul(np.linalg.inv(Sigma1), np.transpose(Mu1[0]-Mu1[1]))
xx1 = (-bb - xx0*aa[0])/aa[1]

plt.figure(1)

plt.subplot(2, 2, 1)
plt.plot(_x1[ np.logical_and(_y1[:, 0]==0, _yp1[:, 0]==0)][:, [0]], _x1[ np.logical_and(_y1[:, 0]==0, _yp1[:, 0]==0)][:, [1]], '.', label='Train_datasets_1 class_0 (T)')
plt.plot(_x1[ np.logical_and(_y1[:, 0]==0, _yp1[:, 0]==1)][:, [0]], _x1[ np.logical_and(_y1[:, 0]==0, _yp1[:, 0]==1)][:, [1]], 'o', label='Train_datasets_1 class_0 (F)')
plt.plot(_x1[ np.logical_and(_y1[:, 0]==1, _yp1[:, 0]==1)][:, [0]], _x1[ np.logical_and(_y1[:, 0]==1, _yp1[:, 0]==1)][:, [1]], '.', label='Train_datasets_1 class_1 (T)')
plt.plot(_x1[ np.logical_and(_y1[:, 0]==1, _yp1[:, 0]==0)][:, [0]], _x1[ np.logical_and(_y1[:, 0]==1, _yp1[:, 0]==0)][:, [1]], 'o', label='Train_datasets_1 class_1 (F)')
plt.plot(xx0, xx1, '-', label='closed form solution')
plt.title('Bayesian classification Train_datasets_1')
plt.ylabel('x1')
plt.xlabel('x0')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(_xt1[ np.logical_and(_yt1[:, 0]==0, _ytp1[:, 0]==0)][:, [0]], _xt1[ np.logical_and(_yt1[:, 0]==0, _ytp1[:, 0]==0)][:, [1]], '.', label='Test_datasets_1 class_0 (T)')
plt.plot(_xt1[ np.logical_and(_yt1[:, 0]==0, _ytp1[:, 0]==1)][:, [0]], _xt1[ np.logical_and(_yt1[:, 0]==0, _ytp1[:, 0]==1)][:, [1]], 'o', label='Test_datasets_1 class_0 (F)')
plt.plot(_xt1[ np.logical_and(_yt1[:, 0]==1, _ytp1[:, 0]==1)][:, [0]], _xt1[ np.logical_and(_yt1[:, 0]==1, _ytp1[:, 0]==1)][:, [1]], '.', label='Test_datasets_1 class_1 (T)')
plt.plot(_xt1[ np.logical_and(_yt1[:, 0]==1, _ytp1[:, 0]==0)][:, [0]], _xt1[ np.logical_and(_yt1[:, 0]==1, _ytp1[:, 0]==0)][:, [1]], 'o', label='Test_datasets_1 class_1 (F)')
plt.plot(xx0, xx1, '-', label='closed form solution')
plt.title('Bayesian classification Test_datasets_1')
plt.ylabel('x1')
plt.xlabel('x0')
plt.legend()

bb = 1/2*mul(mul(Mu2[0], np.linalg.inv(Sigma2)), np.transpose(Mu2[0])) - 1/2*mul(mul(Mu2[1], np.linalg.inv(Sigma2)), np.transpose(Mu2[1]))
aa = mul(np.linalg.inv(Sigma2), np.transpose(Mu2[0]-Mu2[1]))
xx1 = (-bb - xx0*aa[0])/aa[1]

plt.subplot(2, 2, 3)
plt.plot(_x2[ np.logical_and(_y2[:, 0]==0, _yp2[:, 0]==0)][:, [0]], _x2[ np.logical_and(_y2[:, 0]==0, _yp2[:, 0]==0)][:, [1]], '.', label='Train_datasets_2 class_0 (T)')
plt.plot(_x2[ np.logical_and(_y2[:, 0]==0, _yp2[:, 0]==1)][:, [0]], _x2[ np.logical_and(_y2[:, 0]==0, _yp2[:, 0]==1)][:, [1]], 'o', label='Train_datasets_2 class_0 (F)')
plt.plot(_x2[ np.logical_and(_y2[:, 0]==1, _yp2[:, 0]==1)][:, [0]], _x2[ np.logical_and(_y2[:, 0]==1, _yp2[:, 0]==1)][:, [1]], '.', label='Train_datasets_2 class_1 (T)')
plt.plot(_x2[ np.logical_and(_y2[:, 0]==1, _yp2[:, 0]==0)][:, [0]], _x2[ np.logical_and(_y2[:, 0]==1, _yp2[:, 0]==0)][:, [1]], 'o', label='Train_datasets_2 class_1 (F)')
plt.plot(xx0, xx1, '-', label='closed form solution')
plt.title('Bayesian classification Train_datasets_2')
plt.ylabel('x1')
plt.xlabel('x0')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(_xt2[ np.logical_and(_yt2[:, 0]==0, _ytp2[:, 0]==0)][:, [0]], _xt2[ np.logical_and(_yt2[:, 0]==0, _ytp2[:, 0]==0)][:, [1]], '.', label='Test_datasets_2 class_0 (T)')
plt.plot(_xt2[ np.logical_and(_yt2[:, 0]==0, _ytp2[:, 0]==1)][:, [0]], _xt2[ np.logical_and(_yt2[:, 0]==0, _ytp2[:, 0]==1)][:, [1]], 'o', label='Test_datasets_2 class_0 (F)')
plt.plot(_xt2[ np.logical_and(_yt2[:, 0]==1, _ytp2[:, 0]==1)][:, [0]], _xt2[ np.logical_and(_yt2[:, 0]==1, _ytp2[:, 0]==1)][:, [1]], '.', label='Test_datasets_2 class_1 (T)')
plt.plot(_xt2[ np.logical_and(_yt2[:, 0]==1, _ytp2[:, 0]==0)][:, [0]], _xt2[ np.logical_and(_yt2[:, 0]==1, _ytp2[:, 0]==0)][:, [1]], 'o', label='Test_datasets_2 class_1 (F)')
plt.plot(xx0, xx1, '-', label='closed form solution')
plt.title('Bayesian classification Test_datasets_2')
plt.ylabel('x1')
plt.xlabel('x0')
plt.legend()

plt.figure(2)

Z = estimated(_x1, _y1, Mu1, Sigma1, nclass)
ax = plt.subplot(111, projection='3d')
ax.scatter(_x1[_y1[:, 0]==0][:, 0], _x1[_y1[:, 0]==0][:, 1], Z[_y1[:, 0]==0][:, 0])
ax.scatter(_x1[_y1[:, 0]==1][:, 0], _x1[_y1[:, 0]==1][:, 1], Z[_y1[:, 0]==1][:, 1])
plt.title('Estimate dataset_1')
plt.ylabel('x1')
plt.xlabel('x0')

plt.figure(3)

Z = estimated(_x2, _y2, Mu2, Sigma2, nclass)
ax = plt.subplot(111, projection='3d')
ax.scatter(_x2[_y2[:, 0]==0][:, 0], _x2[_y2[:, 0]==0][:, 1], Z[_y2[:, 0]==0][:, 0])
ax.scatter(_x2[_y2[:, 0]==1][:, 0], _x2[_y2[:, 0]==1][:, 1], Z[_y2[:, 0]==1][:, 1])
plt.title('Estimate dataset_2')
plt.ylabel('x1')
plt.xlabel('x0')

plt.show()
