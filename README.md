# Logistic, Softmax Regression(Multiclass Classification) and Bayesian Classification

## Setup and run:

* Install python3

* Install python library:
```bash
$ pip install pandas
$ pip install numpy
$ pip install matplotlib
```

* Clone and run:
```bash
$ git clone https://github.com/VakhshooriEhsan/SPR-Fall2020-HW2-Logistic-SoftmaxRegression-MC--BayesianClassification.git
$ cd SPR-Fall2020-HW2-Logistic-SoftmaxRegression-MC--BayesianClassification/src
$ python PartA-Logistic-SoftmaxRegression(MC).py
$ python PartB-Bayesian-classification.py
```

## A. Logistic, Softmax Regression (Multiclass Classification)

### 1. Reading datas:

* Reading datas by `readData(addr)` function from `./Datas/iris.data` and saving them as a numpy array.
* Finding X, Y of train and test datas.

### 2. Train  multiclass  classification  (one-vs.-one  and one-vs.-all)  by  logistic  regression:

* The result of train and test accuracy:

```
train accuracy for one-vs-one logistic regression:
0.8916666666666667
test accuracy for one-vs-one logistic regression:
1.0

train accuracy for one-vs-all logistic regression:
0.8
test accuracy for one-vs-all logistic regression:
0.9666666666666667
```

* Plot of cost function for one-vs-all method:
```
img
```

### 3. Train multiclass classification by softmax regression:

* The result of train and test accuracy:

```
train accuracy for softmax logistic regression:
0.9
test accuracy for softmax logistic regression:
1.0
```

## B. Linear regression

### 1. Reading datas:

* Reading datas by `readData(addr)` function from `./Datas/BC-Train1.csv`, `./Datas/BC-Test1.csv`, `./Datas/BC-Train2.csv`, `./Datas/BC-Test2.csv` and saving them as a numpy array.
* Finding X1, Y1, X2, Y2 of train and test datas.

### 2. Train classification by Bayesian classifier:

* The result of train and test accuracy:

```
train_dataset_1 accuracy for Bayesian classification:
0.99
test_dataset_1 accuracy for Bayesian classification:
1.0

train_dataset_2 accuracy for Bayesian classification:
0.99125
test_dataset_2 accuracy for Bayesian classification:
1.0
```

* Plot of the decision boundary and classification results:
```
img
```

* Plot of estimate:
```
img
```

