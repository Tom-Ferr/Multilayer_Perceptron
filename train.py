import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

def normalization(value, mini, maxi):
    return (value - mini) / (maxi -  mini)

def linear_interpolation(value, mini, maxi):
    return (maxi - mini) * value + mini

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def hypothesis(x, thetas):
    return sigmoid(np.dot(x, thetas))

def stochastic_gradient_descent(X,Y):
    learning_rate = 0.1
    x = np.hstack((np.ones((X.shape[0],1)), X))
    m = x.shape[0]
    theta = []

    epoch = 0
    for house in np.unique(Y):
        thetas = np.zeros(x.shape[1])
        y = np.where(Y == house, 1, 0)
        for it in range(m):
            print("\r\033[0Klearning in stochastic mode... {}%".format(int((((it + (m * epoch)) +1) / (m * 4))*100)), end="")
            gradient = x[it] * (hypothesis(x[it], thetas.T) - y[it])
            thetas -= learning_rate * gradient
        theta.append((thetas, house))
        epoch += 1
    print(end="\n")
    return theta


"""
MAIN
"""

if __name__ ==  "__main__":
    try:
        data = pd.read_csv("data.csv")
    except:
        print('File not found or it is corrupted.\nPlease, make sure that ./data.csv is available.')
        exit(1)