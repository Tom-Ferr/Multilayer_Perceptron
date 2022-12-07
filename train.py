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

def softmax(V):
    e = np.exp(V)
    return e / e.sum()

def hypothesis(x, thetas):
    return sigmoid(np.dot(x, thetas))

def stochastic_gradient_descent(X,Y):
    learning_rate = 1
    x = np.hstack((np.ones((X.shape[0],1)), X))
    m = x.shape[0]
    theta = []
    keys = np.unique(Y)
    keys_size = keys.shape[0]

    epoch = 0
    for k in keys:
        thetas = np.zeros(x.shape[1])
        y = np.where(Y == k, 1, 0)
        for it in range(m):
            print("\r\033[0Klearning in stochastic mode... {}%".format(int((((it + (m * epoch)) +1) / (m * keys_size))*100)), end="")
            gradient = x[it] * (hypothesis(x[it], thetas.T) - y[it])
            thetas -= learning_rate * gradient
        theta.append((thetas, k))
        epoch += 1
    print(end="\n")
    return theta

def predict(X, thetas, houses):
    x = np.hstack((np.ones((X.shape[0],1)), X))
    Y_hat = np.array([[sigmoid(i.dot(theta)) for theta in thetas] for i in x])
    Y_hat = np.array([houses[np.argmax(j)] for j in Y_hat])
    return Y_hat

def accuracy(y_hat, x):
    acc = np.where(y_hat == x, 1, 0)
    print("Training is completed with an accuracy of {}%".format(round(acc.sum() / len(y_hat), 4) * 100))

def treating_data(data):
    X = data[[1,2,4,5,8,9,12,14,15,22,24,25,28,29]].dropna()
    Y = X[1].to_numpy()
    X.drop(1, axis=1, inplace=True)
    # columns = X.columns.to_numpy()
    X_norm = normalization(X, X.min(), X.max())
    return train_test_split(X_norm,Y, test_size=0.2, random_state=1)

def feed_foward(X):
    hidden_nodes = 10
    output_labels = 2
    rows = X.shape[0]
    cols = X.shape[1]
    alpha = 0.01

    thetas_h1 = np.random.rand(cols, hidden_nodes)
    bias_h1 = np.random.rand(hidden_nodes)

    thetas_h2 = np.random.rand(cols, hidden_nodes)
    bias_h2 = np.random.rand(hidden_nodes)

    thetas_o = np.random.rand(hidden_nodes, output_labels)
    bias_o = np.random.rand(output_labels)

    """
    Phase One
    """
    zh1 = np.dot(X, thetas_h) + bias_h
    ah1 = sigmoid(zh)
    
    """
    Phase Two
    """
    zh2 = np.dot(zh1, thetas_h) + bias_h
    ah2 = sigmoid(zh2)

    """
    Phase Three
    """
    zo = np.dot(ah, thetas_o) + bias_o
    ao = softmax(zo)

def back_propagation():




"""
MAIN
"""

if __name__ ==  "__main__":

    """
    Read File
    """
    try:
        data = pd.read_csv("data.csv", header=None)
    except:
        print('File not found or it is corrupted.\nPlease, make sure that ./data.csv is available.')
        exit(1)

    """
    Code
    """ 
    try:
        X_train, X_test, Y_train, Y_test = treating_data(data)

        theta = stochastic_gradient_descent(X_train,Y_train)

        theta_dict = {column: row for row, column in theta}
        df_theta = pd.DataFrame(theta_dict)
        # df_theta.to_csv("thetas.csv", index=False)

        thetas = df_theta.to_numpy()
        houses = df_theta.columns.to_list()
        Y_hat = predict(X_test, thetas.T, houses)
        accuracy(Y_hat, Y_test)
    except Exception as e:
        print('Please, make sure that {} is well formatted'.format(sys.argv[1]))
        print(e)
        exit(2)