import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys

def split_in_groups(data):
    groups = data.dropna().groupby(1)

    M = groups.get_group('M').drop(1, axis=1, inplace=False)
    M.rename(columns={0: 1}, inplace=True)

    B = groups.get_group('B').drop(1, axis=1, inplace=False)
    B.rename(columns={0: 1}, inplace=True)
    return M,B

def plot_histogram(M,B):
    fig, ax = plt.subplots(4,8)
    i = j = 0
    for col in range(1,32):
        ax[i][j].hist(M[col], bins=15, edgecolor='black',color='red', alpha=0.3)
        ax[i][j].hist(B[col], bins=15, edgecolor='black',color='cyan', alpha=0.3)
        i = (i+1)%4 
        if i == 0:
            j += 1
    fig.delaxes(ax[3,7])
    plt.show()

def plot_scatter(M,B):
    fig, ax = plt.subplots(4,8)
    i = j = 0
    for col in range(1,32):
        ax[i][j].scatter(M.index, M[col], edgecolor='black',color='red', alpha=0.3)
        ax[i][j].scatter(B.index, B[col], edgecolor='black',color='cyan', alpha=0.3)
        i = (i+1)%4 
        if i == 0:
            j += 1
    fig.delaxes(ax[3,7])
    plt.show()

def plot_matrix(M,B):
    fig, ax = plt.subplots(4,8)
    i = j = 0
    try:
        if len(sys.argv) >= 3:
            k = int(sys.argv[2])
        else:
            k = 1
        for col in range(1,32):
            ax[i][j].scatter(M[k], M[col], edgecolor='black',color='red', alpha=0.3)
            ax[i][j].scatter(B[k], B[col], edgecolor='black',color='cyan', alpha=0.3)
            i = (i+1)%4 
            if i == 0:
                j += 1
    except:
        print("Please, the 3rd argument must be an integer from 1 to 31")
        exit(3)
    fig.delaxes(ax[3,7])
    plt.show()



"""
MAIN
"""

if __name__ ==  "__main__":
    if len(sys.argv) < 2:
        print("An argument is needed. Please run this with \'--help\' for more information")
        exit(1)
    if sys.argv[1] == "--help":
            print('''
            [1stARG=Program Name][2ndARG=Data View Mode][3rdARG=# 1-31](optional)\n
            \tPossible inputs as 2nd parameter:\n
            \t\t1: -h or --histogram\n
            \t\t2: -s or --scatter\n
            \t\t3: -m or --matrix\n
            \t 3rd argument should be used only with \'--matrix\' mode
            \t\t it must be an integer from 1 to 31
            \t\t if no 3rd argument is passed when matrix mode is selected, then the default will be 1
            ''')
            exit(0)
    try:
        data = pd.read_csv("data.csv", header=None)
    except:
        print('File not found or it is corrupted.\nPlease, make sure that ./data.csv is available.')
        exit(2)
    try:

        M,B = split_in_groups(data)

        if (sys.argv[1] == '--histogram' or sys.argv[1] == '-h'):
            plot_histogram(M,B)
        elif (sys.argv[1] == '--scatter' or sys.argv[1] == '-s'):
            plot_scatter(M,B)
        elif (sys.argv[1] == '--matrix' or sys.argv[1] == '-m'):
            plot_matrix(M,B)
        else:
            print("Wrong argument was passed.\nPlease, run this with --help for more information")
    except Exception as e:
        print(e)
        exit(4)