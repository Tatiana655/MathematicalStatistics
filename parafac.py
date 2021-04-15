import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time
import numpy as np
from matplotlib import cm
from tensorly.random import random_cp
from tensorly.decomposition import CP, non_negative_parafac
from utils import *
import seaborn as sns

from sklearn.decomposition import FactorAnalysis, PCA

import tensorly as tl
from tensorly import unfold as tl_unfold


import scipy.io

def transform_data(matlab_data):
    ex = []
    for i in range(len(matlab_data["ExAx"][0])):
        ex.append(matlab_data["EmAx"][0])
    ex = np.array(ex)
    em = np.zeros_like(ex)
    ex = np.asarray(ex).reshape(-1)

    for i in range(len(em)):
        em[i][:] = matlab_data["ExAx"][0][i]

    em = np.asarray(em).reshape(-1)
    return ex,em

# рисование исходных данных
def show_data(matlab_data):
    for i in range(5):
        my_col = cm.jet(matlab_data["X"][i] / np.amax(matlab_data["X"][0]))
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(projection='3d')
        ex,em = transform_data(matlab_data)
        ax.scatter(ex, em, matlab_data["X"][i], marker=".", color=my_col)  # plot the point (2,3,4) on the figure
    plt.show()

# ривование факторов
def show_factors(factors,ex,em):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(3):
        mat1 = np.array(factors[1][1]).transpose()[i]
        w = tl.norm(factors[1][0][i])
        my_col = cm.jet(w * mat1 / np.amax(w * mat1))

        ax.scatter(ex,em, w * mat1, marker=".",color=my_col) # plot the point (2,3,4) on the figure
    plt.show()

# проекция на ось Z по высоте
def show_z(factors,ex,em):
    # ax = fig.add_subplot(111, projection='3d')
    all_ind = []
    for i in range(3):
        fig = plt.figure()
        mat1 = np.array(factors[1][1]).transpose()[i]
        w = tl.norm(factors[1][0][i])
        my_col = cm.jet(w * mat1 / np.amax(w * mat1))
        # ax.scatter(ex,em, tl.zeros_like(w*mat1), marker=".",color=my_col) # plot the point (2,3,4) on the figure
        sq_mat = np.array(mat1).reshape(61,201)
        m = np.max(sq_mat)
        ind = tl.where(sq_mat == m)

        plt.scatter(ex, em, marker=".", color=my_col)
        plt.scatter(ex[0]+ind[1], em[0]+ind[0], marker="*", color="w", label="max")
        all_ind.append([ex[0]+ind[1],em[0]+ind[0]])
        plt.legend()
    plt.show()
    return all_ind

def show_proj(factors,matlab_data,ex1_em0):
    color = ["r","g","b"]
    for i in range(3):
        mat1 = np.array(factors[1][1]).transpose()[i]
        w = tl.norm(factors[1][0][i])

        sq_mat = np.array(mat1).reshape(61, 201)
        m = np.max(sq_mat)
        ind = tl.where(sq_mat == m)
        if (ex1_em0 == 1):
            x = matlab_data["EmAx"][0]
            y = w * tl.transpose(sq_mat[ind[0]])
        else:
            x = matlab_data["ExAx"][0]
            y = w * np.transpose(tl.transpose(sq_mat)[ind[1]])
        plt.grid()
        plt.plot(x,  y,color[i])
    plt.show()


if __name__ == "__main__":
    mat = scipy.io.loadmat('C:/Users/Tatiana/Desktop/amino.mat')

    ex,em = transform_data(mat)
    factors = non_negative_parafac(mat["X"], rank=3,n_iter_max=2000)
    show_proj(factors,mat,0)
    show_proj(factors, mat, 1)
