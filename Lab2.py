from scipy.stats import norm, laplace, poisson, cauchy, uniform
import numpy as np
import matplotlib.pyplot as plt
import math as m

sizes = [10, 100, 1000]


def z_R(selection, size):
    z_R = (selection[0] + selection[size - 1]) / 2
    return z_R

def z_p(selection, np):
    if np.is_integer():
        return selection[int(np)]
    else:
        return selection[int(np) + 1]

def z_Q(selection, size):
    z_1 = z_p(selection, size / 4)
    z_2 = z_p(selection, 3 * size / 4)
    return (z_1 + z_2) / 2

def z_tr(selection, size):
    r = int(size / 4)
    sum = 0
    for i in range(r + 1, size - r + 1):
        sum += selection[i]
    return (1 / (size - 2 * r)) * sum

def NormalNumbers():
    for size in sizes:
        mean_list, med_list, z_R_list, z_Q_list, z_tr_list = [], [], [], [], []
        all_list = [mean_list, med_list, z_R_list, z_Q_list, z_tr_list]
        E, D = [], []
        for i in range(1000):
            distribution = norm.rvs(size=size)
            distribution.sort()
            mean_list.append(np.mean(distribution))
            med_list.append(np.median(distribution))
            z_R_list.append(z_R(distribution, size))
            z_Q_list.append(z_Q(distribution, size))
            z_tr_list.append(z_tr(distribution, size))
        for lis in all_list:
            E_1 = round(np.mean(lis),6)
            D_1 = round(np.std(lis) ** 2, 6)
            print("n = ", size)
            print("E(z) = ", E_1)
            print("D(z) = ", D_1)
            print("E(z) - sqrt(D(z)) = ",  round(E_1 - D_1 ** 0.5, 6))
            print("E(z) + sqrt(D(z)) = ",  round(E_1 + D_1 ** 0.5, 6))


def CauchyNumbers():
    for size in sizes:
        mean_list, med_list, z_R_list, z_Q_list, z_tr_list = [], [], [], [], []
        all_list = [mean_list, med_list, z_R_list, z_Q_list, z_tr_list]
        E, D = [], []
        for i in range(1000):
            distribution = cauchy.rvs(size=size)
            distribution.sort()
            mean_list.append(np.mean(distribution))
            med_list.append(np.median(distribution))
            z_R_list.append(z_R(distribution, size))
            z_Q_list.append(z_Q(distribution, size))
            z_tr_list.append(z_tr(distribution, size))
        for lis in all_list:
            E_1 = round(np.mean(lis), 6)
            D_1 = round(np.std(lis) ** 2, 6)
            print("n = ", size)
            print("E(z) = ", E_1)
            print("D(z) = ", D_1)
            print("E(z) - sqrt(D(z)) = ", round(E_1 - D_1 ** 0.5, 6))
            print("E(z) + sqrt(D(z)) = ", round(E_1 + D_1 ** 0.5, 6))

def LaplaceNumbers():
    for size in sizes:
        mean_list, med_list, z_R_list, z_Q_list, z_tr_list = [], [], [], [], []
        all_list = [mean_list, med_list, z_R_list, z_Q_list, z_tr_list]
        E, D = [], []
        for i in range(1000):
            distribution = laplace.rvs(size=size, scale=1 / m.sqrt(2), loc=0)
            distribution.sort()
            mean_list.append(np.mean(distribution))
            med_list.append(np.median(distribution))
            z_R_list.append(z_R(distribution, size))
            z_Q_list.append(z_Q(distribution, size))
            z_tr_list.append(z_tr(distribution, size))
        for lis in all_list:
            E_1 = round(np.mean(lis), 6)
            D_1 = round(np.std(lis) ** 2, 6)
            print("n = ", size)
            print("E(z) = ", E_1)
            print("D(z) = ", D_1)
            print("E(z) - sqrt(D(z)) = ", round(E_1 - D_1 ** 0.5, 6))
            print("E(z) + sqrt(D(z)) = ", round(E_1 + D_1 ** 0.5, 6))

def PoissonNumbers():
    for size in sizes:
        mean_list, med_list, z_R_list, z_Q_list, z_tr_list = [], [], [], [], []
        all_list = [mean_list, med_list, z_R_list, z_Q_list, z_tr_list]
        E, D = [], []
        for i in range(1000):
            distribution = poisson.rvs(10, size=size)
            distribution.sort()
            mean_list.append(np.mean(distribution))
            med_list.append(np.median(distribution))
            z_R_list.append(z_R(distribution, size))
            z_Q_list.append(z_Q(distribution, size))
            z_tr_list.append(z_tr(distribution, size))
        for lis in all_list:
            E_1 = round(np.mean(lis), 6)
            D_1 = round(np.std(lis) ** 2, 6)
            print("n = ", size)
            print("E(z) = ", E_1)
            print("D(z) = ", D_1)
            print("E(z) - sqrt(D(z)) = ", round(E_1 - D_1 ** 0.5, 6))
            print("E(z) + sqrt(D(z)) = ", round(E_1 + D_1 ** 0.5, 6))

def UniformNumbers():
    for size in sizes:
        mean_list, med_list, z_R_list, z_Q_list, z_tr_list = [], [], [], [], []
        all_list = [mean_list, med_list, z_R_list, z_Q_list, z_tr_list]
        E, D = [], []
        for i in range(1000):
            distribution = uniform.rvs(size=size, loc=-m.sqrt(3), scale=2 * m.sqrt(3))
            distribution.sort()
            mean_list.append(np.mean(distribution))
            med_list.append(np.median(distribution))
            z_R_list.append(z_R(distribution, size))
            z_Q_list.append(z_Q(distribution, size))
            z_tr_list.append(z_tr(distribution, size))
        for lis in all_list:
            E_1 = round(np.mean(lis), 6)
            D_1 = round(np.std(lis) ** 2, 6)
            print("n = ", size)
            print("E(z) = ", E_1)
            print("D(z) = ", D_1)
            print("E(z) - sqrt(D(z)) = ", round(E_1 - D_1 ** 0.5, 6))
            print("E(z) + sqrt(D(z)) = ", round(E_1 + D_1 ** 0.5, 6))

UniformNumbers()