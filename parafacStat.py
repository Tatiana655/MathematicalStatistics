import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy as sp
import tensorly.decomposition as decomp
import math as m
import itertools

def read_file(name):

    f = open(name, 'r')

    text = f.read()
    text = text.replace(",",".")
    split_text = text.split("\n")
    for i, str in enumerate(split_text):
        split_text[i] = str.split("\t")
    x_ax = split_text[3][1:len(split_text[3])]
    data = split_text[4:len(split_text)]
    y_ax = []
    for i in range(len(data)):
        y_ax.append(data[i][0])
        data[i]=data[i][1:len(data[i])]

    x_ax = [int(float(x_ax[i])) for i in range(len(x_ax))]
    y_ax = [int(float(y_ax[i])) for i in range(len(y_ax)-1)]
    data = [[float(data[i][j]) for i in range(len(data)-1)] for j in range(len(data[0]))]
    return np.array(x_ax), np.array(y_ax), np.array(data)

def get_ax_data(y,x):
    x_all = []
    for i in range(len(y)):
        x_all.append(x)
    y_all = np.zeros_like(x_all).transpose()
    x_all = np.asarray(x_all).reshape(-1)

    for i in range(len(y_all)):
        y_all[i][:] = y
    y_all = y_all.transpose()
    y_all = np.asarray(y_all).reshape(-1)
    return x_all,y_all

def show_data(x,y,z,fig):
    Z = np.array(z).reshape(1, len(z) * len(z[0]))
    X, Y = get_ax_data(x, y)
    my_col = cm.jet(Z[0] / np.amax(Z))

    ax = fig.add_subplot(projection='3d')
    ax.scatter(X, Y, Z, marker=".", color=my_col)
    #plt.show()

def erase_first_line(x,y,z):
    delt2 = 20
    for i in range(len(x)):
        deg = 5
        # plt.plot(y,z[i], "r.",label="data")
        # print(x[i],y[i]+ i)
        # тут используется y = x
        # план: запоминаем индексы потом вставляем точки
        ind = []
        # левый конец
        if (y[i] + i - delt2 >= 250):
            ind.append(np.where(np.logical_and(y >= 250, y <= y[i] + i - delt2))[0])
        else:
            deg = 2
            delt2 = 20
        # правый конец
        ind.append(np.where(np.logical_and(y >= y[i] + i + delt2, y <= y[i] + i + 2 * delt2))[0])
        ind = np.array(ind)[0]

        fp = np.polyfit(y[ind], z[i][ind], deg)

        ind_line = np.where(np.logical_and(y > y[i] + i - delt2, y < y[i] + i + delt2))[0]
        f = sp.poly1d(fp)
        if (np.where(f(y[ind_line]) < -1)):
            fp = np.polyfit(y[ind], z[i][ind], 2)
            f = sp.poly1d(fp)
        # print(f(y[ind_line]))
        # plt.plot(y[ind_line], f(y[ind_line]), "g.", label="LS")
        for j in ind_line:
            z[i][j] = max(0, f(y[j]))
        # plt.plot(y, z[i], "r.", label="data")
        # plt.title("Фронтальный срез данных")
        # plt.grid()
        # plt.legend()
        # plt.show()
    return  z
# Вообще можно и одну функцию сделать, но у меня горят деделайны, поэтому оставила до лучших времён (оптимистично было бы написать in process:) )
def erase_second_line(x,y,z):
    delt2 = 15
    for i in range(len(x)):
        if x[i] == 312:
            break
        deg = 5

        # plt.plot(y,z[i], "r.",label="data")

        # print(x[i],2 * x[i])
        # тут используется y = 2 * x
        # план: запоминаем индексы потом вставляем точки
        ind = []
        # правый конец
        if (2 * x[i] + delt2 <= 600):
            ind.append(np.where(np.logical_and(y >= 2 * x[i] + delt2, y <= 600))[0])
        else:
            deg = 1
            delt2 = 15
        # левый конец
        ind.append(np.where(np.logical_and(y <= 2 * x[i] - delt2, y >= 2 * x[i] - 2 * delt2))[0])
        if len(ind) == 2:
            ind = np.concatenate((np.array(ind[1]).transpose(), (np.array(ind[0])).transpose()))
        else:
            ind = ind[0]
        fp = np.polyfit(y[ind], z[i][ind], deg)
        # индексы линии
        ind_line = np.where(np.logical_and(y > 2 * x[i] - delt2, y < 2 * x[i] + delt2))[0]
        f = sp.poly1d(fp)
        if (np.where(f(y[ind_line]) < -1)):
            fp = np.polyfit(y[ind], z[i][ind], 2)
            f = sp.poly1d(fp)
        # print(f(y[ind_line]))
        # plt.plot(y[ind_line], f(y[ind_line]), "g.", label="LS")
        for j in ind_line:
            z[i][j] = max(0, f(y[j]))
        # plt.plot(y, z[i], "r.", label="data")
        # plt.title("Фронтальный срез данных")
        # plt.grid()
        # plt.legend()
    # plt.show()
    return z

def show_proj(factors, x, ex1_em0, rank):
    color = ["r", "g", "b", "m", "c", "k", "y"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(rank):

        mat1 = np.array(factors[1][1]).transpose()[i]
        w = 1#norm(np.transpose(factors[1][0])[i])

        sq_mat = np.array(mat1)
        m = np.max(sq_mat)
        ind = np.where(sq_mat == m)
        if ex1_em0 == 1:
            # fx = np.linspace(250, 450, 1000)
            x = x
            y = w * np.transpose(sq_mat[ind[0]])
        else:
            # fx = np.linspace(240, 300, 1000)
            x = x
            y = w * np.transpose(np.transpose(sq_mat)[ind[1]])

        plt.grid()
        ax.plot(x, y, color[i])
        ax.set_ylabel("Intensity")


    plt.grid()
    plt.savefig("factor_proj"+ str(ex1_em0) + str(rank) + '.png')


eps1 = 1
step = 15
delt = 50

#кто-то где-то потерялся мб, но размеры вроде совпадают
default_way = "C:\\Users\\Tatiana\\Desktop\\ToBazhenov\\VD_DOM_Permafrost\\"
#4, 9,
#names = ["1701", "1702","1704","1706","1708_1to10","1708_1to20", "1711", "1712","1727", "1729","1730","1732", "1733","1734"]
names = ["1701", "1702","1704", "1711", "1729"]
tens = []
x = 0
y = 0
delt1 = 10

for i in range(len(names)):
    x,y,z = read_file(default_way + names[i] + '.txt')
    z = erase_first_line(x, y, z)
    z = erase_second_line(x, y, z)
    #fig = plt.figure(figsize=(5, 5))
    #show_data(x, y, z, fig)
    tens.append(z)

#plt.show()
# получение факторв
rank = 6
factors = decomp.parafac2(np.array(tens),rank,normalize_factors=True)
fig = plt.figure(figsize=(5, 5))
print(factors[0])
for i in range(rank):
    z_f1 = np.transpose(factors[1][2])[i]
    plt.plot(y,z_f1)

plt.show()
#show_proj(factors, x, 1, 3)

# print all data
'''for i in range(len(names)):
    x,y,z = read_file(default_way + names[i] +'.txt')
    fig = plt.figure(figsize=(5, 5))
    show_data(x,y,z,fig)
plt.show()'''
'''x,y,z = read_file(default_way + names[0] +'.txt')
for i in range(1):
    i = 50
    y_use = y[max(0,i+delt-step):min(i+delt+step, len(x))]
    z_use = z[i][max(0,i+delt-step):min(i+delt+step, len(x))]
    plt.plot(y_use,z_use,"r.")
    fig, ax = plt.subplots(1, 1)
    counts, bins = np.histogram(z_use, bins=25)
    ax.hist(bins[:-1], bins, weights=counts, alpha=0.6, color='g')
    plt.show()
    z_mean =  np.where(np.logical_and(z_use >= bins[0], z_use <= bins[1]))[0]
    z_mean = np.mean(z_mean)

    n_max = np.argmax(counts)
    for ind,el in enumerate(z_use):
        if (el > bins[n_max+1]):
            z[i][ind] = z_mean
    plt.plot(y,z[i],"g.")
fig = plt.figure(figsize=(5, 5))
show_data(x, y, z, fig)
plt.show()'''

