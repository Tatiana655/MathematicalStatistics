import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorly.decomposition import non_negative_parafac
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

eps1 = 1
step = 15
delt = 50

#кто-то где-то потерялся мб, но размеры вроде совпадают
default_way = "C:\\Users\\Tatiana\\Desktop\\ToBazhenov\\VD_DOM_Permafrost\\"
names = ["1701", "1702","1704","1706","1708_1to10","1708_1to20", "1711", "1712","1727", "1729","1730","1732", "1733","1734"]
tens = []
x = 0
y = 0
for i in range(len(names)):
    x,y,z = read_file(default_way + names[i] + '.txt')
    tens.append(z)

factors = non_negative_parafac(np.array(tens),rank=5, n_iter_max=2000,
                                                    tol=10e-6)
fig = plt.figure(figsize=(5, 5))
#show_data(x,y,factors[0],fig)
#print(factors[1][0])
z_f1 = np.transpose(factors[1][2])[0]
plt.plot(y,z_f1)
plt.show()
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




plt.show()
