import numpy as np
from decimal import Decimal
from matplotlib import pyplot as plt
from scipy import signal
import math as m

NUM = 2  # номер сигнала с 0 нумерация сигналов
MXXIV = 1024  # вообще, так лучше не делать, но кто это будет читать?)

#находит индексы элементов в определённых границах
def find_ind(data, left, right):
    far = []
    this_data = data
    for i in range(MXXIV):
        if left <= this_data[i] <= right or left >= this_data[i] >= right:
            far.append(i)
    return far

#собсна сам F Фишер
def fisher(k_d, begin, end):
    delt = (end - begin + 1) // (k_d)
    glsum = 0
    meanarr = []
    for k in range(k_d):
        ar = dffilt[(begin + k * delt): (begin + (k + 1) * delt) - 1]
        meanarr.append(np.mean(ar))
        glsum += sum([(ar[i] - np.mean(ar)) ** 2 for i in range(len(ar))])

    s_inta = glsum / k_d / (k_d - 1)
    x_gl_mean = np.mean(meanarr)

    s_inter = sum([(meanarr[j] - x_gl_mean) ** 2 for j in range(len(meanarr))]) * k_d / (k_d - 1)
    print(round(s_inter / s_inta, 4))
    return s_inter / s_inta


# чтение из файла данных и перевод их в человеческий вид (double)
f = open('D:/Downloads/wave_ampl.txt', 'r')
lines = []
for line in f:
    lines.append(line)
lines[0] = lines[0].replace('[', '')
lines[0] = lines[0].replace(']', '')
data = lines[0].split(', ')
dataf = [float(data[i]) for i in range(len(data))]

# print signal
fig, ax = plt.subplots(1, 1)
t1024 = np.linspace(0, MXXIV, MXXIV, endpoint=False)
plt.plot(t1024, dataf[MXXIV * NUM: MXXIV * (NUM + 1)])
plt.grid()
plt.show()

# print histogram
fig1, ax1 = plt.subplots(1, 1)
counts, bins = np.histogram(dataf[MXXIV * NUM: MXXIV * (NUM + 1)])
ax1.hist(bins[:-1], bins, weights=counts, alpha=0.6, color='g')
plt.grid()
plt.show()

# print filtered signal
dffilt = signal.medfilt(dataf[MXXIV * NUM: MXXIV * (NUM + 1)], kernel_size=5)
plt.plot(np.linspace(0, MXXIV - 1, MXXIV, endpoint=False), dffilt, color='g')
plt.grid()
plt.show()

# searsh for boundaries. ONLY for bottom signals
ind_signal = find_ind(dataf[MXXIV * NUM: MXXIV * (NUM + 1)], bins[0], bins[1])
ind_ground = find_ind(dataf[MXXIV * NUM: MXXIV * (NUM + 1)], bins[len(bins) - 2], bins[len(bins) - 1])

for i in range(len(ind_ground) - 1):
    if ind_ground[i + 1] - ind_ground[i] > 1:
        left = ind_ground[i]
        right = ind_ground[i + 1]

a = [0, left, ind_signal[0], ind_signal[len(ind_signal) - 1], right, MXXIV - 1]

for i in range(len(a) - 1):
    if (i == 0) or (i == 4):
        col = 'b'
        lab = 'Фон'
    if (i == 1) or (i == 3):
        col = 'g'
        lab = 'Переход'
    if i == 2:
        col = 'r'
        lab = 'Сигнал'
    plt.plot(np.linspace(a[i] + 1, a[i + 1], a[i + 1] - a[i] - 1), dffilt[a[i] + 1:a[i + 1]], color=col, label=lab)

fisher(7, a[0], a[1])
fisher(4, a[1], a[2])
fisher(4, a[2], a[3])
fisher(4, a[3], a[4])
fisher(6, a[4], a[5])
plt.legend()
plt.grid()
plt.show()
