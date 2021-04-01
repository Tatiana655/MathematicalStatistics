import numpy as np
from decimal import Decimal
from matplotlib import pyplot as plt
from scipy import signal
import math as m

NUM = 543  # номер сигнала с 0 нумерация сигналов
MXXIV = 1024  # вообще, так лучше не делать, но кто это будет читать?)
MAX_SPLIT = 40
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

# возвращает крайние индексы
def ret_left_right(ind):
    for i in range(len(ind) - 1):
        if ind[i + 1] - ind[i] > 5:
            left = ind[i]
            right = ind[i + 1]
            return left, right


def draw_hist_signal(a, step):
    fig, ax = plt.subplots(1, 1)
    counts, bins = np.histogram(dataf[MXXIV * NUM: MXXIV * (NUM + 1)], bins=step)
    ax.hist(bins[:-1], bins, weights=counts, alpha=0.6, color='g')

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
        plt.plot(dffilt[a[i] + 1:a[i + 1]], np.linspace(a[i] + 1, a[i + 1], a[i + 1] - a[i] - 1),  color=col, label=lab)
    plt.legend()
    plt.grid()
    plt.show()
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
#fig, ax = plt.subplots(1, 1)
#t1024 = np.linspace(0, MXXIV, MXXIV, endpoint=False)
#plt.plot(t1024, dataf[MXXIV * NUM: MXXIV * (NUM + 1)])
#plt.grid()
#plt.show()

# print histogram
fig1, ax1 = plt.subplots(1, 1)
counts, bins = np.histogram(dataf[MXXIV * NUM: MXXIV * (NUM + 1)], bins = MAX_SPLIT)
ax1.hist(bins[:-1], bins, weights=counts, alpha=0.6, color='g')
plt.grid()
plt.show()

# print filtered signal
dffilt = signal.medfilt(dataf[MXXIV * NUM: MXXIV * (NUM + 1)], kernel_size=11)
#plt.plot(np.linspace(0, MXXIV - 1, MXXIV, endpoint=False), dffilt, color='g')
#plt.grid()
#plt.show()

# searsh for boundaries. ONLY for bottom signals
counts40, bins40 = np.histogram(dataf[MXXIV * NUM: MXXIV * (NUM + 1)], bins = MAX_SPLIT)
counts10, bins10 = np.histogram(dataf[MXXIV * NUM: MXXIV * (NUM + 1)], bins = 10)

# индексы сигналов. Отнять крайние правые
ind_signal10 = find_ind(dataf[MXXIV * NUM: MXXIV * (NUM + 1)], bins10[0], bins10[1])
ind_signal40 = find_ind(dataf[MXXIV * NUM: MXXIV * (NUM + 1)], bins40[0], bins40[1])
# индексы фона найти границы (левые, правые) потом отнять
ind_ground10 = find_ind(dataf[MXXIV * NUM: MXXIV * (NUM + 1)], bins10[len(bins10) - 2], bins10[len(bins10) - 1])
ind_ground40 = find_ind(dataf[MXXIV * NUM: MXXIV * (NUM + 1)], bins40[len(bins40) - 2], bins40[len(bins40) - 1])

l10,r10=ret_left_right(ind_ground10 )
a10 = [0, l10, ind_signal10[0], ind_signal10[len(ind_signal10) - 1], r10, MXXIV - 1]
draw_hist_signal(a10, 10)

l40,r40=ret_left_right(ind_ground40 )
a40 = [0, l40, ind_signal40[0], ind_signal40[len(ind_signal40) - 1], r40, MXXIV - 1]
draw_hist_signal(a40, MAX_SPLIT)

a_split = [0, l40, l10, ind_signal10[0], ind_signal40[0], ind_signal40[len(ind_signal40) - 1],  ind_signal10[len(ind_signal10) - 1], r10, r40, MXXIV - 1]

fig1, ax1 = plt.subplots(1, 1)
for i in range(0,len(a_split) - 1,2):
    if (i == 0) or (i == 4*2):
        col = 'b'
        lab = 'Фон'
    if (i == 1*2) or (i == 3*2):
        col = 'g'
        lab = 'Переход'
    if i == 2*2:
        col = 'r'
        lab = 'Сигнал'
    plt.plot(np.linspace(a_split[i] + 1, a_split[i + 1], a_split[i + 1] - a_split[i] - 1),dffilt[a_split[i] + 1:a_split[i + 1]],  color=col, label=lab)
plt.grid()
plt.legend()
plt.show()
#fig1, ax1 = plt.subplots(1, 1)
fisher(7, a_split[0], a_split[1])
fisher(4, a_split[2], a_split[3])
fisher(5, a_split[4], a_split[5])
fisher(4, a_split[6], a_split[7])
fisher(4, a_split[8], a_split[9])
#построить 0-1(фон) 2-3(переход) 4-5 (сигнал) 6-7 (переход) 8-9 (фон) и потом подбирать n
#ax1.hist(bins[:-1], bins, weights=counts, alpha=0.6, color='g')





