import numpy as np
from decimal import Decimal
from matplotlib import pyplot as plt
from scipy import signal
import math as m

def fisher(k_d, begin,end):
    #k_d = 7
    delt = (end - begin + 1) // k_d
    glsum = 0
    srarr = []
    for k in range(k_d - 1):
        ar = dffilt[(begin + k * delt): (begin + (k + 1) * delt)]
        sr = np.mean(ar)
        srarr.append(sr)
        glsum += sum([(ar[i] - sr) ** 2 for i in range(len(ar))])

    s_inta = glsum / k_d / (k_d - 1)
    x_sr = np.mean(srarr)

    s_inter = sum([(srarr[j] - x_sr) ** 2 for j in range(len(srarr))]) * k_d / (k_d - 1)
    print(round(s_inter / s_inta,4))
    return s_inter / s_inta


f = open('D:/Downloads/wave_ampl.txt', 'r')
lines = []
for line in f:
    lines.append(line)
lines[0] = lines[0].replace('[', '')
lines[0] = lines[0].replace(']', '')
data = lines[0].split(', ')
dataf = [float(data[i]) for i in range(len(data))]

t = np.linspace(0, len(dataf), len(dataf), endpoint=False)

# plt.plot(t, dataf)
# plt.show()
fig, ax = plt.subplots(1, 1)
t1024 = np.linspace(0, 1024, 1024, endpoint=False)
plt.plot(t1024, dataf[0+1024:1024*2])
plt.show()
#counts, bins = np.histogram(dataf[0+1024:(0+1024+1024)])
#ax.hist(bins[:-1], bins, weights=counts, alpha=0.5, color='g')

print(len(dataf))
dffilt = signal.medfilt(dataf[0 + 1024:1024 * 2], kernel_size=15)
plt.plot(t1024, dffilt,color='g')
plt.grid()
plt.show()
# plt.plot(283,dffilt[283], 'ro')
delta_array = [m.fabs(dffilt[i] - dffilt[i + 1]) for i in range(1023)]

ind = []
for i in range(len(delta_array)):
    if delta_array[i] > 0.005:
        ind.append(i)
a = [0, 258, 310, 744, 800, 1023]
# plt.plot(ind, dffilt[ind], 'ro') #1,258,310,744,800,1024

for i in range(len(a)-1):
    if (i == 0) or (i == 4):
        col = 'b'
        lab = 'Фон'
    if (i == 1) or (i == 3):
        col = 'g'
        lab = 'Переход'
    if i == 2:
        col = 'r'
        lab = 'Сигнал'
    plt.plot(np.linspace(a[i] + 1, a[i+1], a[i+1] - a[i] - 1, endpoint=True), dffilt[a[i]+1:a[i+1]], color=col, label = lab)

# типа поделить на 7 для каждого отднльно
k_d = 7
fisher(7,a[0],a[1])
fisher(4,a[1],a[2])
fisher(4,a[2],a[3])
fisher(4,a[3],a[4])
fisher(7,a[4],a[5])
plt.legend()
plt.grid()
plt.show()
