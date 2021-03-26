import scipy.stats as stats
import math as m
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import openpyxl

alpha = 0.05
k=6 # но на самом деле 8

def mle(sample):
    f = lambda theta: -sum([m.log(m.fabs(stats.norm.pdf(x, *theta))) for x in sample])
    return minimize(f,[1,1]).x

def grid_generator(n):
    #k = math.ceil(1.72* (n**(1/3)))
    k = 3
    l_b, r_b = -1.1, 1.1
    step = (r_b-l_b)/k
    res = [(-m.inf, l_b)]
    res += [(l_b + step*i, l_b+step*(i+1)) for i in range(k)]
    res += [(r_b, m.inf)]
    return res

def n_i(col):
    new_col = []
    c = 0
    for cut in col:
        new_col.append(0)
        for i in sample:
            new_col[c] += 0 if (i <= cut[0] or i >= cut[1]) else 1
        c += 1
    return new_col

def p_i(col, F):
    new_col = []
    c = 0
    for cut in col:
        new_col.append(F(cut[1]) - F(cut[0]))
    return new_col
'''
sample = np.array(stats.norm.rvs(0, 1, size=100))
mu, sig = mle(sample)
print(mu, sig )

hipotise = lambda x: stats.norm.cdf(x,mu, sig)
frame = []
#frame = pd.DataFrame(columns=['Границы'])
frame.append(grid_generator(len(sample)))
frame.append(n_i(frame[0]))
frame.append(p_i(frame[0], hipotise))
frame.append(len(sample) * np.array(frame[2]))
frame.append(np.array(frame[1]) - np.array(frame[3]))
frame.append((frame[1] - frame[3])**2 / (frame[3]))

#frame = frame.round(4)
print(frame)
print(sum(frame[5]))
#frame.to_csv('table.csv')
p = pd.DataFrame(frame)
p.to_excel('t.xlsx', index=False)'''
# 0.02085002581137858 1.0278921992243657
sample = stats.uniform.rvs(0, 1 / m.sqrt(2), size = 20)
hipotise = lambda x: stats.laplace.cdf(x,0.02085002581137858, 1.0278921992243657/ m.sqrt(2))
frame = []
#frame = pd.DataFrame(columns=['Границы'])
frame.append(grid_generator(len(sample)))
frame.append(n_i(frame[0]))
frame.append(p_i(frame[0], hipotise))
frame.append(len(sample) * np.array(frame[2]))
frame.append(np.array(frame[1]) - np.array(frame[3]))
frame.append((frame[1] - frame[3])**2 / (frame[3]))
p = pd.DataFrame(frame)
p.to_excel('lap.xlsx', index=False)