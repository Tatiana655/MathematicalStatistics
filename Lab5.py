import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import math as m
from matplotlib import pyplot as plt
from math import pi, cos, sin

def f(x,y, a, b):
    1/(2*pi*a*b) * m.exp(-x ** 2 / (2* a ** 2) - y ** 2 / (2 * b ** 2))

def Pirson_r(sel):
    return stats.pearsonr(np.transpose(np.transpose(sel)[0]), np.transpose(sel)[1])[0]

def r_q(sel):
    n1=n2=n3=n4=0
    med_x = 0#np.median(sel[0])
    med_y = 0#np.median(sel[1])
    for s in sel:
        if s[0]>med_x and s[1]>med_y: #1 четверть
            n1+=1
        if s[0]<med_x and s[1]>med_y: #2 четверть
            n2+=1
        if s[0]<med_x and s[1]<med_y: #3 четверть
            n3+=1
        if s[0]>med_x and s[1]<med_y: #4 четверть
            n4+=1
    return ((n1+n3) - (n2+n4))/len(sel)

def r_s(sel):
    u_=v_= (len(sel) + 1) / 2
    np.transpose(sel)
    return stats.spearmanr(sel)[0]

def generate_selections(correlation, mean_x, mean_y, std_x, std_y, size):
    return stats.multivariate_normal(mean=[mean_x, mean_y], cov=[[std_x, correlation], [correlation, std_y]]).rvs(size=size)

Ro = 0.9 #0 0.5 , 0.9
x_ = 0
y_ = 0
sig_x = 1
sig_y = 1
size = 100

strin = ['E(z) = ', 'E(z^2) = ', 'D(z) = ']
str_r = [' r ', ' r_s ', ' r_q ']

r_s_all = []
r_q_all = []
r_all = []

for i in range(1000):
    #selection_1 = generate_selections(0.9, 0, 0, 1, 1, size)
    #selection_2 = generate_selections(-0.9, 0, 0, 10, 10, size)
    #sel = 0.9 * selection_1 + 0.1 * selection_2
    sel = stats.multivariate_normal(mean=[x_, y_], cov=[[sig_x, Ro], [Ro, sig_y]]).rvs(size=size)#генерация выборки
    r_all.append(Pirson_r(sel))
    r_s_all.append(r_s(sel))
    r_q_all.append(r_q(sel))

print(str_r[0],str[0],round(np.mean(r_all),4))
print(str_r[1],str[0],round(np.mean(r_s_all),4))
print(str_r[2],str[0],round(np.mean(r_q_all),4))


print(str_r[0],str[1],round(np.mean([r_all[i] ** 2 for i in range(len(r_all))]),4))
print(str_r[1],str[1],round(np.mean([r_s_all[i] ** 2 for i in range(len(r_s_all))]),4))
print(str_r[2],str[1],round(np.mean([r_q_all[i] ** 2 for i in range(len(r_q_all))]),4))

print(str_r[0],str[2],round(np.var(r_all),4))
print(str_r[1],str[2],round(np.var(r_s_all),4))
print(str_r[2],str[2],round(np.var(r_q_all),4))

#print(r_s(sel))
#print(Pirson(sel,x_,y_))
#среднее
#np.mean(dist)

u=x_       #x-position of the center
v=y_      #y-position of the center
a=sig_x   #radius on the x-axis
b=sig_y     #radius on the y-axis
if sig_x ** 2 - sig_y ** 2 == 0:
    t_rot = pi/4;
else:
    t_rot=0.5 * m.atan(2*Ro*sig_y*sig_x/(sig_x ** 2 - sig_y ** 2))#0#pi/4 #rotation angle


a = sig_x **2 * m.cos(t_rot) ** 2 + Ro * sig_x * sig_y * 1 + sig_y **2 * m.sin(t_rot) ** 2
b = sig_x **2 * m.sin(t_rot) ** 2 - Ro * sig_x * sig_y * 1 + sig_y **2 * m.cos(t_rot) ** 2

sel = stats.multivariate_normal(mean=[x_, y_], cov=[[sig_x, Ro], [Ro, sig_y]]).rvs(size=size)#генерация выборки

a = a ** 0.5
b = b ** 0.5
print("var = ", np.var(sel))
k = 3  #2 * np.var(sel.transpose()[0]) ** 0.5 * 2 ** 0.5
#k = a ** 2 / np.var([sel.transpose()[0][i] for i in range(len(sel))]) + a ** 2 / (np.var([sel.transpose()[1][i] for i in range(len(sel))]))
#k = 1/**2/2 +1/b**2/2

a = a*k
b = b*k

t = np.linspace(0, 2*pi, 100)
Ell = np.array([a*np.cos(t) , b*np.sin(t)])
     #u,v removed to keep the same center location
R_rot = np.array([[cos(t_rot) , -sin(t_rot)],[sin(t_rot) , cos(t_rot)]])
     #2-D rotation matrix

Ell_rot = np.zeros((2,Ell.shape[1]))
for i in range(Ell.shape[1]):
    Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])

#plt.plot( u+Ell[0,:] , v+Ell[1,:] )     #initial ellipse
plt.plot( u+Ell_rot[0,:] , v+Ell_rot[1,:],'darkorange' )    #rotated ellipse
plt.grid(color='lightgray',linestyle='--')

plt.plot(sel.transpose()[0],sel.transpose()[1], 'o', color = 'indigo')
plt.title ("Ro = "+ str(Ro)+ "; n = "+ str(size))
#plt.savefig('Ellip' + str(size) + '-' + str(Ro) + '.png')
plt.show()