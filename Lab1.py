from scipy.stats import norminvgauss, laplace, poisson, cauchy, uniform
import numpy as np
import matplotlib.pyplot as plt
import math as m

size = [10,50,1000]

LINE_TYPE = 'k--'

#density - плотность
def Gauss():
    for s in size:
        den = norminvgauss(1, 0)
        hist = norminvgauss.rvs(1, 0, size=s)
        fig, ax = plt.subplots(1, 1)
        ax.hist(hist, density=True, alpha=0.6)
        x = np.linspace(den.ppf(0.01), den.ppf(0.99), 100)
        ax.plot(x, den.pdf(x), LINE_TYPE, lw=1.5)
        ax.set_xlabel("NORMAL")
        ax.set_ylabel("DENSITY")
        ax.set_title("SIZE: " + str(s))
        plt.grid()
        plt.show()


def Cauchy():
    for s in size:
        density = cauchy()
        histogram = cauchy.rvs(size=s)
        fig, ax = plt.subplots(1, 1)
        ax.hist(histogram, density=True, alpha=0.6)
        x = np.linspace(density.ppf(0.01), density.ppf(0.99), 100)
        ax.plot(x, density.pdf(x), LINE_TYPE, lw=1.5)
        ax.set_xlabel("CAUCHY")
        ax.set_ylabel("DENSITY")
        ax.set_title("SIZE: " + str(s))
        plt.grid()
        plt.show()

def Laplace():
    for s in size:
        den = laplace(scale=1 / m.sqrt(2), loc=0)
        hist = laplace.rvs(size=s, scale=1 / m.sqrt(2), loc=0)
        fig, ax = plt.subplots(1, 1)
        ax.hist(hist, density=True, alpha=0.6)
        x = np.linspace(den.ppf(0.01), den.ppf(0.99), 100)
        ax.plot(x, den.pdf(x), LINE_TYPE, lw=1.5)
        ax.set_xlabel("LAPLACE")
        ax.set_ylabel("DENSITY")
        ax.set_title("SIZE: " + str(s))
        plt.grid()
        plt.show()

def Poisson():
    for s in size:
        den = poisson(10)
        hist = poisson.rvs(10, size=s)
        fig, ax = plt.subplots(1, 1)
        ax.hist(hist, density=True, alpha=0.6)
        x = np.arange(poisson.ppf(0.01, 10), poisson.ppf(0.99, 10))
        ax.plot(x, den.pmf(x), LINE_TYPE, lw=1.5)
        ax.set_xlabel("POISSON")
        ax.set_ylabel("DENSITY")
        ax.set_title("SIZE: " + str(s))
        plt.grid()
        plt.show()


def Uniform():
    for s in size:
        density = uniform(loc=-m.sqrt(3), scale=2 * m.sqrt(3))
        histogram = uniform.rvs(size=s, loc=-m.sqrt(3), scale=2 * m.sqrt(3))
        fig, ax = plt.subplots(1, 1)
        ax.hist(histogram, density=True, alpha=0.6)
        x = np.linspace(density.ppf(0.01), density.ppf(0.99), 100)
        ax.plot(x, density.pdf(x), LINE_TYPE, lw=1.5)
        ax.set_xlabel("UNIFORM")
        ax.set_ylabel("DENSITY")
        ax.set_title("SIZE: " + str(s))
        plt.grid()
        plt.show()
Gauss()
Cauchy()
Uniform()
Laplace()
Poisson()
Uniform()
