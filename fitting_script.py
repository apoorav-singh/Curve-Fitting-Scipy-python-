# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 17:49:41 2023

@author: Apoorav
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import *
import pandas as pd
from IPython.display import display, Math

# Importing Data
df_1 = pd.read_csv('data.csv', skiprows = 0)
# print(df_1)

# Figure
fig, ax = plt.subplots(3)

# Figure padding
# fig.tight_layout(pad=2.0)

for i in range(ax.size):
    ax[i].set_box_aspect(1/6)

fig.subplots_adjust(hspace=0.8, wspace=0.1)


X1 = np.array(df_1["Frequency [Hz]"])
Y1 = np.array(df_1["Magnitude [W]"])


k = 0
l = -1

X = X1[k:l]
Y = 10**((Y1[k:l])/10.) # dBm to mW


def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma], maxfev = 1000000)
    return popt

def Voigt(x, H, ampG1, cenG1, sigmaG1, ampL1, cenL1, widL1):
    return H + (ampG1*(1/(sigmaG1*(np.sqrt(2*np.pi))))*(np.exp(-((x-cenG1)**2)/((2*sigmaG1)**2)))) +\
              ((ampL1*widL1**2/((x-cenL1)**2+widL1**2)) )

def Voigt_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(Voigt, x, y, p0=[min(y), max(y), mean, sigma, max(y), mean, sigma], maxfev = 1000000)
    return popt

def lorentzian(x, H, ampL1, cenL1, widL1):
    return H + ((ampL1*widL1**2/((x-cenL1)**2+widL1**2)))

def lorentzian_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(lorentzian, x, y, p0=[min(y), max(y), mean, sigma], maxfev = 1000000)
    return popt

H, A, x0, sigma = gauss_fit(X, Y)
FWHM = 2.35482 * sigma
HV, AGV, CGV, sigmaGV, ALV, CLV, widthLV = Voigt_fit(X, Y)
HL, AL, CL, sigmaL = lorentzian_fit(X, Y)


def latex_sym(f):
    float_str = "{0:.4g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    else:
        return float_str

prec_X = np.arange(min(X), max(X), 200)    
Z = gauss(prec_X, H, A, x0, sigma) # Gaussian Values
V = Voigt(prec_X, HV, AGV, CGV, sigmaGV, ALV, CLV, widthLV)
L = lorentzian(prec_X, HL, AL, CL, sigmaL)
    

print("------------ Gaussian Fit Parameters ------------\n")
print('The offset of the gaussian baseline is (mW):', H) 
print('The center of the gaussian fit is (Hz)', x0)
print('The sigma of the gaussian fit is (Hz)', sigma)
print('The intensity of the gaussian fit is (mW)', (H + A))
print('The FWHM of the gaussian fit is (Hz)', FWHM)
print("\n")
print("------------ Voigt Fit Parameters ------------\n")
print('The offset of the Voigt baseline is (mW):', HV) 
print('The center of the gaussian fit is (Hz)', CGV)
print('The sigma of the gaussian fit is (Hz)', 2.35482 * sigmaGV)
print('The FWHM of the gaussian fit is (Hz)', sigmaGV)
print('The center of the Lorentzian fit is (Hz)', CLV)
print('The width of the Lorentzian fit is (Hz)', widthLV)
print('The FWHM of the Lorentzian fit is (Hz)', 2*widthLV)
print('The intensity of the Voigt fit is (mW)', max(V))
print("\n")
print("------------ Lorentzian Fit Parameters ------------\n")
print('The offset of the Lornetzian baseline is (mW):', HL) 
print('The center of the Lorentzian fit is (Hz)', CL)
print('The width of the Lorentzian fit is (Hz)', sigmaL)
print('The intensity of the Lorentzian fit is (mW)', HL + AL)
print('The FWHM of the Loretzian fit is (Hz)', 2*sigmaL)



# ax[0].set_yscale("log")
# ax[1].set_yscale("log")
# ax[2].set_yscale("log")


ax[0].plot(prec_X, Z, "b", label="Gaussian Fit")
ax[0].plot(X,Y, "ro", markersize=2, label="RAW")
ax[0].set_xlabel(r"X")
ax[0].grid("major")
ax[0].grid("minor")
ax[0].set_ylabel("Y")
# ax1.set_xlim([3.19e8, 3.22e8])
ax[0].set_title("Gaussian Fitting")
ax[0].legend()


ax[1].plot(prec_X, V, "b", label="Voigt Fit")
ax[1].plot(X,Y, "ro", markersize=2, label="RAW")
ax[1].set_xlabel(r"X")
ax[1].grid("major")
ax[1].grid("minor")
ax[1].set_ylabel("Y")
# ax[1].set_xlim([3.19e8, 3.22e8])
ax[1].set_title("Voigt Fitting")
ax[1].legend()

ax[2].plot(prec_X, L, "b", label="Lorentzian Fit")
ax[2].plot(X,Y, "ro", markersize=2, label="RAW")
ax[2].set_xlabel(r"X")
ax[2].grid("major")
ax[2].grid("minor")
ax[2].set_ylabel("Y (mW)")
# ax[2].set_ylim([0, 5e-7])
ax[2].set_title("Lorentzian Fitting")
ax[2].legend()

# plt.savefig('fit.svg')
plt.show()