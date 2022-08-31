import time
time_start = time.time()

import numpy as np
import cupy as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from scipy.optimize import curve_fit
import csv
import scipy
from multiprocessing import Pool
import matplotlib.ticker as ticker

mpl.rcParams["text.usetex"] = True

# Functions Files
import Function_cpu as cpu
import Function_gpu as gpu

def multiplicity_func(xx, BB, N):
    return xx*np.exp(-BB/N)

fig3 = plt.figure()
ax = plt.axes()
fig3.set_size_inches(35, 16.534, forward=True)
'''Hanul's pp 13TeV FrNk result plot [[pt(Average)], [FrNk]]'''
Hanul_FrNk = [[1.5, 2.5], [0.93, 1.37]]
Hanul_FrNk_error = [[0.5, 0.5], [0.5, 0.5]]
multiplicity = np.arange(0.01,140,0.01)
y = np.zeros(len(multiplicity)) + 6.709677786197748


plt.plot(multiplicity, multiplicity_func(1.04208347e+02, 2.57620706e+02, multiplicity), color = 'blue', linewidth=7, label=r'$pp, \, 13TeV$')
plt.plot(multiplicity, y, color = 'red', linewidth=7)


plt.xlabel(r'$N_{ch}^{rec}$',size=70)
plt.ylabel(r'$\langle N_k \rangle$',size=70)
plt.xlim(0,140)

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '${:g}$'.format(y)))
plt.tick_params(axis='both',which='major',direction='in',width=2,length=30,labelsize=45, top='true')
plt.tick_params(axis='both',which='minor',direction='in',width=2,length=15,labelsize=45, top='true')

plt.grid(color='silver',linestyle=':',linewidth=5, zorder=0)
# plt.legend(fontsize=45, loc='upper left')

plt.tight_layout()

fig3.savefig('./multiplicity.png')