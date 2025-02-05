import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import pprint as pp
import sys
import matplotlib.ticker as ticker
import pandas as pd

Table = pd.read_csv('./Results/EarlyStopping.csv', delimiter=',')


fig = plt.figure()
ax = plt.axes()
fig.set_size_inches(35, 16.534, forward=True)

plt.plot(Table['Number'], Table['ALICE+CMS Error'], color = 'red', linestyle='-', linewidth=7, label=r'ALICE + CMS Error')
plt.plot(Table['Number'], Table['ATLAS Error'], color = 'black', linestyle='-', linewidth=7, label=r'ATLAS Error')

plt.xlabel(r'Number of Loop',size=50)
plt.ylabel(r'Error(%)',size=50)

plt.xlim([0, 100])

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '${:g}$'.format(y)))
plt.tick_params(axis='both',which='major',direction='in',width=2,length=30,labelsize=45, top='true')
plt.tick_params(axis='both',which='minor',direction='in',width=2,length=15,labelsize=45, top='true')

plt.grid(color='silver',linestyle=':',linewidth=5, zorder=0)
plt.legend(fontsize=45, loc='upper left')

plt.tight_layout()


fig.savefig('./Results/Error_Results.png')
