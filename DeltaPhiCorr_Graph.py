import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import pprint as pp
import sys
import matplotlib.ticker as ticker

phi_13TeV_ptdep = []
dat_13TeV_ptdep = []
err_13TeV_ptdep = []
phi_07TeV_ptdep = []
dat_07TeV_ptdep = []
err_07TeV_ptdep = []
fitting_error = []

mpl.rcParams["text.usetex"] = True

path = ['./data/atldata/13TeV/', './data/alidata/', './data/cmsdata/', './data/atldata/2.76TeV/', './data/atldata/5.02TeV/', './data/atldata/13TeV_distribution/', './data/atldata/']

'''append 13TeV pt dependence data'''
for i in range(3):
    if i==0:
        '''ALICE data, 1~2, 2~3, 3~4'''
        for j in range(3):
            skip = 38*j+129
            phi_13TeV_ptdep.append(np.loadtxt(path[1]+'1-NTRIGDN-DPHI.csv',delimiter=',',usecols=[0], skiprows=skip, max_rows=13))
            dat_13TeV_ptdep.append(np.loadtxt(path[1]+'1-NTRIGDN-DPHI.csv',delimiter=',',usecols=[3], skiprows=skip, max_rows=13))
            err_sta1 = np.loadtxt(path[1]+'1-NTRIGDN-DPHI.csv',delimiter=',',usecols=[4],skiprows=skip, max_rows=13)
            err_sta2 = np.loadtxt(path[1]+'1-NTRIGDN-DPHI.csv',delimiter=',',usecols=[5],skiprows=skip, max_rows=13)
            err_sys1 = np.loadtxt(path[1]+'1-NTRIGDN-DPHI.csv',delimiter=',',usecols=[6],skiprows=skip, max_rows=13)
            err_sys2 = np.loadtxt(path[1]+'1-NTRIGDN-DPHI.csv',delimiter=',',usecols=[7],skiprows=skip, max_rows=13)
            err_13TeV_ptdep.append((err_sta1**2+err_sys1**2)**0.5)
            err_13TeV_ptdep.append((err_sta2**2+err_sys2**2)**0.5)
            fitting_error.append((err_sta2**2+err_sys2**2)**0.5)
    
    '''13TeV'''
    if i==1:
        '''CMS data, 0.1~1, 1~2, 2~3, 3~4'''
        for j in range(4):
            table = 2*j+25
            if j == 0:
                phi_13TeV_ptdep.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[0], skiprows=19, max_rows=11))
                dat_13TeV_ptdep.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[1], skiprows=19, max_rows=11))
                err_13TeV_ptdep.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[2], skiprows=19, max_rows=11))
                err_13TeV_ptdep.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[3], skiprows=19, max_rows=11))
                fitting_error.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[3], skiprows=19, max_rows=11))
            else:                
                phi_13TeV_ptdep.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[0], skiprows=18, max_rows=13))
                dat_13TeV_ptdep.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[1], skiprows=18, max_rows=13))
                err_13TeV_ptdep.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[2], skiprows=18, max_rows=13))
                err_13TeV_ptdep.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[3], skiprows=18, max_rows=13))
                fitting_error.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[3], skiprows=18, max_rows=13))

    '''7TeV'''
    if i==2:
        '''CMS data, 0.1~1, 1~2, 2~3, 3~4'''
        for j in range(4):
            table = 2*j+26
            if j == 0:
                phi_07TeV_ptdep.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[0], skiprows=19, max_rows=11))
                dat_07TeV_ptdep.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[1], skiprows=19, max_rows=11))
                err_07TeV_ptdep.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[2], skiprows=19, max_rows=11))
                err_07TeV_ptdep.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[3], skiprows=19, max_rows=11))
            else:                
                phi_07TeV_ptdep.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[0], skiprows=18, max_rows=13))
                dat_07TeV_ptdep.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[1], skiprows=18, max_rows=13))
                err_07TeV_ptdep.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[2], skiprows=18, max_rows=13))
                err_07TeV_ptdep.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[3], skiprows=18, max_rows=13))


phi_13TeV_ptdep.append(np.loadtxt(path[0]+'90~.csv',delimiter=',',usecols=[0],skiprows=3,max_rows=12))
dat_13TeV_ptdep.append(np.loadtxt(path[0]+'90~.csv',delimiter=',',usecols=[1],skiprows=3,max_rows=12))

def Yridge_append():
    err_sta1=np.loadtxt(path[1]+'Y^mathrm{ridge}.csv',delimiter=',',usecols=[4],skiprows=12,max_rows=7)
    err_sta2=np.loadtxt(path[1]+'Y^mathrm{ridge}.csv',delimiter=',',usecols=[5],skiprows=12,max_rows=7)
    err_sys1=np.loadtxt(path[1]+'Y^mathrm{ridge}.csv',delimiter=',',usecols=[6],skiprows=12,max_rows=7)
    err_sys2=np.loadtxt(path[1]+'Y^mathrm{ridge}.csv',delimiter=',',usecols=[7],skiprows=12,max_rows=7)
    phi_13TeV_ptdep.append(np.loadtxt(path[1]+'Y^mathrm{ridge}.csv',delimiter=',',usecols=[0],skiprows=12,max_rows=7))
    dat_13TeV_ptdep.append(np.loadtxt(path[1]+'Y^mathrm{ridge}.csv',delimiter=',',usecols=[3],skiprows=12,max_rows=7))
    err_13TeV_ptdep.append((err_sta1**2+err_sys1**2)**0.5)
    err_13TeV_ptdep.append((err_sta2**2+err_sys2**2)**0.5)
    fitting_error.append((err_sta2**2+err_sys2**2)**0.5)
    err_sta1=np.loadtxt(path[2]+'Table33.csv',delimiter=',',usecols=[2],skiprows=14,max_rows=9)
    err_sta2=np.loadtxt(path[2]+'Table33.csv',delimiter=',',usecols=[3],skiprows=14,max_rows=9)
    err_sys1=np.loadtxt(path[2]+'Table33.csv',delimiter=',',usecols=[4],skiprows=14,max_rows=9)
    err_sys2=np.loadtxt(path[2]+'Table33.csv',delimiter=',',usecols=[5],skiprows=14,max_rows=9)
    phi_13TeV_ptdep.append(np.loadtxt(path[2]+'Table33.csv',delimiter=',',usecols=[0],skiprows=14,max_rows=9))
    dat_13TeV_ptdep.append(np.loadtxt(path[2]+'Table33.csv',delimiter=',',usecols=[1],skiprows=14,max_rows=9))
    err_13TeV_ptdep.append((err_sta1**2+err_sys1**2)**0.5)
    err_13TeV_ptdep.append((err_sta2**2+err_sys2**2)**0.5)
    fitting_error.append((err_sta2**2+err_sys2**2)**0.5)
    err_sta1=np.loadtxt(path[2]+'Table34.csv',delimiter=',',usecols=[2],skiprows=14,max_rows=9)
    err_sta2=np.loadtxt(path[2]+'Table34.csv',delimiter=',',usecols=[3],skiprows=14,max_rows=9)
    err_sys1=np.loadtxt(path[2]+'Table34.csv',delimiter=',',usecols=[4],skiprows=14,max_rows=9)
    err_sys2=np.loadtxt(path[2]+'Table34.csv',delimiter=',',usecols=[5],skiprows=14,max_rows=9)
    phi_07TeV_ptdep.append(np.loadtxt(path[2]+'Table34.csv',delimiter=',',usecols=[0],skiprows=14,max_rows=9))
    dat_07TeV_ptdep.append(np.loadtxt(path[2]+'Table34.csv',delimiter=',',usecols=[1],skiprows=14,max_rows=9))
    err_07TeV_ptdep.append((err_sta1**2+err_sys1**2)**0.5)
    err_07TeV_ptdep.append((err_sta2**2+err_sys2**2)**0.5)
Yridge_append()

Yridge_phi07 = phi_07TeV_ptdep[-1]
Yridge_dat07 = dat_07TeV_ptdep[-1]
Yridge_phi = phi_13TeV_ptdep[-2::]
Yridge_dat = dat_13TeV_ptdep[-2::]
phi_13TeV_ptdep = phi_13TeV_ptdep[0:-2]
dat_13TeV_ptdep = dat_13TeV_ptdep[0:-2]

fig1, axes1 = plt.subplots(nrows=1, ncols=5,figsize=(125,20))

for i in range(5):
    if i==0:
        ptf = (0.1, 1)
        '''cms plot 13TeV'''
        axes1[i].errorbar(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3], yerr=(abs(err_13TeV_ptdep[2*(i+3)+1]),err_13TeV_ptdep[2*(i+3)]), color="black", linestyle=' ', linewidth=7, capthick=3, capsize=15)
        axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3], edgecolors="black", s=800, marker='o', facecolors='none', linewidths=7)
        axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3], s=800, marker='+', facecolors='black', linewidths=7)
        '''cms plot 7TeV'''
        axes1[i].errorbar(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i], yerr=(abs(err_07TeV_ptdep[2*i+1]),err_07TeV_ptdep[2*i]), color="grey", linestyle=' ', linewidth=7, capthick=3, capsize=15)
        axes1[i].scatter(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i], edgecolors="grey", s=800, marker='o', facecolors='none', linewidths=7)
        axes1[i].scatter(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i], s=800, marker='+', facecolors='grey', linewidths=7)
        axes1[i].set_title(r'$0.1<p_{T, \, \mathrm{trig(assoc)}}<1$', size = 70, pad=30)
    elif i==4:
        axes1[i].scatter(phi_13TeV_ptdep[-1], dat_13TeV_ptdep[-1]-min(dat_13TeV_ptdep[-1]), facecolors='blue', edgecolors="blue", s=600, marker='o', linewidths=7)
        axes1[i].set_title(r'0.5$<p_{T, \, \mathrm{trig(assoc)}}<$5', size = 70, pad=30)
    else:
        ptf = (i, i+1)
        '''alice plot'''
        axes1[i].errorbar(phi_13TeV_ptdep[i-1], dat_13TeV_ptdep[i-1], yerr=(abs(err_13TeV_ptdep[2*i-1]),err_13TeV_ptdep[2*i-2]), color="red", linestyle=' ', linewidth=7, capthick=3, capsize=15)
        axes1[i].scatter(phi_13TeV_ptdep[i-1], dat_13TeV_ptdep[i-1], edgecolors="red", s=800, marker='o', facecolors='none', linewidths=7)
        axes1[i].scatter(phi_13TeV_ptdep[i-1], dat_13TeV_ptdep[i-1], s=800, marker='+', facecolors='red', linewidths=7)
        '''cms plot 13TeV'''
        axes1[i].errorbar(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3], yerr=(abs(err_13TeV_ptdep[2*(i+3)+1]),err_13TeV_ptdep[2*(i+3)]), color="black", linestyle=' ', linewidth=7, capthick=3, capsize=15)
        axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3], edgecolors="black", s=800, marker='o', facecolors='none', linewidths=7)
        axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3], s=800, marker='+', facecolors='black', linewidths=7)
        '''cms plot 7TeV'''
        axes1[i].errorbar(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i], yerr=(abs(err_07TeV_ptdep[2*i+1]),err_07TeV_ptdep[2*i]), color="grey", linestyle=' ', linewidth=7, capthick=3, capsize=15)
        axes1[i].scatter(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i], edgecolors="grey", s=800, marker='o', facecolors='none', linewidths=7)
        axes1[i].scatter(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i], s=800, marker='+', facecolors='grey', linewidths=7)
        st = i
        en = i+1
        # axes1[i].set_title(str(st)+r'$<p_{T, \, \mathrm{trig(assoc)}}<$'+str(en), size = 70, pad=30)
        axes1[i].set_title(str(st)+r'$<p_{T, \, \mathrm{trig(assoc)}}<$'+str(en), size = 70, pad=30)

    axes1[i].set_xlabel(r'$\Delta\phi$', size=70)
    axes1[i].minorticks_on()
    axes1[i].tick_params(axis='both',which='major',direction='in',width=2,length=30,labelsize=45, top = 'true', right='true')
    axes1[i].tick_params(axis='both',which='minor',direction='in',width=2,length=15,labelsize=45, top = 'true', right='true')
    axes1[i].grid(color='silver',linestyle=':',linewidth=3)

axes1[0].set_ylabel(r'$\frac{1}{N_{\mathrm{trig}}}\frac{dN^{\mathrm{pair}}}{d\Delta\phi}-C_{\mathrm{ZYAM}}$', size=70)
fig1.tight_layout(h_pad=-1)

fig1.savefig('./phiCorr_Plot.png')




fig2, axis2 = plt.subplots(nrows=1, ncols=1,figsize=(40,20))

axis2.errorbar(Yridge_phi[0], Yridge_dat[0], yerr=(abs(err_13TeV_ptdep[15]),err_13TeV_ptdep[14]), color="red", linestyle=' ', linewidth=5, capthick=3, capsize=15, zorder=0)
axis2.errorbar(Yridge_phi[1], Yridge_dat[1], yerr=(abs(err_13TeV_ptdep[17]),err_13TeV_ptdep[16]), color="black", linestyle=' ', linewidth=5, capthick=3, capsize=15, zorder=0)
axis2.scatter(Yridge_phi[0], Yridge_dat[0], edgecolors="red", s=800, marker='o', facecolors='none', linewidths=5, zorder=0)
axis2.scatter(Yridge_phi[0], Yridge_dat[0], s=800, marker='+', facecolors='red', linewidths=5, zorder=0)
axis2.scatter(Yridge_phi[1], Yridge_dat[1], edgecolors="black", s=800, marker='o', facecolors='none', linewidths=5, zorder=0)
axis2.scatter(Yridge_phi[1], Yridge_dat[1], s=800, marker='+', facecolors='black', linewidths=5, zorder=0)
axis2.errorbar(Yridge_phi07, Yridge_dat07, yerr=(abs(err_07TeV_ptdep[-2]),err_07TeV_ptdep[-1]), color="grey", linestyle=' ', linewidth=5, capthick=3, capsize=15, zorder=0)
axis2.scatter(Yridge_phi07, Yridge_dat07, edgecolors="grey", s=800, marker='o', facecolors='none', linewidths=5, zorder=0)
axis2.scatter(Yridge_phi07, Yridge_dat07, s=800, marker='+', facecolors='grey', linewidths=5, zorder=0)

axis2.set_xlabel(r'$p_{T, \, \mathrm{trig(assoc)}} \,\, \mathrm{(GeV/c)}$', size=70)
axis2.set_ylabel(r'$Y^{\mathrm{ridge}} \,\, \mathrm{(GeV/c)}$', size=70)
axis2.minorticks_on()
axis2.tick_params(axis='both',which='major',direction='in',width=2,length=30,labelsize=45, top = 'true', right='true')
axis2.tick_params(axis='both',which='minor',direction='in',width=2,length=15,labelsize=45, top = 'true', right='true')
axis2.grid(color='silver',linestyle=':',linewidth=3)
fig2.tight_layout()
fig2.savefig('./Yridge_Plot.png')