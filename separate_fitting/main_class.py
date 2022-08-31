import time
time_start = time.time()

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import pprint as pp
import matplotlib.ticker as ticker

# Functions Files
import class_pool as classes


mpl.rcParams["text.usetex"] = True

path = ['/home/jaesung/OneDrive/Code/WongCode/13TeV_atlas/atlasgraphs', '/home/jaesung/OneDrive/Code/WongCode/13TeV-Alice/HEPData-ins1840098-v1-csv', '/home/jaesung/OneDrive/Code/WongCode/13TeV/HEPData-ins1397173-v1-csv']
phi_13TeV_multi_atlas = []
dat_13TeV_multi_atlas = []
# 순서 : alice, alice, alice, ..., cms, cms, cms, ....., atlas, alice, cms
# 마지막 두개는 Yridge
phi_13TeV_ptdep = []
dat_13TeV_ptdep = []
err_13TeV_ptdep = []
fitting_error = []
for i in range(5):
    start = 10*i + 90
    end = 10*i + 100
    if start == 130:
        phi_13TeV_multi_atlas.append(np.loadtxt(path[0]+f'/13TeV_{start}~.csv',delimiter=',',usecols=[0], skiprows=3, max_rows=12))
        dat_13TeV_multi_atlas.append(np.loadtxt(path[0]+f'/13TeV_{start}~.csv',delimiter=',',usecols=[1], skiprows=3, max_rows=12))
    else:
        phi_13TeV_multi_atlas.append(np.loadtxt(path[0]+f'/13TeV_{start}~{end}.csv',delimiter=',',usecols=[0], skiprows=3, max_rows=12))
        dat_13TeV_multi_atlas.append(np.loadtxt(path[0]+f'/13TeV_{start}~{end}.csv',delimiter=',',usecols=[1], skiprows=3, max_rows=12))
    dat_13TeV_multi_atlas[i] -= min(dat_13TeV_multi_atlas[i])
for i in range(2):
    if i==0:
        '''ALICE data, 1~2, 2~3, 3~4'''
        for j in range(3):
            skip = 38*j+129
            phi_13TeV_ptdep.append(np.loadtxt(path[1]+'/1-NTRIGDN-DPHI.csv',delimiter=',',usecols=[0], skiprows=skip, max_rows=13))
            dat_13TeV_ptdep.append(np.loadtxt(path[1]+'/1-NTRIGDN-DPHI.csv',delimiter=',',usecols=[3], skiprows=skip, max_rows=13))
            err_sta1 = np.loadtxt(path[1]+'/1-NTRIGDN-DPHI.csv',delimiter=',',usecols=[4],skiprows=skip, max_rows=13)
            err_sta2 = np.loadtxt(path[1]+'/1-NTRIGDN-DPHI.csv',delimiter=',',usecols=[5],skiprows=skip, max_rows=13)
            err_sys1 = np.loadtxt(path[1]+'/1-NTRIGDN-DPHI.csv',delimiter=',',usecols=[6],skiprows=skip, max_rows=13)
            err_sys2 = np.loadtxt(path[1]+'/1-NTRIGDN-DPHI.csv',delimiter=',',usecols=[7],skiprows=skip, max_rows=13)
            err_13TeV_ptdep.append((err_sta1**2+err_sys1**2)**0.5)
            err_13TeV_ptdep.append((err_sta2**2+err_sys2**2)**0.5)
            fitting_error.append((err_sta2**2+err_sys2**2)**0.5)
    if i==1:
        '''CMS data, 0.1~1, 1~2, 2~3, 3~4'''
        for j in range(4):
            table = 2*j+25
            phi_13TeV_ptdep.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[0], skiprows=18, max_rows=13))
            dat_13TeV_ptdep.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[1], skiprows=18, max_rows=13))
            err_13TeV_ptdep.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[2], skiprows=18, max_rows=13))
            err_13TeV_ptdep.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[3], skiprows=18, max_rows=13))
            fitting_error.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[3], skiprows=18, max_rows=13))

'''ATLAS data'''
phi_13TeV_ptdep.append(np.loadtxt(path[0]+'/13TeV_90~.csv',delimiter=',',usecols=[0],skiprows=3,max_rows=12))
dat_13TeV_ptdep.append(np.loadtxt(path[0]+'/13TeV_90~.csv',delimiter=',',usecols=[1],skiprows=3,max_rows=12))

for i in range(len(dat_13TeV_ptdep)):
    dat_13TeV_ptdep[i] -= min(dat_13TeV_ptdep[i])

fitting_error.append(np.zeros(len(phi_13TeV_ptdep[-1]))+0.000703141335931843)
# atlas_error_indexing = len(err_13TeV_ptdep)
# atlaserror = np.zeros(len(phi_13TeV_ptdep[-1]))

'''Yridge data append'''
err_sta1=np.loadtxt(path[1]+'/Y^mathrm{ridge}.csv',delimiter=',',usecols=[4],skiprows=12,max_rows=7)
err_sta2=np.loadtxt(path[1]+'/Y^mathrm{ridge}.csv',delimiter=',',usecols=[5],skiprows=12,max_rows=7)
err_sys1=np.loadtxt(path[1]+'/Y^mathrm{ridge}.csv',delimiter=',',usecols=[6],skiprows=12,max_rows=7)
err_sys2=np.loadtxt(path[1]+'/Y^mathrm{ridge}.csv',delimiter=',',usecols=[7],skiprows=12,max_rows=7)
phi_13TeV_ptdep.append(np.loadtxt(path[1]+'/Y^mathrm{ridge}.csv',delimiter=',',usecols=[0],skiprows=12,max_rows=7))
dat_13TeV_ptdep.append(np.loadtxt(path[1]+'/Y^mathrm{ridge}.csv',delimiter=',',usecols=[3],skiprows=12,max_rows=7))
err_13TeV_ptdep.append((err_sta1**2+err_sys1**2)**0.5)
err_13TeV_ptdep.append((err_sta2**2+err_sys2**2)**0.5)
fitting_error.append((err_sta2**2+err_sys2**2)**0.5)
err_sta1=np.loadtxt(path[2]+'/Table33.csv',delimiter=',',usecols=[2],skiprows=14,max_rows=9)
err_sta2=np.loadtxt(path[2]+'/Table33.csv',delimiter=',',usecols=[3],skiprows=14,max_rows=9)
err_sys1=np.loadtxt(path[2]+'/Table33.csv',delimiter=',',usecols=[4],skiprows=14,max_rows=9)
err_sys2=np.loadtxt(path[2]+'/Table33.csv',delimiter=',',usecols=[5],skiprows=14,max_rows=9)
phi_13TeV_ptdep.append(np.loadtxt(path[2]+'/Table33.csv',delimiter=',',usecols=[0],skiprows=14,max_rows=9))
dat_13TeV_ptdep.append(np.loadtxt(path[2]+'/Table33.csv',delimiter=',',usecols=[1],skiprows=14,max_rows=9))
err_13TeV_ptdep.append((err_sta1**2+err_sys1**2)**0.5)
err_13TeV_ptdep.append((err_sta2**2+err_sys2**2)**0.5)
fitting_error.append((err_sta2**2+err_sys2**2)**0.5)
# fitting_error = err_13TeV_ptdep.insert(atlas_error_indexing, atlaserror)
# fitting_error.insert(atlas_error_indexing, atlaserror)


# Yridge pt range
ptloww = []
pthigh = []

ptloww_ali = np.loadtxt(path[1]+'/Y^mathrm{ridge}.csv',delimiter=',',usecols=[1],skiprows=12,max_rows=7)
pthigh_ali = np.loadtxt(path[1]+'/Y^mathrm{ridge}.csv',delimiter=',',usecols=[2],skiprows=12,max_rows=7)
ptloww_cms = np.loadtxt(path[2]+'/Table33.csv',delimiter=',',usecols=[6],skiprows=14,max_rows=9)
pthigh_cms = np.loadtxt(path[2]+'/Table33.csv',delimiter=',',usecols=[7],skiprows=14,max_rows=9)

ptloww.append(np.loadtxt(path[1]+'/Y^mathrm{ridge}.csv',delimiter=',',usecols=[1],skiprows=12,max_rows=7))
pthigh.append(np.loadtxt(path[1]+'/Y^mathrm{ridge}.csv',delimiter=',',usecols=[2],skiprows=12,max_rows=7))
ptloww.append(np.loadtxt(path[2]+'/Table33.csv',delimiter=',',usecols=[6],skiprows=14,max_rows=9))
pthigh.append(np.loadtxt(path[2]+'/Table33.csv',delimiter=',',usecols=[7],skiprows=14,max_rows=9))

'''Fitting with ATLAS Data'''
# ptf = [(1, 2), (2, 3), (3, 4), (1, 2), (2, 3), (3, 4), (0.5, 5)]
# etaf = [(1.6, 1.8), (1.6, 1.8), (1.6, 1.8), (2, 4), (2, 4), (2, 4), (2, 5)]
'''Fitting without ATLAS data'''
ptf = [(1, 2), (2, 3), (3, 4), (0.1, 1), (1, 2), (2, 3), (3, 4)]
etaf = [(1.6, 1.8), (1.6, 1.8), (1.6, 1.8), (2, 4), (2, 4), (2, 4), (2, 4)]
'''boundary conditions'''
boundary = ((0.2, 0.1, 0, 0, 0),(5, 1.3, 5, 10, 10))
# initial = (0.9, 0.65, 0.83, 0.5)
'''initial parameters'''
# initial = (0.82, 0.57, .36, .6)
# initial = (0.72, 0.3, 0.18, 1.6, 0.5)
initial = (1.11919, 0.98740754, 1.7766, .1051285459, 0.0)


# phi_13TeV_ptdep_fitting = phi_13TeV_ptdep[0:-2]     # fitting에 사용하는 데이터에서 Yridge를 제외하고 fitting 하려는 경우
# dat_13TeV_ptdep_fitting = dat_13TeV_ptdep[0:-2]     # fitting에 사용하는 데이터에서 Yridge를 제외하고 fitting 하려는 경우
'''[:] 는 deep copy를 위해 필요함'''
phi_13TeV_ptdep_fitting = phi_13TeV_ptdep[:]          # fitting에 사용하는 데이터에서 Yridge를 포함하려는 경우
dat_13TeV_ptdep_fitting = dat_13TeV_ptdep[:]          # fitting에 사용하는 데이터에서 Yridge를 포함하려는 경우
del phi_13TeV_ptdep_fitting[7]                        # delete ATLAS phi
del dat_13TeV_ptdep_fitting[7]                        # delete ATLAS data

'''ALICE 데이터 모음'''
total_phi_alice = phi_13TeV_ptdep_fitting[0:2]
total_phi_alice.append(phi_13TeV_ptdep_fitting[-2])
total_dat_alice = dat_13TeV_ptdep_fitting[0:2]
total_dat_alice.append(dat_13TeV_ptdep_fitting[-2])
'''CMS 데이터 모음'''
total_phi_cms = phi_13TeV_ptdep_fitting[3:6]
total_phi_cms.append(phi_13TeV_ptdep_fitting[-1])
total_dat_cms = dat_13TeV_ptdep_fitting[3:6]
total_dat_cms.append(dat_13TeV_ptdep_fitting[-1])

ptdep_alice = classes.Fitting_gpu(total_phi_alice, total_dat_alice, (ptloww, pthigh), None, ptf, etaf, boundary, initial, "pTdependence")
ptdep_alice_result, ptdep_alice_error = ptdep_alice.fitting(None)                  # error를 고려하지 않으려는 경우
print('ALICE results : ', ptdep_alice_result)
print('ALICE error : ', ptdep_alice_error)

ptdep_cms = classes.Fitting_gpu(total_phi_alice, total_dat_alice, (ptloww, pthigh), None, ptf, etaf, boundary, initial, "pTdependence")
ptdep_cms_result, ptdep_cms_error = ptdep_cms.fitting(None)                  # error를 고려하지 않으려는 경우
print(ptdep_cms_result)
print(ptdep_cms_error)



time_calculate = time.time()
print(f"calculate end : {time_calculate-time_start:.3f} sec")



Yridge_phi = phi_13TeV_ptdep[-2::]
Yridge_dat = dat_13TeV_ptdep[-2::]
phi_13TeV_ptdep = phi_13TeV_ptdep[0:-2]
dat_13TeV_ptdep = dat_13TeV_ptdep[0:-2]

fig1, axes1 = plt.subplots(nrows=1, ncols=5,figsize=(125,20))
#그래프 그리기
alice = classes.Drawing_Graphs((1.6, 1.8), *ptdep_alice_result, None, None)
cms = classes.Drawing_Graphs((2, 4), *ptdep_cms_result, None, None)
atlas = classes.Drawing_Graphs((2, 5), *ptdep_cms_result, None, None)       # 일단 CMS 결과로 대체
# print(dat_13TeV_ptdep)
# print(err_13TeV_ptdep)
for i in range(5):
    if i==0:
        ptf = (0.1, 1)
        cms_result = cms.result_plot("pTdependence", None, ptf, (min(phi_13TeV_ptdep[i+3]), max(phi_13TeV_ptdep[i+3])))
        axes1[i].plot(cms_result[0], cms_result[1], color = "black", linewidth=7, linestyle='-')
        axes1[i].errorbar(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3]-min(dat_13TeV_ptdep[i+3]), yerr=(abs(err_13TeV_ptdep[2*(i+3)+1]),err_13TeV_ptdep[2*(i+3)]), color="black", linestyle=' ', linewidth=7, capthick=3, capsize=15)
        axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3]-min(dat_13TeV_ptdep[i+3]), edgecolors="black", s=800, marker='o', facecolors='none', linewidths=7)
        axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3]-min(dat_13TeV_ptdep[i+3]), s=800, marker='+', facecolors='black', linewidths=7)
        # axes1[i].set_title(str(st)+r'$<p_{T,\,trig(assoc)}<$'+str(en), size = 70, pad=30)
        axes1[i].set_title(r'$0.1<p_{T,\,trig(assoc)}<1$', size = 70, pad=30)
    elif i==4:
        atlas_result = atlas.result_plot("pTdependence", None, (0.5, 5), (min(phi_13TeV_ptdep[-1]), max(phi_13TeV_ptdep[-1])))
        axes1[i].plot(atlas_result[0], atlas_result[1], color = "blue", linewidth=7, linestyle='-')
        axes1[i].scatter(phi_13TeV_ptdep[-1], dat_13TeV_ptdep[-1]-min(dat_13TeV_ptdep[-1]), edgecolors="blue", s=600, marker='o', linewidths=7)
        axes1[i].set_title(r'0.5$<p_{T,\,trig(assoc)}<$5', size = 70, pad=30)
    else:
        ptf = (i, i+1)
        alice_result = alice.result_plot("pTdependence", None, ptf, (min(phi_13TeV_ptdep[i-1]), max(phi_13TeV_ptdep[i-1])))
        cms_result = cms.result_plot("pTdependence", None, ptf, (min(phi_13TeV_ptdep[i+3]), max(phi_13TeV_ptdep[i+3])))
        axes1[i].plot(alice_result[0], alice_result[1], color = "red", linewidth=7, linestyle='-')
        axes1[i].plot(cms_result[0], cms_result[1], color = "black", linewidth=7, linestyle='-')
        '''alice plot'''
        # print(i)
        # print(dat_13TeV_ptdep[i-1])
        # print(err_13TeV_ptdep[2*i-1])
        axes1[i].errorbar(phi_13TeV_ptdep[i-1], dat_13TeV_ptdep[i-1]-min(dat_13TeV_ptdep[i-1]), yerr=(abs(err_13TeV_ptdep[2*i-1]),err_13TeV_ptdep[2*i-2]), color="red", linestyle=' ', linewidth=7, capthick=3, capsize=15)
        axes1[i].scatter(phi_13TeV_ptdep[i-1], dat_13TeV_ptdep[i-1]-min(dat_13TeV_ptdep[i-1]), edgecolors="red", s=800, marker='o', facecolors='none', linewidths=7)
        axes1[i].scatter(phi_13TeV_ptdep[i-1], dat_13TeV_ptdep[i-1]-min(dat_13TeV_ptdep[i-1]), s=800, marker='+', facecolors='red', linewidths=7)
        '''cms plot'''
        axes1[i].errorbar(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3]-min(dat_13TeV_ptdep[i+3]), yerr=(abs(err_13TeV_ptdep[2*(i+3)+1]),err_13TeV_ptdep[2*(i+3)]), color="black", linestyle=' ', linewidth=7, capthick=3, capsize=15)
        axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3]-min(dat_13TeV_ptdep[i+3]), edgecolors="black", s=800, marker='o', facecolors='none', linewidths=7)
        axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3]-min(dat_13TeV_ptdep[i+3]), s=800, marker='+', facecolors='black', linewidths=7)
        st = i
        en = i+1
        # axes1[i].set_title(str(st)+r'$<p_{T,\,trig(assoc)}<$'+str(en), size = 70, pad=30)
        axes1[i].set_title(str(st)+r'$<p_{T,\,trig(assoc)}<$'+str(en), size = 70, pad=30)
    axes1[i].set_xlabel(r'$\Delta\phi$', size=70)
    axes1[i].minorticks_on()
    axes1[i].tick_params(axis='both',which='major',direction='in',width=2,length=30,labelsize=45, top = 'true', right='true')
    axes1[i].tick_params(axis='both',which='minor',direction='in',width=2,length=15,labelsize=45, top = 'true', right='true')
    axes1[i].grid(color='silver',linestyle=':',linewidth=3)

axes1[0].set_ylabel(r'$\frac{1}{N_{trig}}\frac{dN^{pair}}{d\Delta\phi}-C_{ZYAM}$', size=70)
fig1.tight_layout(h_pad=-1)

fig1.savefig('./phiCorr_Test.png')
fig1.savefig('/home/jaesung/Dropbox/ohno/phiCorr_Test.png')

time_phicorr = time.time()
print(f"Graph, Phi correlation end : {time_phicorr-time_start:.3f} sec")

fig2, axis2 = plt.subplots(nrows=1, ncols=1,figsize=(40,20))
alice_Yridge = alice.Yridge((ptloww_ali, pthigh_ali), "Subtract")
# print(alice_Yridge)
# cms_Yridge = cms.Yridge((ptloww_cms, pthigh_cms), "Remove")
cms_Yridge = cms.Yridge((ptloww_cms, pthigh_cms), "Subtract")

axis2.scatter(alice_Yridge[0], alice_Yridge[1], edgecolor = "red", marker='s', facecolors='none', s=800, linewidth=5, zorder=1)
axis2.scatter(alice_Yridge[0], alice_Yridge[1], color = "red", marker='+', s=800, linewidth=5, zorder=1)
axis2.scatter(cms_Yridge[0], cms_Yridge[1], edgecolor = "black", marker='s', facecolors='none', s=800, linewidth=5, zorder=1)
axis2.scatter(cms_Yridge[0], cms_Yridge[1], color = "black", marker='+', s=800, linewidth=5, zorder=1)
# print(abs(err_13TeV_ptdep[15]),err_13TeV_ptdep[14])
# print(Yridge_dat)
axis2.errorbar(Yridge_phi[0], Yridge_dat[0], yerr=(abs(err_13TeV_ptdep[15]),err_13TeV_ptdep[14]), color="orange", linestyle=' ', linewidth=5, capthick=3, capsize=15, zorder=0)
axis2.errorbar(Yridge_phi[1], Yridge_dat[1], yerr=(abs(err_13TeV_ptdep[17]),err_13TeV_ptdep[16]), color="grey", linestyle=' ', linewidth=5, capthick=3, capsize=15, zorder=0)
axis2.scatter(Yridge_phi[0], Yridge_dat[0], edgecolors="orange", s=800, marker='o', facecolors='none', linewidths=5, zorder=0)
axis2.scatter(Yridge_phi[0], Yridge_dat[0], s=800, marker='+', facecolors='orange', linewidths=5, zorder=0)
axis2.scatter(Yridge_phi[1], Yridge_dat[1], edgecolors="grey", s=800, marker='o', facecolors='none', linewidths=5, zorder=0)
axis2.scatter(Yridge_phi[1], Yridge_dat[1], s=800, marker='+', facecolors='grey', linewidths=5, zorder=0)

axis2.set_xlabel(r'$p_{T,\,trig(assoc)}$', size=70)
axis2.set_ylabel(r'$Y^{ridge}$', size=70)
axis2.minorticks_on()
axis2.tick_params(axis='both',which='major',direction='in',width=2,length=30,labelsize=45, top = 'true', right='true')
axis2.tick_params(axis='both',which='minor',direction='in',width=2,length=15,labelsize=45, top = 'true', right='true')
axis2.grid(color='silver',linestyle=':',linewidth=3)
fig2.tight_layout()
fig2.savefig('./Yridge_Test.png')
fig2.savefig('/home/jaesung/Dropbox/ohno/Yridge_Test.png')

time_yridge = time.time()
print(f"Graph, Yridge end : {time_yridge-time_start:.3f} sec")

fig3 = plt.figure()
ax = plt.axes()
fig3.set_size_inches(35, 16.534, forward=True)


def FrNk_func(pt, xx, yy, zz):
    return xx*np.exp(-yy/pt-zz*pt)

AuAu_200GeV = [4, 0, 0]
PbPb_276TeV = [20.2, 1.395, 0.207]
pp_13TeV = [ptdep_alice_result[2], ptdep_alice_result[3], ptdep_alice_result[4]]
'''Hanul's pp 13TeV FrNk result plot [[pt(Average)], [FrNk]]'''
Hanul_FrNk = [[1.5, 2.5], [0.93, 1.37]]
ptf = np.arange(0.01,4,0.01)

plt.plot(ptf, FrNk_func(ptf, *AuAu_200GeV), color = 'red', linewidth=7, label=r'$AuAu, \, 200GeV$')
plt.plot(ptf, FrNk_func(ptf, *PbPb_276TeV), color = 'black', linewidth=7, label=r'$PbPb, \, 2.76TeV$')
plt.plot(ptf, FrNk_func(ptf, *pp_13TeV), color = 'blue', linewidth=7, label=r'$pp, \, 13TeV$')
plt.scatter(Hanul_FrNk[0], Hanul_FrNk[1], edgecolor = 'green', facecolors='none', s=900, marker='o', linewidths=5, zorder=2, label=r'$pp, \, 13TeV \, (reference)$')
plt.scatter(Hanul_FrNk[0], Hanul_FrNk[1], facecolors='green', s=900, marker='+', linewidths=5, zorder=2)

plt.xlabel(r'$p_T^{trig}$',size=70)
plt.ylabel(r'$f_{R} \langle N_k \rangle $',size=70)
plt.xlim(0,4)

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '${:g}$'.format(y)))
plt.tick_params(axis='both',which='major',direction='in',width=2,length=30,labelsize=45, top='true')
plt.tick_params(axis='both',which='minor',direction='in',width=2,length=15,labelsize=45, top='true')

plt.grid(color='silver',linestyle=':',linewidth=5, zorder=0)
plt.legend(fontsize=45, loc='upper left')

plt.tight_layout()

fig3.savefig('./FrNk_Test.png')
fig3.savefig('/home/jaesung/Dropbox/ohno/FrNk_Test.png')


time_end = time.time()
print(f"Total end : {time_end-time_start:.3f} sec")