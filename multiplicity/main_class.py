import time
time_start = time.time()

'''
    여기에서는 전체 multiplicity 데이터를 이용해 fitting하는 작업을 시행한다.
    따라서 atlas, cms를 따로따로 fitting할 예정이며(나중에는 같이 fitting하는 것도 고려중),
    이에 따라 따로따로 list를 만들어 append하고 fitting을 진행한다.
    + cms데이터는 너무 fluctuation이 심해서 atlas만 진행할 것으로 보임.
'''

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

'''Multiplicity에 대한 associated yield
   순서 : ALTAS, CMS                   '''
Multi_N = []
Multi_Y = []
Multi_N.append(np.loadtxt(path[0]+'/Multiplicity_yield.csv',delimiter=',',usecols=[0], skiprows=1))
Multi_Y.append(np.loadtxt(path[0]+'/Multiplicity_yield.csv',delimiter=',',usecols=[1], skiprows=1))
Multi_N.append(np.loadtxt(path[2]+'/Table35.csv',delimiter=',',usecols=[0],skiprows=14))
Multi_Y.append(np.loadtxt(path[2]+'/Table35.csv',delimiter=',',usecols=[1],skiprows=14))

phi_13TeV_multi_atlas = []
phi_13TeV_multi_atlas_fitting = []
dat_13TeV_multi_atlas = []
dat_13TeV_multi_atlas_fitting = []
# min_13TeV_multi_atlas_index = []
multiplicity_atlas = []
# 순서 : alice, alice, alice, ..., cms, cms, cms, ....., atlas, alice, cms
# 마지막 두개는 Yridge
phi_13TeV_ptdep = []
dat_13TeV_ptdep = []
err_13TeV_ptdep = []
fitting_error = []
for i in range(9):
    start = 10*i + 50
    end = 10*i + 60
    if start == 130:
        phi_13TeV_multi_atlas.append(np.loadtxt(path[0]+f'/13TeV_{start}~.csv',delimiter=',',usecols=[0], skiprows=3, max_rows=12))
        dat_13TeV_multi_atlas.append(np.loadtxt(path[0]+f'/13TeV_{start}~.csv',delimiter=',',usecols=[1], skiprows=3, max_rows=12))
        '''multiplicity가 130이상이기 때문에 평균값보다 더 커서 변경할 수도 있어서 이렇게 두었음'''
        multiplicity_atlas.append((start+140)/2)
    else:
        multiplicity_atlas.append((start+end)/2)
        phi_13TeV_multi_atlas.append(np.loadtxt(path[0]+f'/13TeV_{start}~{end}.csv',delimiter=',',usecols=[0], skiprows=3, max_rows=12))
        dat_13TeV_multi_atlas.append(np.loadtxt(path[0]+f'/13TeV_{start}~{end}.csv',delimiter=',',usecols=[1], skiprows=3, max_rows=12))
    dat_13TeV_multi_atlas[i] -= min(dat_13TeV_multi_atlas[i])
    # print(phi_13TeV_multi_atlas[i])
    # print(dat_13TeV_multi_atlas[i])
    check1 = int(np.where(dat_13TeV_multi_atlas[i]==0)[0])
    check2 = int(len(dat_13TeV_multi_atlas[i])-check1)-1
    # print(check1, check2)
    if check1<check2:
        phi_13TeV_multi_atlas_fitting.append(phi_13TeV_multi_atlas[i][check1:check2+1])
        dat_13TeV_multi_atlas_fitting.append(dat_13TeV_multi_atlas[i][check1:check2+1])
    elif check1>check2:
        phi_13TeV_multi_atlas_fitting.append(phi_13TeV_multi_atlas[i][check2:check1+1])
        dat_13TeV_multi_atlas_fitting.append(dat_13TeV_multi_atlas[i][check2:check1+1])

'''CMS data 가져오기'''
phi_13TeV_multi_cms = []
phi_13TeV_multi_cms_fitting = []
dat_13TeV_multi_cms = []
dat_13TeV_multi_cms_fitting = []
err_13TeV_multi_cms = []
'''multiplicity 간격이 등간격이 되도록 마지막을 조정함. 추후에 조정할 수도 있음'''
multiplicity_cms = [57.5, 57.5, 57.5, 57.5, 92.5, 92.5, 92.5, 92.5, 127.5, 127.5, 127.5, 127.5]
for i in range(12):
    '''80<N<105, 3<pT<4 의 데이터가 많이 이상하여 if문으로 빼서 따로 처리해야 할 수도'''
    table = 2*i+9
    phi_13TeV_multi_cms.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[0], skiprows=18, max_rows=13))
    dat_13TeV_multi_cms.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[1], skiprows=18, max_rows=13))
    err_13TeV_multi_cms.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[2], skiprows=18, max_rows=13))
    err_13TeV_multi_cms.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[3], skiprows=18, max_rows=13))
    dat_13TeV_multi_cms[i] -= min(dat_13TeV_multi_cms[i])
    check1 = int(np.where(dat_13TeV_multi_cms[i]==0)[0][0])
    check2 = int(len(dat_13TeV_multi_cms[i])-check1)-1
    if check1<check2:
        phi_13TeV_multi_cms_fitting.append(phi_13TeV_multi_cms[i][check1:check2+1])
        dat_13TeV_multi_cms_fitting.append(dat_13TeV_multi_cms[i][check1:check2+1])
    elif check1>check2:
        phi_13TeV_multi_cms_fitting.append(phi_13TeV_multi_cms[i][check2:check1+1])
        dat_13TeV_multi_cms_fitting.append(dat_13TeV_multi_cms[i][check2:check1+1])

'''Yridge data append'''
'''이 파일에서는 일단 delta phi correlation만 계산할 것이다.'''
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

'''Fitting ATLAS Data'''
ptf_atl = [(0.5, 5), (0.5, 5), (0.5, 5), (0.5, 5), (0.5, 5), (0.5, 5), (0.5, 5), (0.5, 5), (0.5, 5)]
etaf_atl = [(2, 5), (2, 5), (2, 5), (2, 5), (2, 5), (2, 5), (2, 5), (2, 5), (2, 5)]
'''Fitting CMS Data'''
ptf_cms = [(0.1, 1), (1, 2), (2, 3), (3, 4), (0.1, 1), (1, 2), (2, 3), (3, 4), (0.1, 1), (1, 2), (2, 3), (3, 4)]
etaf_cms = [(2, 4), (2, 4), (2, 4), (2, 4), (2, 4), (2, 4), (2, 4), (2, 4), (2, 4), (2, 4), (2, 4), (2, 4)]
'''boundary conditions'''
# boundary = ((1.0, 0.8, 0, 0, 0, 0, 50),(1.5, 1.3, 100, 10, 10, 20, 300))
boundary = ((0, 0, 50),(500, 30, 400))      # fix parameters : kick, Tem, yy, zz
# initial = (0.9, 0.65, 0.83, 0.5)
'''initial parameters'''
# initial = (0.82, 0.57, .36, .6)
# initial = (0.72, 0.3, 0.18, 1.6, 0.5)
'''kick, temperature, xx, yy, zz, AA, BB'''
# initial = (1.11919, 0.9, 1., .10, 0.0, 5, 150)
# initial = (1., 5, 150)                      # fix parameters : kick, Tem, yy, zz
initial = (31.5451844, 1.24936717e+01, 2.45950926e+02)


'''ptdep_alice = classes.Fitting_gpu(total_phi_alice, total_dat_alice, (ptloww, pthigh), None, ptf, etaf, boundary, initial, "pTdependence")
ptdep_alice_result, ptdep_alice_error = ptdep_alice.fitting(None)                  # error를 고려하지 않으려는 경우
print('ALICE results : ', ptdep_alice_result)
print('ALICE error : ', ptdep_alice_error)'''

'''multi_atlas = classes.Fitting_gpu(phi_13TeV_multi_atlas_fitting, dat_13TeV_multi_atlas_fitting, None, multiplicity_atlas, ptf_atl, etaf_atl, boundary, initial, "Multiplicity")
multi_atlas_result, multi_atlas_error = multi_atlas.fitting(None)                  # error를 고려하지 않으려는 경우
kick = 0.798958702
Tem = 0.840820694
yy = 1.49370324
zz = 3.51939886e-12
multi_atlas_result = np.array([kick, Tem, multi_atlas_result[0], yy, zz, multi_atlas_result[1], multi_atlas_result[2]])
print('ATLAS results : ', multi_atlas_result)
print('ATLAS error : ', multi_atlas_error)'''

# '''multiplicity에 대한 associated yield만 가지고 fitting하고 phi correlation그리기'''
# multi_atlas = classes.Fitting_gpu(Multi_N, Multi_Y, None, multiplicity_atlas, (0.5, 5), (2, 5), boundary, initial, "Multiplicity")
# multi_atlas_result, multi_atlas_error = multi_atlas.multi_fitting()                  # error를 고려하지 않으려는 경우
# kick = 0.798958702
# Tem = 0.840820694
# yy = 1.49370324
# zz = 3.51939886e-12
# multi_atlas_result = np.array([kick, Tem, multi_atlas_result[0], yy, zz, multi_atlas_result[1], multi_atlas_result[2]])
# print('ATLAS results : ', multi_atlas_result)
# print('ATLAS error : ', multi_atlas_error)
# '''
#         ***Results : [0.798958702    0.840820694    5.38517847    1.49370324    3.51939886e-12    19.4957273e    117.807870]***
# '''

'''cms는 의미 least square fitting이 의미 없어 보임.'''
'''multi_cms = classes.Fitting_gpu(phi_13TeV_multi_cms_fitting, dat_13TeV_multi_cms_fitting, None, multiplicity_cms, ptf_cms, etaf_cms, boundary, initial, "Multiplicity")
multi_cms_result, multi_cms_error = multi_cms.fitting(None)                  # error를 고려하지 않으려는 경우
print('CMS results : ', multi_cms_result)
print('CMS error : ', multi_cms_error)'''

'''[7.98958702e-01 8.40820694e-01 9.99999859e+01 1.49370324e+00 3.51939886e-12 3.23136196e+00 2.52900687e+02]'''

multi_atlas_result = [0.798958702, 0.840820694, 107.325052959534, 1.49370324, 3.51939886e-12, 12.53484982633691, 260.21397797151315]

time_calculate = time.time()
print(f"calculate end : {time_calculate-time_start:.3f} sec")

'''그냥 일시적인 값'''
# ptdep_alice_result = [1.1192341315246068, 0.9874661996137095, 1.7763482993436224, 0.10489742070498227, 1.4902195191772722e-08]
# ptdep_cms_result = [1.1192341315246068, 0.9874661996137095, 1.7763482993436224, 0.10489742070498227, 1.4902195191772722e-08]



atlaserror = []     # Only 120<mutli<130, 130<multi
atlaserror.append(np.loadtxt(path[0]+f'/13TeV_120~130.csv',delimiter=',',usecols=[2], skiprows=3, max_rows=12))
atlaserror.append(np.loadtxt(path[0]+f'/13TeV_120~130.csv',delimiter=',',usecols=[3], skiprows=3, max_rows=12))
atlaserror.append(np.loadtxt(path[0]+f'/13TeV_130~.csv',delimiter=',',usecols=[2], skiprows=3, max_rows=12))
atlaserror.append(np.loadtxt(path[0]+f'/13TeV_130~.csv',delimiter=',',usecols=[3], skiprows=3, max_rows=12))
print(atlaserror    )

Yridge_phi = phi_13TeV_ptdep[-2::]
Yridge_dat = dat_13TeV_ptdep[-2::]
phi_13TeV_ptdep = phi_13TeV_ptdep[0:-2]
dat_13TeV_ptdep = dat_13TeV_ptdep[0:-2]

# multi_atlas_result = [2.66494908e-01, 2.00045938e-01, 5.00000000e+00, 3.60079032e-14, 1.02365240e-16, 2.00000000e+01, 2.68914081e+02]
fig1, axes1 = plt.subplots(nrows=3, ncols=3,figsize=(90,90),sharey='row', sharex='col')
#그래프 그리기
# alice = classes.Drawing_Graphs((1.6, 1.8), *ptdep_alice_result, None, None)
# cms = classes.Drawing_Graphs((2, 4), *multi_cms_result)
atlas = classes.Drawing_Graphs((2, 5), *multi_atlas_result)
# print(phi_13TeV_multi_atlas_fitting)
for j in range(3):
    axes1[j][0].set_ylabel(r'$\frac{1}{N_{trig}}\frac{dN^{pair}}{d\Delta\phi}-C_{ZYAM}$', size = 150)
    for i in range(3):
        multiplicity = (3*j+i)*10 + 55
        atlas_result = atlas.result_plot("Multiplicity", multiplicity, (0.5, 5), (min(phi_13TeV_multi_atlas_fitting[3*j+i]), max(phi_13TeV_multi_atlas_fitting[3*j+i])))
        axes1[j][i].scatter(phi_13TeV_multi_atlas[3*j+i], dat_13TeV_multi_atlas[3*j+i]-min(dat_13TeV_multi_atlas[3*j+i]), color = 'blue', s=2000, marker='o')
        axes1[j][i].plot(atlas_result[0], atlas_result[1]-min(atlas_result[1]), color = 'blue', linewidth=14, linestyle = '-')
        if j==2:
            axes1[j][i].set_xlabel(r'$\Delta\phi$', size=150)

        axes1[j][i].minorticks_on()

        # axes1[j][i].set_ylim(-0.001,0.07)
        axes1[j][i].set_xlim(-1.1,1.1)

        axes1[j][i].tick_params(axis='both',which='major',direction='in',width=4,length=35,labelsize=100, top = 'true', right='true')
        axes1[j][i].tick_params(axis='both',which='minor',direction='in',width=4,length=20,labelsize=100, top = 'true', right='true')
        axes1[j][i].grid(color='silver',linestyle=':',linewidth=3)

        # axes1[i][j].legend(framealpha=False, fontsize = 70)

axes1[2][1].errorbar(phi_13TeV_multi_atlas[7], dat_13TeV_multi_atlas[7]-min(dat_13TeV_multi_atlas[7]), yerr=(abs(atlaserror[0]), atlaserror[1]), color="blue", linestyle=' ', linewidth=14, capthick=6, capsize=30, zorder=1)
axes1[2][2].errorbar(phi_13TeV_multi_atlas[8], dat_13TeV_multi_atlas[8]-min(dat_13TeV_multi_atlas[8]), yerr=(abs(atlaserror[2]), atlaserror[3]), color="blue", linestyle=' ', linewidth=14, capthick=6, capsize=30, zorder=1)

axes1[0][0].text(-0.9, 0.0145, r'$50 \leq N^{rec}_{ch} < 60$', size = 150)
axes1[0][1].text(-0.9, 0.0145, r'$60 \leq N^{rec}_{ch} < 70$', size = 150)
axes1[0][2].text(-0.9, 0.0145, r'$70 \leq N^{rec}_{ch} < 80$', size = 150)
axes1[1][0].text(-0.9, 0.032, r'$80 \leq N^{rec}_{ch} < 90$', size = 150)
axes1[1][1].text(-0.9, 0.032, r'$90 \leq N^{rec}_{ch} < 100$', size = 150)
axes1[1][2].text(-0.9, 0.032, r'$100 \leq N^{rec}_{ch} < 110$', size = 150)
axes1[2][0].text(-0.9, 0.0625, r'$110 \leq N^{rec}_{ch} < 120$', size = 150)
axes1[2][1].text(-0.9, 0.0625, r'$120 \leq N^{rec}_{ch} < 130$', size = 150)
axes1[2][2].text(-0.9, 0.0625, r'$130 \leq N^{rec}_{ch}$', size = 150)

axes1[0][0].set_ylim(-0.001,0.0159)
axes1[1][0].set_ylim(-0.001,0.0349)
axes1[2][0].set_ylim(-0.001,0.0699)

fig1.tight_layout(h_pad = -1)
fig1.savefig('./rezero_atlas_Nk.png')
# fig1.savefig('/home/jaesung/Desktop/Dropbox/논문/rezero_atlas_Nk.png')
# , transparent = True

fig1.clear()

time_phicorr = time.time()
print(f"Graph, Phi correlation end : {time_phicorr-time_start:.3f} sec")



time_end = time.time()
print(f"Total end : {time_end-time_start:.3f} sec")