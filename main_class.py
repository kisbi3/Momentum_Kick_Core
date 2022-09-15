import time
time_start = time.time()
print("main_class.py Start")

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import pprint as pp
import sys
import matplotlib.ticker as ticker

# Functions Files
import class_pool as classes

#Check CM energy dependence

'''
    다른 Center of mass Energy라고 하더라도 동일한 multiplicity라면 그래프를 동시에 그려서 비교할 수 있는가?
    일단 동일한 multiplicity에 대해서 비교해보도록 하자.
'''

mpl.rcParams["text.usetex"] = True

path = ['./data/atldata/13TeV/', './data/alidata/', './data/cmsdata/', './data/atldata/2.76TeV/', './data/atldata/5.02TeV/']
phi_13TeV_multi_atlas = []
dat_13TeV_multi_atlas = []

# 순서 : alice, alice, alice, ..., cms, cms, cms, ....., atlas, alice, cms
# 마지막 두개는 Yridge
phi_13TeV_ptdep = []
dat_13TeV_ptdep = []
err_13TeV_ptdep = []
phi_07TeV_ptdep = []
dat_07TeV_ptdep = []
err_07TeV_ptdep = []
fitting_error = []
# delta_phi_zyam = []
'''append atlas data'''
for i in range(5):
    start = 10*i + 90
    end = 10*i + 100
    if start == 130:
        phi_13TeV_multi_atlas.append(np.loadtxt(path[0]+f'{start}~.csv',delimiter=',',usecols=[0], skiprows=3, max_rows=12))
        dat_13TeV_multi_atlas.append(np.loadtxt(path[0]+f'{start}~.csv',delimiter=',',usecols=[1], skiprows=3, max_rows=12))
    else:
        phi_13TeV_multi_atlas.append(np.loadtxt(path[0]+f'{start}~{end}.csv',delimiter=',',usecols=[0], skiprows=3, max_rows=12))
        dat_13TeV_multi_atlas.append(np.loadtxt(path[0]+f'{start}~{end}.csv',delimiter=',',usecols=[1], skiprows=3, max_rows=12))
    dat_13TeV_multi_atlas[i] -= min(dat_13TeV_multi_atlas[i])
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

'''ATLAS data'''
phi_13TeV_ptdep.append(np.loadtxt(path[0]+'90~.csv',delimiter=',',usecols=[0],skiprows=3,max_rows=12))
dat_13TeV_ptdep.append(np.loadtxt(path[0]+'90~.csv',delimiter=',',usecols=[1],skiprows=3,max_rows=12))

''' table에 있는 delta phi_CZYAM값을 직접 이용하여 fitting하기 위함 '''
''' ALICE의 CZYAM을 알 수 없기 때문에 이를 통일하기 위해서 table의 delta phi_CZYAM를 이용하기 힘들어 보임. '''
# for i in range(len(dat_13TeV_ptdep)):
#     dat_13TeV_ptdep[i] -= min(dat_13TeV_ptdep[i])

'''append 하는 list가 전역변수 이므로 return이 필요없다. return이 없을 경우 원래는 함수 내에서 변수로 끝나기 때문에 이를 주의하자.'''

'''Yridge data append'''
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

'''Yridge pt range'''
ptloww = []
pthigh = []
ptloww_07 = []
pthigh_07 = []
def Yridge_ptrange():
    ptloww_ali = np.loadtxt(path[1]+'Y^mathrm{ridge}.csv',delimiter=',',usecols=[1],skiprows=12,max_rows=7)
    pthigh_ali = np.loadtxt(path[1]+'Y^mathrm{ridge}.csv',delimiter=',',usecols=[2],skiprows=12,max_rows=7)
    ptloww_cms = np.loadtxt(path[2]+'Table33.csv',delimiter=',',usecols=[6],skiprows=14,max_rows=9)
    pthigh_cms = np.loadtxt(path[2]+'Table33.csv',delimiter=',',usecols=[7],skiprows=14,max_rows=9)
    ptloww.append(np.loadtxt(path[1]+'Y^mathrm{ridge}.csv',delimiter=',',usecols=[1],skiprows=12,max_rows=7))
    pthigh.append(np.loadtxt(path[1]+'Y^mathrm{ridge}.csv',delimiter=',',usecols=[2],skiprows=12,max_rows=7))
    ptloww.append(np.loadtxt(path[2]+'Table33.csv',delimiter=',',usecols=[6],skiprows=14,max_rows=9))
    pthigh.append(np.loadtxt(path[2]+'Table33.csv',delimiter=',',usecols=[7],skiprows=14,max_rows=9))
    ptloww_07.append(np.loadtxt(path[2]+'Table34.csv',delimiter=',',usecols=[6],skiprows=14,max_rows=9))
    pthigh_07.append(np.loadtxt(path[2]+'Table34.csv',delimiter=',',usecols=[7],skiprows=14,max_rows=9))
Yridge_ptrange()
Yridge_append()

'''
    Append data for check center of mass energy
    data : ATLAS(2.76), ATLAS(5.02), ATLAS(13)
    Only append 90<N_ch<100
'''
CMenergydep_phi = []
CMenergydep_dat = []
CMenergydep_err = []
for i in range(3):
    # 2.76TeV
    if i==0:
        CMenergydep_phi.append(np.loadtxt(path[3]+'90~100.csv',delimiter=',',usecols=[0],skiprows=2,max_rows=12))
        CMenergydep_dat.append(np.loadtxt(path[3]+'90~100.csv',delimiter=',',usecols=[1],skiprows=2,max_rows=12))
        CMenergydep_err.append(np.loadtxt(path[3]+'90~100.csv',delimiter=',',usecols=[2],skiprows=2,max_rows=12))
        CMenergydep_err.append(np.loadtxt(path[3]+'90~100.csv',delimiter=',',usecols=[3],skiprows=2,max_rows=12))
    # 5.02TeV
    elif i==1:
        CMenergydep_phi.append(np.loadtxt(path[4]+'90~100.csv',delimiter=',',usecols=[0],skiprows=1,max_rows=12))
        CMenergydep_dat.append(np.loadtxt(path[4]+'90~100.csv',delimiter=',',usecols=[1],skiprows=1,max_rows=12))
    # 13TeV
    elif i==2:
        CMenergydep_phi.append(np.loadtxt(path[0]+'90~100.csv',delimiter=',',usecols=[0],skiprows=3,max_rows=12))
        CMenergydep_dat.append(np.loadtxt(path[0]+'90~100.csv',delimiter=',',usecols=[1],skiprows=3,max_rows=12))

'''[:] 는 deep copy를 위해 필요함'''
phi_13TeV_ptdep_fitting = phi_13TeV_ptdep[:]          # fitting에 사용하는 데이터에서 Yridge를 포함하려는 경우
dat_13TeV_ptdep_fitting = dat_13TeV_ptdep[:]          # fitting에 사용하는 데이터에서 Yridge를 포함하려는 경우
del phi_13TeV_ptdep_fitting[7]                        # delete ATLAS phi
del dat_13TeV_ptdep_fitting[7]                        # delete ATLAS data
del phi_13TeV_ptdep_fitting[3]                        # delete 0.1<pT<1 phi
del dat_13TeV_ptdep_fitting[3]                        # delete 0.1<pT<1 data

phi_07TeV_ptdep_fitting = phi_07TeV_ptdep[1::]
dat_07TeV_ptdep_fitting = dat_07TeV_ptdep[1::]
'''
    최솟값인 부분을 찾아서 fitting 데이터 자르기(양쪽 끝 자르기)
    Yridge가 있는 경우를 상정하므로 Yridge 데이터를 fitting하지 않으려는 경우, 이 부분을 다시 세팅해야 함
'''
def datacut():
    for i in range(len(dat_13TeV_ptdep_fitting)-2):
        argmin = np.argmin(dat_13TeV_ptdep_fitting[i])
        if argmin == 0:
            pass
        elif argmin == len(dat_13TeV_ptdep_fitting[i]):
            pass
        elif argmin<len(dat_13TeV_ptdep_fitting[i])/2:
            dat_13TeV_ptdep_fitting[i] = dat_13TeV_ptdep_fitting[i][argmin:-argmin]
            phi_13TeV_ptdep_fitting[i] = phi_13TeV_ptdep_fitting[i][argmin:-argmin]
        elif argmin>len(dat_13TeV_ptdep_fitting[i])/2:
            dat_13TeV_ptdep_fitting[i] = dat_13TeV_ptdep_fitting[i][len(dat_13TeV_ptdep_fitting[i])-argmin-1:argmin+1]
            phi_13TeV_ptdep_fitting[i] = phi_13TeV_ptdep_fitting[i][len(phi_13TeV_ptdep_fitting[i])-argmin-1:argmin+1]
    for i in range(len(dat_07TeV_ptdep_fitting)-1):
        argmin = np.argmin(dat_07TeV_ptdep_fitting[i])
        if argmin == 0:
            pass
        elif argmin == len(dat_07TeV_ptdep_fitting[i]):
            pass
        elif argmin<len(dat_07TeV_ptdep_fitting[i])/2:
            dat_07TeV_ptdep_fitting[i] = dat_07TeV_ptdep_fitting[i][argmin:-argmin]
            phi_07TeV_ptdep_fitting[i] = phi_07TeV_ptdep_fitting[i][argmin:-argmin]
        elif argmin>len(dat_07TeV_ptdep_fitting[i])/2:
            dat_07TeV_ptdep_fitting[i] = dat_07TeV_ptdep_fitting[i][len(dat_07TeV_ptdep_fitting[i])-argmin-1:argmin+1]
            phi_07TeV_ptdep_fitting[i] = phi_07TeV_ptdep_fitting[i][len(phi_07TeV_ptdep_fitting[i])-argmin-1:argmin+1]
datacut()

total_boundary = ((0.5, .5, 0, 0, 0),(3, 2., 4, 1e-10, 1e-10))
total_initial = (.7, 1., 1.5, 0, 0)
'''Fitting 13TeV data'''
def fit_13tev():
    ptf = [(1, 2), (2, 3), (3, 4), (1, 2), (2, 3), (3, 4)]
    etaf = [(1.6, 1.8), (1.6, 1.8), (1.6, 1.8), (2, 4), (2, 4), (2, 4)]
    '''boundary conditions'''
    # boundary = ((0., 0., -20, -10, -10),(5, 3., 20, 10, 10))
    boundary = total_boundary
    '''initial parameters'''
    # initial = (1., 0.5, 2, 3, 0)
    initial = total_initial
    ptdep = classes.Fitting_gpu(13000, phi_13TeV_ptdep_fitting, dat_13TeV_ptdep_fitting, (ptloww, pthigh), None, ptf, etaf, boundary, initial, "pTdependence")
    ptdep_result, ptdep_error = ptdep.fitting(None)                  # error를 고려하지 않으려는 경우
    print(ptdep_result)
    print(ptdep_error)

'''Fitting 7TeV data'''
def fit_7tev():
    ptf = [(1, 2), (2, 3), (3, 4)]
    etaf = [(2, 4), (2, 4), (2, 4)]
    '''boundary conditions'''
    # boundary = ((0., 0., 0, 0, 0),(5, 3., 20, 10, 10))
    boundary = total_boundary
    '''initial parameters'''
    # initial = (1., 0.5, 2, 3, 0)
    initial = total_initial
    ptdep = classes.Fitting_gpu(7000, phi_07TeV_ptdep_fitting, dat_07TeV_ptdep_fitting, (ptloww_07, pthigh_07), None, ptf, etaf, boundary, initial, "pTdependence")
    ptdep_result_07, ptdep_error_07 = ptdep.fitting(None)                  # error를 고려하지 않으려는 경우
    print(ptdep_result_07)
    print(ptdep_error_07)

'''multiplicity fitting'''
def fit_multipl():
    # boundary2 = ((0.7, 0.4, 0, 0, 1, 100),(5, 0.7, 5, 10, 20, 300))
    # initial2 = (0.9, 0.65, 0.83, 0.5, 14, 250)
    # multipl = classes.Fitting_gpu(phi_13TeV_multi_atlas, dat_13TeV_multi_atlas, 95, (0.5, 5), (2, 5), boundary2, initial2, "Multiplicity")
    # multipl_result, multipl_error = multipl.fitting()
    # print(multipl_result)
    # print(multipl_error)
    # dist = []
    # for i in range(5):
    #     multi= 10*i + 95
    #     dist.append(Ridge(Aridge, *popt, multi, 2, 5))
    pass

'''To Check Center of mass Energy dependence'''
''' Multiplicity : 90~100'''
ptdep_result_cm = []
ptdep_error_cm = []
def fit_cmenerg():
    sqrSnn_str = [2.76, 5.02, 13]
    sqrSnn = [2760, 5020, 13000]
    ptf = [(0.5, 5)]
    etaf = [(2, 5)]
    '''boundary conditions'''
    # boundary = ((0.0, 0.8, -30, -10, -10),(5, 2., 30, 10, 10))
    boundary = total_boundary
    '''initial parameters'''
    # initial = (1., 1., 2, 3, 0)
    initial = total_initial
    for i in range(len(sqrSnn)):
        print(f"\n**********\n sqrSnn = {sqrSnn_str[i]} Start \n\n**********\n")
        ptdep = classes.Fitting_gpu(sqrSnn[i], CMenergydep_phi[i], CMenergydep_dat[i]-min(CMenergydep_dat[i]), None, None, ptf, etaf, boundary, initial, "CMenergy")
        result, error = ptdep.fitting(None)
        print(result)
        ptdep_result_cm.append(result)
        ptdep_error_cm.append(error)
        print(f"\n**********\n sqrSnn = {sqrSnn_str[i]} End \n\n**********\n")
    print(ptdep_result_cm)
    print(ptdep_error_cm)


# fit_13tev()
# fit_7tev()
# fit_multipl()
fit_cmenerg()


# ptdep_result_cm = [np.array([6.50898500e-02, 1.00053422e+00, 1.99918447e+01, 3.15781968e-12, 1.91795188e-28]),
#                    np.array([2.10274974e-01, 6.00607568e-02, 1.89447712e+01, 1.66079658e-02, 1.14899736e-06]), 
#                    np.array([1.00006451e+00, 1.15455510e+00, 1.76066500e+01, 5.72307164e-06, 1.36854475e+00])]


time_calculate = time.time()
print(f"calculate end : {time_calculate-time_start:.3f} sec")


Yridge_phi07 = phi_07TeV_ptdep[-1]
Yridge_dat07 = dat_07TeV_ptdep[-1]
Yridge_phi = phi_13TeV_ptdep[-2::]
Yridge_dat = dat_13TeV_ptdep[-2::]
phi_13TeV_ptdep = phi_13TeV_ptdep[0:-2]
dat_13TeV_ptdep = dat_13TeV_ptdep[0:-2]


ptdep_result = [9.62040979e-01, 1.08113653e+00, 2.62053618e+00, 2.86583565e-01, 4.45772582e-04]
ptdep_result_07 = [1.68531124, 1.43640594, 7.85469003, 0.80428453, 0.88283783]

fig1, axes1 = plt.subplots(nrows=1, ncols=5,figsize=(125,20))
#그래프 그리기
alice = classes.Drawing_Graphs(13000, (1.6, 1.8), *ptdep_result, None, None)
cms = classes.Drawing_Graphs(13000, (2, 4), *ptdep_result, None, None)
cms_07 = classes.Drawing_Graphs(7000, (2, 4), *ptdep_result_07, None, None)
atlas = classes.Drawing_Graphs(13000, (2, 5), *ptdep_result, None, None)

'''pT dependence phi correlation graph'''
def drawgraph_ptdep_phicorr():
    for i in range(5):
        if i==0:
            ptf = (0.1, 1)
            '''cms plot 13TeV'''
            cms_result = cms.result_plot("pTdependence", None, ptf, (min(phi_13TeV_ptdep[i+3]), max(phi_13TeV_ptdep[i+3])))
            axes1[i].plot(cms_result[0], cms_result[1], color = "black", linewidth=7, linestyle='-')
            axes1[i].errorbar(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3], yerr=(abs(err_13TeV_ptdep[2*(i+3)+1]),err_13TeV_ptdep[2*(i+3)]), color="black", linestyle=' ', linewidth=7, capthick=3, capsize=15)
            axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3], edgecolors="black", s=800, marker='o', facecolors='none', linewidths=7)
            axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3], s=800, marker='+', facecolors='black', linewidths=7)
            '''cms plot 7TeV'''
            cms_07result = cms_07.result_plot("pTdependence", None, ptf, (min(phi_07TeV_ptdep[i]), max(phi_07TeV_ptdep[i])))
            axes1[i].plot(cms_07result[0], cms_07result[1], color = "grey", linewidth=7, linestyle='-')
            axes1[i].errorbar(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i], yerr=(abs(err_07TeV_ptdep[2*i+1]),err_07TeV_ptdep[2*i]), color="grey", linestyle=' ', linewidth=7, capthick=3, capsize=15)
            axes1[i].scatter(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i], edgecolors="grey", s=800, marker='o', facecolors='none', linewidths=7)
            axes1[i].scatter(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i], s=800, marker='+', facecolors='grey', linewidths=7)
            axes1[i].set_title(r'$0.1<p_{T, \, \mathrm{trig(assoc)}}<1$', size = 70, pad=30)
        elif i==4:
            atlas_result = atlas.result_plot("pTdependence", None, (0.5, 5), (min(phi_13TeV_ptdep[-1]), max(phi_13TeV_ptdep[-1])))
            axes1[i].plot(atlas_result[0], atlas_result[1], color = "blue", linewidth=7, linestyle='-')
            axes1[i].scatter(phi_13TeV_ptdep[-1], dat_13TeV_ptdep[-1]-min(dat_13TeV_ptdep[-1]), facecolors='blue', edgecolors="blue", s=600, marker='o', linewidths=7)
            axes1[i].set_title(r'0.5$<p_{T, \, \mathrm{trig(assoc)}}<$5', size = 70, pad=30)
        else:
            ptf = (i, i+1)
            alice_result = alice.result_plot("pTdependence", None, ptf, (min(phi_13TeV_ptdep[i-1]), max(phi_13TeV_ptdep[i-1])))
            cms_result = cms.result_plot("pTdependence", None, ptf, (min(phi_13TeV_ptdep[i+3]), max(phi_13TeV_ptdep[i+3])))
            axes1[i].plot(alice_result[0], alice_result[1], color = "red", linewidth=7, linestyle='-')
            axes1[i].plot(cms_result[0], cms_result[1], color = "black", linewidth=7, linestyle='-')
            '''alice plot'''
            axes1[i].errorbar(phi_13TeV_ptdep[i-1], dat_13TeV_ptdep[i-1], yerr=(abs(err_13TeV_ptdep[2*i-1]),err_13TeV_ptdep[2*i-2]), color="red", linestyle=' ', linewidth=7, capthick=3, capsize=15)
            axes1[i].scatter(phi_13TeV_ptdep[i-1], dat_13TeV_ptdep[i-1], edgecolors="red", s=800, marker='o', facecolors='none', linewidths=7)
            axes1[i].scatter(phi_13TeV_ptdep[i-1], dat_13TeV_ptdep[i-1], s=800, marker='+', facecolors='red', linewidths=7)
            '''cms plot 13TeV'''
            axes1[i].errorbar(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3], yerr=(abs(err_13TeV_ptdep[2*(i+3)+1]),err_13TeV_ptdep[2*(i+3)]), color="black", linestyle=' ', linewidth=7, capthick=3, capsize=15)
            axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3], edgecolors="black", s=800, marker='o', facecolors='none', linewidths=7)
            axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3], s=800, marker='+', facecolors='black', linewidths=7)
            '''cms plot 7TeV'''
            cms_07result = cms_07.result_plot("pTdependence", None, ptf, (min(phi_07TeV_ptdep_fitting[i-1]), max(phi_07TeV_ptdep_fitting[i-1])))
            axes1[i].plot(cms_07result[0], cms_07result[1], color = "grey", linewidth=7, linestyle='-')
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

    fig1.savefig('./phiCorr_Test.png')
    fig1.savefig('/home/jaesung/Dropbox/ohno/phiCorr_Test.png')

'''pT dependence Y^ridge graph'''
def drawgraph_ptdep_Yridge():
    fig2, axis2 = plt.subplots(nrows=1, ncols=1,figsize=(40,20))

    '''Line'''
    alice_Yridge = alice.Yridge_line("Subtract")
    cms_Yridge = cms.Yridge_line("Subtract")
    cms_07Yridge = cms_07.Yridge_line("Subtract")
    axis2.plot(alice_Yridge[0], alice_Yridge[1], color = "red", linewidth=7, linestyle='-')
    axis2.plot(cms_Yridge[0], cms_Yridge[1], color = "black", linewidth=7, linestyle='-')
    axis2.plot(cms_07Yridge[0], cms_07Yridge[1], color = "grey", linewidth=7, linestyle='-')
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
    fig2.savefig('./Yridge_Test.png')
    fig2.savefig('/home/jaesung/Dropbox/ohno/Yridge_Test.png')

'''pT dependence FrNk graph'''
def drawgraph_ptdep_frnk():
    fig3 = plt.figure()
    ax = plt.axes()
    fig3.set_size_inches(35, 16.534, forward=True)

    def FrNk_func(pt, xx, yy, zz):
        return xx*np.exp(-yy/pt-zz*pt)

    AuAu_200GeV = [4, 0, 0]
    PbPb_276TeV = [20.2, 1.395, 0.207]
    pp_13TeV = [ptdep_result[2], ptdep_result[3], ptdep_result[4]]
    pp_07TeV = [ptdep_result_07[2], ptdep_result_07[3], ptdep_result_07[4]]
    '''Hanul's pp 13TeV FrNk result plot [[pt(Average)], [FrNk]]'''
    Hanul_FrNk = [[1.5, 2.5], [0.93, 1.37]]
    Hanul_FrNk_error = [[0.5, 0.5], [0.5, 0.5]]
    ptf = np.arange(0.01,4,0.01)

    plt.plot(ptf, FrNk_func(ptf, *AuAu_200GeV), color = 'red', linewidth=7, label=r'$AuAu, \, 200\mathrm{GeV}$')
    plt.plot(ptf, FrNk_func(ptf, *PbPb_276TeV), color = 'black', linewidth=7, label=r'$PbPb, \, 2.76\mathrm{TeV}$')
    plt.plot(ptf, FrNk_func(ptf, *pp_13TeV), color = 'blue', linewidth=7, label=r'$pp, \, 13\mathrm{TeV}$')
    plt.plot(ptf, FrNk_func(ptf, *pp_07TeV), color = 'grey', linewidth=7, label=r'$pp, \, 07\mathrm{TeV}$')
    plt.scatter(Hanul_FrNk[0], Hanul_FrNk[1], edgecolor = 'green', facecolors='none', s=900, marker='o', linewidths=5, zorder=2, label=r'$pp, \, 13\mathrm{TeV}$')
    plt.scatter(Hanul_FrNk[0], Hanul_FrNk[1], facecolors='green', s=900, marker='+', linewidths=5, zorder=2)
    plt.errorbar(Hanul_FrNk[0], Hanul_FrNk[1], xerr=Hanul_FrNk_error, color="green", linestyle=' ', linewidth=7, capthick=3, capsize=15)

    plt.xlabel(r'$p_T^{\mathrm{trig}}$',size=70)
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

'''CM energy dependence phi correlation graph + CM energy frNk'''
def drawgraph_cmdep_phicorr():
    fig3 = plt.figure()
    ax = plt.axes()
    fig3.set_size_inches(35, 16.534, forward=True)
    atlas_276 = classes.Drawing_Graphs(2760, (2, 5), *ptdep_result_cm[0], None, None)
    atlas_502 = classes.Drawing_Graphs(5020, (2, 5), *ptdep_result_cm[1], None, None)
    atlas_130 = classes.Drawing_Graphs(13000, (2, 5), *ptdep_result_cm[2], None, None)
    cmenergy_2760 = atlas_276.result_plot("pTdependence", None, (0.5, 5), (min(CMenergydep_phi[0]), max(CMenergydep_phi[0])))
    cmenergy_5020 = atlas_502.result_plot("pTdependence", None, (0.5, 5), (min(CMenergydep_phi[1]), max(CMenergydep_phi[1])))
    cmenergy_13000 = atlas_130.result_plot("pTdependence", None, (0.5, 5), (min(CMenergydep_phi[2]), max(CMenergydep_phi[2])))
    plt.plot(cmenergy_2760[0], (5-0.5)*cmenergy_2760[1], color = "red", linewidth=7, linestyle='-')
    plt.plot(cmenergy_5020[0], (5-0.5)*cmenergy_5020[1], color = "green", linewidth=7, linestyle='-')
    plt.plot(cmenergy_13000[0], (5-0.5)*cmenergy_13000[1], color = "blue", linewidth=7, linestyle='-')
    plt.scatter(CMenergydep_phi[0], CMenergydep_dat[0]-min(CMenergydep_dat[0]), facecolors='red', edgecolors="red", s=600, marker='o', linewidths=7, label=r'$\sqrt{s_{NN}} \,=\, 2.76 \mathrm{TeV}$')
    plt.errorbar(CMenergydep_phi[0], CMenergydep_dat[0]-min(CMenergydep_dat[0]), yerr=(abs(CMenergydep_err[1]-CMenergydep_dat[0]),CMenergydep_err[0]-CMenergydep_dat[0]), color="red", linestyle=' ', linewidth=7, capthick=3, capsize=15)
    plt.scatter(CMenergydep_phi[1], CMenergydep_dat[1]-min(CMenergydep_dat[1]), facecolors='green', edgecolors="green", s=600, marker='o', linewidths=7, label=r'$\sqrt{s_{NN}} \,=\, 5.02 \mathrm{TeV}$')
    plt.scatter(CMenergydep_phi[2], CMenergydep_dat[2]-min(CMenergydep_dat[2]), facecolors='blue', edgecolors="blue", s=600, marker='o', linewidths=7, label=r'$\sqrt{s_{NN}} \,=\, 13 \mathrm{TeV}$')

    plt.xlabel(r'$\Delta\phi$',size=70)
    plt.ylabel(r'$\frac{1}{N_{\mathrm{trig}}}\frac{dN^{\mathrm{pair}}}{d\Delta\phi}-C_{\mathrm{ZYAM}}$',size=70)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '${:g}$'.format(y)))
    plt.tick_params(axis='both',which='major',direction='in',width=2,length=30,labelsize=45, top='true')
    plt.tick_params(axis='both',which='minor',direction='in',width=2,length=15,labelsize=45, top='true')

    plt.grid(color='silver',linestyle=':',linewidth=5, zorder=0)
    plt.legend(fontsize=45, loc='upper left')

    plt.tight_layout()

    fig3.savefig('./cmdep_phicorr.png')
    fig3.savefig('/home/jaesung/Dropbox/ohno/cmdep_phicorr.png')

    fig3 = plt.figure()
    ax = plt.axes()
    fig3.set_size_inches(35, 16.534, forward=True)

    def FrNk_func(pt, xx, yy, zz):
        return xx*np.exp(-yy/pt-zz*pt)

    AuAu_200GeV = [4, 0, 0]
    PbPb_276TeV = [20.2, 1.395, 0.207]
    pp_27TeV = [ptdep_result_cm[0][2], ptdep_result_cm[0][3], ptdep_result_cm[0][4]]
    pp_50TeV = [ptdep_result_cm[1][2], ptdep_result_cm[1][3], ptdep_result_cm[1][4]]
    pp_13TeV = [ptdep_result_cm[2][2], ptdep_result_cm[2][3], ptdep_result_cm[2][4]]
    pp_13TeV_alicms = [ptdep_result[2], ptdep_result[3], ptdep_result[4]]
    pp_07TeV = [ptdep_result_07[2], ptdep_result_07[3], ptdep_result_07[4]]
    '''Hanul's pp 13TeV FrNk result plot [[pt(Average)], [FrNk]]'''
    Hanul_FrNk = [[1.5, 2.5], [0.93, 1.37]]
    Hanul_FrNk_error = [[0.5, 0.5], [0.5, 0.5]]
    ptf = np.arange(0.01,4,0.01)

    plt.plot(ptf, FrNk_func(ptf, *AuAu_200GeV), color = 'blueviolet', linewidth=7, label=r'$AuAu, \, 200\mathrm{GeV}$')
    plt.plot(ptf, FrNk_func(ptf, *PbPb_276TeV), color = 'black', linewidth=7, label=r'$PbPb, \, 2.76\mathrm{TeV}$')
    plt.plot(ptf, FrNk_func(ptf, *pp_13TeV_alicms), color = 'cyan', linewidth=7, label=r'$pp \,\mathrm{(ALICE+CMS)}, \, 13\mathrm{TeV}$')
    plt.plot(ptf, FrNk_func(ptf, *pp_13TeV), color = 'blue', linewidth=7, label=r'$pp, \, 13\mathrm{TeV} \,\,\mathrm{(ATLAS)} $')
    plt.plot(ptf, FrNk_func(ptf, *pp_50TeV), color = 'green', linewidth=7, label=r'$pp, \, 5.02\mathrm{TeV}\,\,\mathrm{(ATLAS)}$')
    plt.plot(ptf, FrNk_func(ptf, *pp_27TeV), color = 'red', linewidth=7, label=r'$pp, \, 2.76\mathrm{TeV}\,\,\mathrm{(ATLAS)}$')
    plt.plot(ptf, FrNk_func(ptf, *pp_07TeV), color = 'grey', linewidth=7, label=r'$pp, \, 7\mathrm{TeV}$')
    plt.scatter(Hanul_FrNk[0], Hanul_FrNk[1], edgecolor = 'violet', facecolors='none', s=900, marker='o', linewidths=5, zorder=2, label=r'$pp, \, 13\mathrm{TeV}(\mathrm{Hanul})$')
    plt.scatter(Hanul_FrNk[0], Hanul_FrNk[1], facecolors='violet', s=900, marker='+', linewidths=5, zorder=2)
    plt.errorbar(Hanul_FrNk[0], Hanul_FrNk[1], xerr=Hanul_FrNk_error, color="violet", linestyle=' ', linewidth=7, capthick=3, capsize=15)

    plt.xlabel(r'$p_T^{\mathrm{trig}}$',size=70)
    plt.ylabel(r'$f_{R} \langle N_k \rangle $',size=70)
    plt.xlim(0,4)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '${:g}$'.format(y)))
    plt.tick_params(axis='both',which='major',direction='in',width=2,length=30,labelsize=45, top='true')
    plt.tick_params(axis='both',which='minor',direction='in',width=2,length=15,labelsize=45, top='true')

    plt.grid(color='silver',linestyle=':',linewidth=5, zorder=0)
    plt.legend(fontsize=45, loc='upper right')

    plt.tight_layout()

    fig3.savefig('./CMenerg_FrNk.png')
    fig3.savefig('/home/jaesung/Dropbox/ohno/CMenerg_FrNk.png')

# drawgraph_ptdep_phicorr()
time_phicorr = time.time()
print(f"Graph, Phi correlation end : {time_phicorr-time_calculate:.3f} sec")
# drawgraph_ptdep_Yridge()
time_yridge = time.time()
print(f"Graph, Yridge end : {time_yridge-time_phicorr:.3f} sec")
# drawgraph_ptdep_frnk()
time_frnk = time.time()
print(f"FrNk end : {time_frnk-time_yridge:.3f} sec")
drawgraph_cmdep_phicorr()

time_end = time.time()
print(f"Total end : {time_end-time_start:.3f} sec")