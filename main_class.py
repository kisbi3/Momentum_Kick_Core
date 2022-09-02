import time
time_start = time.time()
print("main_class.py Start")

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import pprint as pp
import matplotlib.ticker as ticker

# Functions Files
import class_pool as classes


mpl.rcParams["text.usetex"] = True

path = ['/home/jaesung/OneDrive/Code/WongCode/Momentum_Kick/13TeV_atlas/atlasgraphs', '/home/jaesung/OneDrive/Code/WongCode/Momentum_Kick/13TeV-Alice/HEPData-ins1840098-v1-csv', '/home/jaesung/OneDrive/Code/WongCode/Momentum_Kick/13TeV/HEPData-ins1397173-v1-csv']
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
for i in range(3):
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
    
    '''13TeV'''
    if i==1:
        '''CMS data, 0.1~1, 1~2, 2~3, 3~4'''
        for j in range(4):
            table = 2*j+25
            if j == 0:
                phi_13TeV_ptdep.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[0], skiprows=19, max_rows=11))
                dat_13TeV_ptdep.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[1], skiprows=19, max_rows=11))
                err_13TeV_ptdep.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[2], skiprows=19, max_rows=11))
                err_13TeV_ptdep.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[3], skiprows=19, max_rows=11))
                fitting_error.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[3], skiprows=19, max_rows=11))
            else:                
                phi_13TeV_ptdep.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[0], skiprows=18, max_rows=13))
                dat_13TeV_ptdep.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[1], skiprows=18, max_rows=13))
                err_13TeV_ptdep.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[2], skiprows=18, max_rows=13))
                err_13TeV_ptdep.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[3], skiprows=18, max_rows=13))
                fitting_error.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[3], skiprows=18, max_rows=13))

    '''7TeV'''
    if i==2:
        '''CMS data, 0.1~1, 1~2, 2~3, 3~4'''
        for j in range(4):
            table = 2*j+26
            if j == 0:
                phi_07TeV_ptdep.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[0], skiprows=19, max_rows=11))
                dat_07TeV_ptdep.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[1], skiprows=19, max_rows=11))
                err_07TeV_ptdep.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[2], skiprows=19, max_rows=11))
                err_07TeV_ptdep.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[3], skiprows=19, max_rows=11))
            else:                
                phi_07TeV_ptdep.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[0], skiprows=18, max_rows=13))
                dat_07TeV_ptdep.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[1], skiprows=18, max_rows=13))
                err_07TeV_ptdep.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[2], skiprows=18, max_rows=13))
                err_07TeV_ptdep.append(np.loadtxt(path[2]+f'/Table{table}.csv',delimiter=',',usecols=[3], skiprows=18, max_rows=13))

'''ATLAS data'''
phi_13TeV_ptdep.append(np.loadtxt(path[0]+'/13TeV_90~.csv',delimiter=',',usecols=[0],skiprows=3,max_rows=12))
dat_13TeV_ptdep.append(np.loadtxt(path[0]+'/13TeV_90~.csv',delimiter=',',usecols=[1],skiprows=3,max_rows=12))

''' table에 있는 delta phi_CZYAM값을 직접 이용하여 fitting하기 위함 '''
''' ALICE의 CZYAM을 알 수 없기 때문에 이를 통일하기 위해서 table의 delta phi_CZYAM를 이용하기 힘들어 보임. '''
# for i in range(len(dat_13TeV_ptdep)):
#     dat_13TeV_ptdep[i] -= min(dat_13TeV_ptdep[i])

fitting_error.append(np.zeros(len(phi_13TeV_ptdep[-1]))+0.000703141335931843)


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
err_sta1=np.loadtxt(path[2]+'/Table34.csv',delimiter=',',usecols=[2],skiprows=14,max_rows=9)
err_sta2=np.loadtxt(path[2]+'/Table34.csv',delimiter=',',usecols=[3],skiprows=14,max_rows=9)
err_sys1=np.loadtxt(path[2]+'/Table34.csv',delimiter=',',usecols=[4],skiprows=14,max_rows=9)
err_sys2=np.loadtxt(path[2]+'/Table34.csv',delimiter=',',usecols=[5],skiprows=14,max_rows=9)
phi_07TeV_ptdep.append(np.loadtxt(path[2]+'/Table34.csv',delimiter=',',usecols=[0],skiprows=14,max_rows=9))
dat_07TeV_ptdep.append(np.loadtxt(path[2]+'/Table34.csv',delimiter=',',usecols=[1],skiprows=14,max_rows=9))
err_07TeV_ptdep.append((err_sta1**2+err_sys1**2)**0.5)
err_07TeV_ptdep.append((err_sta2**2+err_sys2**2)**0.5)

# Yridge pt range
ptloww = []
pthigh = []
ptloww_07 = []
pthigh_07 = []

ptloww_ali = np.loadtxt(path[1]+'/Y^mathrm{ridge}.csv',delimiter=',',usecols=[1],skiprows=12,max_rows=7)
pthigh_ali = np.loadtxt(path[1]+'/Y^mathrm{ridge}.csv',delimiter=',',usecols=[2],skiprows=12,max_rows=7)
ptloww_cms = np.loadtxt(path[2]+'/Table33.csv',delimiter=',',usecols=[6],skiprows=14,max_rows=9)
pthigh_cms = np.loadtxt(path[2]+'/Table33.csv',delimiter=',',usecols=[7],skiprows=14,max_rows=9)

ptloww.append(np.loadtxt(path[1]+'/Y^mathrm{ridge}.csv',delimiter=',',usecols=[1],skiprows=12,max_rows=7))
pthigh.append(np.loadtxt(path[1]+'/Y^mathrm{ridge}.csv',delimiter=',',usecols=[2],skiprows=12,max_rows=7))
ptloww.append(np.loadtxt(path[2]+'/Table33.csv',delimiter=',',usecols=[6],skiprows=14,max_rows=9))
pthigh.append(np.loadtxt(path[2]+'/Table33.csv',delimiter=',',usecols=[7],skiprows=14,max_rows=9))
ptloww_07.append(np.loadtxt(path[2]+'/Table34.csv',delimiter=',',usecols=[6],skiprows=14,max_rows=9))
pthigh_07.append(np.loadtxt(path[2]+'/Table34.csv',delimiter=',',usecols=[7],skiprows=14,max_rows=9))


'''Fitting with ATLAS Data'''
# ptf = [(1, 2), (2, 3), (3, 4), (1, 2), (2, 3), (3, 4), (0.5, 5)]
# etaf = [(1.6, 1.8), (1.6, 1.8), (1.6, 1.8), (2, 4), (2, 4), (2, 4), (2, 5)]
'''Fitting without ATLAS data'''
# ptf = [(1, 2), (2, 3), (3, 4), (0.1, 1), (1, 2), (2, 3), (3, 4)]
# etaf = [(1.6, 1.8), (1.6, 1.8), (1.6, 1.8), (2, 4), (2, 4), (2, 4), (2, 4)]
ptf = [(1, 2), (2, 3), (3, 4), (1, 2), (2, 3), (3, 4)]
etaf = [(1.6, 1.8), (1.6, 1.8), (1.6, 1.8), (2, 4), (2, 4), (2, 4)]
'''boundary conditions'''
boundary = ((0.2, 0.1, 0, 0, 0),(5, 3., 20, 10, 10))
# initial = (0.9, 0.65, 0.83, 0.5)
'''initial parameters'''
# initial = (0.82, 0.57, .36, .6)
# initial = (0.72, 0.3, 0.18, 1.6, 0.5)
# initial = (1.11919, 0.98740754, 1.7766, .1051285459, 0.0)
initial = (1., 0.5, 2, 3, 0)
# initial = (1.08674554e+00, 1.16530031e+00, 1.74236505e+00, 7.99187696e-09, 3.67480403e-18)

'''입력 데이터를 변경할 때에 꼭 제대로 들어가는지 확인하자'''
# phi_13TeV_ptdep_fitting = phi_13TeV_ptdep[0:-2]     # fitting에 사용하는 데이터에서 Yridge를 제외하고 fitting 하려는 경우
# dat_13TeV_ptdep_fitting = dat_13TeV_ptdep[0:-2]     # fitting에 사용하는 데이터에서 Yridge를 제외하고 fitting 하려는 경우

'''CMS Yridge만 제거 (아래 6줄)'''
# phi_13TeV_ptdep_fitting = phi_13TeV_ptdep[0:-1]     # fitting에 사용하는 데이터에서 CMS Yridge와 0.1<pT<1을 제외하고 fitting 하려는 경우
# dat_13TeV_ptdep_fitting = dat_13TeV_ptdep[0:-1]     # fitting에 사용하는 데이터에서 CMS Yridge와 0.1<pT<1을 제외하고 fitting 하려는 경우
# del phi_13TeV_ptdep_fitting[3]                      # fitting에 사용하는 데이터에서 CMS Yridge와 0.1<pT<1을 제외하고 fitting 하려는 경우
# del dat_13TeV_ptdep_fitting[3]                      # fitting에 사용하는 데이터에서 CMS Yridge와 0.1<pT<1을 제외하고 fitting 하려는 경우
# del phi_13TeV_ptdep_fitting[6]                      # delete ATLAS phi
# del dat_13TeV_ptdep_fitting[6]                      # delete ATLAS data


'''
                        phiCorr + ALICE Yridge result
[   kick             Tem             xx              yy              zz      ]
[0.798958702    0.840820694       6.70967779      1.49370324    3.51939886e-12]
Error : 4.456e-05
'''

# phi_13TeV_ptdep_fitting = phi_13TeV_ptdep[0:-2]     # fitting에 사용하는 데이터에서 Yridge와 0.1<pT<1을 제외하고 fitting 하려는 경우
# dat_13TeV_ptdep_fitting = dat_13TeV_ptdep[0:-2]     # fitting에 사용하는 데이터에서 Yridge와 0.1<pT<1을 제외하고 fitting 하려는 경우
# del phi_13TeV_ptdep_fitting[3]                      # fitting에 사용하는 데이터에서 Yridge와 0.1<pT<1을 제외하고 fitting 하려는 경우
# del dat_13TeV_ptdep_fitting[3]                      # fitting에 사용하는 데이터에서 Yridge와 0.1<pT<1을 제외하고 fitting 하려는 경우
# del phi_13TeV_ptdep_fitting[6]                      # delete ATLAS phi
# del dat_13TeV_ptdep_fitting[6]                      # delete ATLAS data

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

# '''Fitting 13TeV data'''
# ptdep = classes.Fitting_gpu(13000, phi_13TeV_ptdep_fitting, dat_13TeV_ptdep_fitting, (ptloww, pthigh), None, ptf, etaf, boundary, initial, "pTdependence")
# # ptdep_result, ptdep_error = ptdep.fitting(fitting_error)         # error를 대입하려는 경우(absolute sigma)
# ptdep_result, ptdep_error = ptdep.fitting(None)                  # error를 고려하지 않으려는 경우
# # ptdep_result = [0.68893858, 0.58398214, 0.39126606, 1.31503716]

# print(ptdep_result)
# print(ptdep_error)

'''Fitting 7TeV data'''
ptf_07 = [(1, 2), (2, 3), (3, 4)]
# etaf_07 = [(2, 4.8), (2, 4.8), (2, 4.8)]
etaf_07 = [(2, 4), (2, 4), (2, 4)]

# ptdep = classes.Fitting_gpu(7000, phi_07TeV_ptdep_fitting, dat_07TeV_ptdep_fitting, (ptloww_07, pthigh_07), None, ptf_07, etaf, boundary, initial, "pTdependence")
# # ptdep_result, ptdep_error = ptdep.fitting(fitting_error)         # error를 대입하려는 경우(absolute sigma)
# ptdep_result_07, ptdep_error_07 = ptdep.fitting(None)                  # error를 고려하지 않으려는 경우
# # ptdep_result = [0.68893858, 0.58398214, 0.39126606, 1.31503716]

# print(ptdep_result_07)
# print(ptdep_error_07)

'''
FrNk의 크기(xx)에 따라 fitting이 전혀 다르게 되는 것 같은 느낌이 듦. 이 아래는 이를 체크해보기 위함.
[   kick             Tem             xx              yy              zz      ]
[0.8370979       1.00076857      10.             1.54066307      0.18506123]
[0.838259744     0.955162987     4.99997430      1.00930624      9.12474390e-07]
[0.929536242     1.21218695      2.99999981      0.152785945     1.77795470e-08]
[0.90065667      1.28945833      14.99687732     1.47631147      0.38479797]
'''

'''
    일단 결과는 다음과 같지만, alice의 pT distribution까지만 추가해서 fitting 해보자.
    phi correlation만 가지고 fitting한 결과는 다음과 같다.
[   kick             Tem             xx              yy              zz      ]
[0.91627167      1.37536975      16.517866       1.45838023      0.4270991 ]
'''

'''
                alice의 Yridge만 포함하여 fitting 해보자.

'''


'''multiplicity fitting'''
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



time_calculate = time.time()
print(f"calculate end : {time_calculate-time_start:.3f} sec")


Yridge_phi07 = phi_07TeV_ptdep[-1]
Yridge_dat07 = dat_07TeV_ptdep[-1]
Yridge_phi = phi_13TeV_ptdep[-2::]
Yridge_dat = dat_13TeV_ptdep[-2::]
phi_13TeV_ptdep = phi_13TeV_ptdep[0:-2]
dat_13TeV_ptdep = dat_13TeV_ptdep[0:-2]
# ptdep_result = [0.9, 0.65, 0.83, 0.5]
# ptdep_result = [ 0.76163513, 0.73795214, 1.35738805, 0.48186191 ]
# ptdep_result = [ 0.63868284, 0.62552238, 0.88055133, 0.93343803 ]
# ptdep_result = [ 0.72109582, 0.68919741, 1.25759108, 0.59493454 ]
# ptdep_result = [ 0.87561104, 0.61632276, 0.62806332, 0.47714599 ]
# ptdep_result = [ 0.81979403, 0.57121306, 0.36814736, 0.61201023 ]
# ptdep_result = [0.87561104, 0.61632276, 0.62806332, 0.47714599]
# ptdep_result = [0.66957013, 0.34329916, 0.12407301, 2.01835855]
# ptdep_result = [1.11868344, 0.987193973, 1.77968244, 0.107430581, 2.94262803e-15]
# ptdep_result = [1.11908047, 0.987320254, 1.77729279, 0.105646718, 1.18514214e-43]
# ptdep_result = [0.963541360, 0.928095736, 1.93925018, 6.29802478e-13, 6.00231971e-26]
# ptdep_result = [0.798958702, 0.840820694, 6.70967779, 1.49370324, 3.51939886e-12]
# ptdep_result = [1.08674554, 1.16530031, 1.74236505, 7.99187696e-09, 3.67480403e-18]
# ptdep_result = [1.16705241, 3.00000000, 11.2761222, 9.72160460e-14, 0.492948044]
# ptdep_result = [9.77048585e-01, 1.79374925e+00, 5.29529630e+00, 8.52747999e-35, 2.20904141e-01]
# ptdep_result_07 = [1.00266407e+00, 4.83737711e-01, 1.02320562e+01, 4.08927964e+00, 9.12644624e-10]
ptdep_result = [1.26263183e+00, 3.00000000e+00, 1.16320702e+01, 5.74133130e-16, 6.00599565e-01]
# ptdep_result_07 = [1.70772461, 1.4136777,  9.6661728, 0.83413059, 1.01163764]  # Without czyam
ptdep_result_07 = [1.68531124, 1.43640594, 7.85469003, 0.80428453, 0.88283783]

fig1, axes1 = plt.subplots(nrows=1, ncols=5,figsize=(125,20))
#그래프 그리기
alice = classes.Drawing_Graphs(13000, (1.6, 1.8), *ptdep_result, None, None)
cms = classes.Drawing_Graphs(13000, (2, 4), *ptdep_result, None, None)
cms_07 = classes.Drawing_Graphs(7000, (2, 4), *ptdep_result_07, None, None)
atlas = classes.Drawing_Graphs(13000, (2, 5), *ptdep_result, None, None)
# print(dat_13TeV_ptdep)
# print(err_13TeV_ptdep)
for i in range(5):
    if i==0:
        ptf = (0.1, 1)
        '''cms plot 13TeV'''
        cms_result = cms.result_plot("pTdependence", None, ptf, (min(phi_13TeV_ptdep[i+3]), max(phi_13TeV_ptdep[i+3])))
        axes1[i].plot(cms_result[0], cms_result[1], color = "black", linewidth=7, linestyle='-')
        # axes1[i].errorbar(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3]-min(dat_13TeV_ptdep[i+3]), yerr=(abs(err_13TeV_ptdep[2*(i+3)+1]),err_13TeV_ptdep[2*(i+3)]), color="black", linestyle=' ', linewidth=7, capthick=3, capsize=15)
        # axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3]-min(dat_13TeV_ptdep[i+3]), edgecolors="black", s=800, marker='o', facecolors='none', linewidths=7)
        # axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3]-min(dat_13TeV_ptdep[i+3]), s=800, marker='+', facecolors='black', linewidths=7)
        axes1[i].errorbar(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3], yerr=(abs(err_13TeV_ptdep[2*(i+3)+1]),err_13TeV_ptdep[2*(i+3)]), color="black", linestyle=' ', linewidth=7, capthick=3, capsize=15)
        axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3], edgecolors="black", s=800, marker='o', facecolors='none', linewidths=7)
        axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3], s=800, marker='+', facecolors='black', linewidths=7)
        '''cms plot 7TeV'''
        cms_07result = cms_07.result_plot("pTdependence", None, ptf, (min(phi_07TeV_ptdep[i]), max(phi_07TeV_ptdep[i])))
        axes1[i].plot(cms_07result[0], cms_07result[1], color = "grey", linewidth=7, linestyle='-')
        # axes1[i].errorbar(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i]-min(dat_07TeV_ptdep[i]), yerr=(abs(err_07TeV_ptdep[2*i+1]),err_07TeV_ptdep[2*i]), color="grey", linestyle=' ', linewidth=7, capthick=3, capsize=15)
        # axes1[i].scatter(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i]-min(dat_07TeV_ptdep[i]), edgecolors="grey", s=800, marker='o', facecolors='none', linewidths=7)
        # axes1[i].scatter(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i]-min(dat_07TeV_ptdep[i]), s=800, marker='+', facecolors='grey', linewidths=7)
        axes1[i].errorbar(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i], yerr=(abs(err_07TeV_ptdep[2*i+1]),err_07TeV_ptdep[2*i]), color="grey", linestyle=' ', linewidth=7, capthick=3, capsize=15)
        axes1[i].scatter(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i], edgecolors="grey", s=800, marker='o', facecolors='none', linewidths=7)
        axes1[i].scatter(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i], s=800, marker='+', facecolors='grey', linewidths=7)
        # axes1[i].set_title(str(st)+r'$<p_{T, \, \mathrm{trig(assoc)}}<$'+str(en), size = 70, pad=30)
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
        # print(i)
        # print(dat_13TeV_ptdep[i-1])
        # print(err_13TeV_ptdep[2*i-1])
        # axes1[i].errorbar(phi_13TeV_ptdep[i-1], dat_13TeV_ptdep[i-1]-min(dat_13TeV_ptdep[i-1]), yerr=(abs(err_13TeV_ptdep[2*i-1]),err_13TeV_ptdep[2*i-2]), color="red", linestyle=' ', linewidth=7, capthick=3, capsize=15)
        # axes1[i].scatter(phi_13TeV_ptdep[i-1], dat_13TeV_ptdep[i-1]-min(dat_13TeV_ptdep[i-1]), edgecolors="red", s=800, marker='o', facecolors='none', linewidths=7)
        # axes1[i].scatter(phi_13TeV_ptdep[i-1], dat_13TeV_ptdep[i-1]-min(dat_13TeV_ptdep[i-1]), s=800, marker='+', facecolors='red', linewidths=7)
        axes1[i].errorbar(phi_13TeV_ptdep[i-1], dat_13TeV_ptdep[i-1], yerr=(abs(err_13TeV_ptdep[2*i-1]),err_13TeV_ptdep[2*i-2]), color="red", linestyle=' ', linewidth=7, capthick=3, capsize=15)
        axes1[i].scatter(phi_13TeV_ptdep[i-1], dat_13TeV_ptdep[i-1], edgecolors="red", s=800, marker='o', facecolors='none', linewidths=7)
        axes1[i].scatter(phi_13TeV_ptdep[i-1], dat_13TeV_ptdep[i-1], s=800, marker='+', facecolors='red', linewidths=7)
        '''cms plot 13TeV'''
        # axes1[i].errorbar(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3]-min(dat_13TeV_ptdep[i+3]), yerr=(abs(err_13TeV_ptdep[2*(i+3)+1]),err_13TeV_ptdep[2*(i+3)]), color="black", linestyle=' ', linewidth=7, capthick=3, capsize=15)
        # axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3]-min(dat_13TeV_ptdep[i+3]), edgecolors="black", s=800, marker='o', facecolors='none', linewidths=7)
        # axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3]-min(dat_13TeV_ptdep[i+3]), s=800, marker='+', facecolors='black', linewidths=7)
        axes1[i].errorbar(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3], yerr=(abs(err_13TeV_ptdep[2*(i+3)+1]),err_13TeV_ptdep[2*(i+3)]), color="black", linestyle=' ', linewidth=7, capthick=3, capsize=15)
        axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3], edgecolors="black", s=800, marker='o', facecolors='none', linewidths=7)
        axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3], s=800, marker='+', facecolors='black', linewidths=7)
        '''cms plot 7TeV'''
        cms_07result = cms_07.result_plot("pTdependence", None, ptf, (min(phi_07TeV_ptdep_fitting[i-1]), max(phi_07TeV_ptdep_fitting[i-1])))
        axes1[i].plot(cms_07result[0], cms_07result[1], color = "grey", linewidth=7, linestyle='-')
        # axes1[i].errorbar(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i]-min(dat_07TeV_ptdep[i]), yerr=(abs(err_07TeV_ptdep[2*i+1]),err_07TeV_ptdep[2*i]), color="grey", linestyle=' ', linewidth=7, capthick=3, capsize=15)
        # axes1[i].scatter(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i]-min(dat_07TeV_ptdep[i]), edgecolors="grey", s=800, marker='o', facecolors='none', linewidths=7)
        # axes1[i].scatter(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i]-min(dat_07TeV_ptdep[i]), s=800, marker='+', facecolors='grey', linewidths=7)
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

time_phicorr = time.time()
print(f"Graph, Phi correlation end : {time_phicorr-time_calculate:.3f} sec")

fig2, axis2 = plt.subplots(nrows=1, ncols=1,figsize=(40,20))

'''Scatter'''
# alice_Yridge = alice.Yridge((ptloww_ali, pthigh_ali), "Subtract")
# cms_Yridge = cms.Yridge((ptloww_cms, pthigh_cms), "Subtract")
# axis2.scatter(alice_Yridge[0], alice_Yridge[1], edgecolor = "red", marker='s', facecolors='none', s=800, linewidth=5, zorder=1)
# axis2.scatter(alice_Yridge[0], alice_Yridge[1], color = "red", marker='+', s=800, linewidth=5, zorder=1)
# axis2.scatter(cms_Yridge[0], cms_Yridge[1], edgecolor = "black", marker='s', facecolors='none', s=800, linewidth=5, zorder=1)
# axis2.scatter(cms_Yridge[0], cms_Yridge[1], color = "black", marker='+', s=800, linewidth=5, zorder=1)

'''Line'''
alice_Yridge = alice.Yridge_line("Subtract")
cms_Yridge = cms.Yridge_line("Subtract")
cms_07Yridge = cms_07.Yridge_line("Subtract")
axis2.plot(alice_Yridge[0], alice_Yridge[1], color = "red", linewidth=7, linestyle='-')
axis2.plot(cms_Yridge[0], cms_Yridge[1], color = "black", linewidth=7, linestyle='-')
axis2.plot(cms_07Yridge[0], cms_07Yridge[1], color = "grey", linewidth=7, linestyle='-')


# print(abs(err_13TeV_ptdep[15]),err_13TeV_ptdep[14])
# print(Yridge_dat)
axis2.errorbar(Yridge_phi[0], Yridge_dat[0], yerr=(abs(err_13TeV_ptdep[15]),err_13TeV_ptdep[14]), color="red", linestyle=' ', linewidth=5, capthick=3, capsize=15, zorder=0)
axis2.errorbar(Yridge_phi[1], Yridge_dat[1], yerr=(abs(err_13TeV_ptdep[17]),err_13TeV_ptdep[16]), color="black", linestyle=' ', linewidth=5, capthick=3, capsize=15, zorder=0)
axis2.scatter(Yridge_phi[0], Yridge_dat[0], edgecolors="red", s=800, marker='o', facecolors='none', linewidths=5, zorder=0)
axis2.scatter(Yridge_phi[0], Yridge_dat[0], s=800, marker='+', facecolors='red', linewidths=5, zorder=0)
axis2.scatter(Yridge_phi[1], Yridge_dat[1], edgecolors="black", s=800, marker='o', facecolors='none', linewidths=5, zorder=0)
axis2.scatter(Yridge_phi[1], Yridge_dat[1], s=800, marker='+', facecolors='black', linewidths=5, zorder=0)
# print(Yridge_phi07, Yridge_dat07, err_07TeV_ptdep[-2], err_07TeV_ptdep[-1])
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

time_yridge = time.time()
print(f"Graph, Yridge end : {time_yridge-time_phicorr:.3f} sec")

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


time_end = time.time()
print(f"FrNk end : {time_end-time_yridge:.3f} sec")
print(f"Total end : {time_end-time_start:.3f} sec")