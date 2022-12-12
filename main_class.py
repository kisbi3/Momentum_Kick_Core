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

path = ['./data/atldata/13TeV/', './data/alidata/', './data/cmsdata/', './data/atldata/2.76TeV/', './data/atldata/5.02TeV/', './data/atldata/13TeV_distribution/', './data/atldata/']


'''Multiplicity Dependence'''
phi_13TeV_multi_atlas = []
phi_13TeV_multi_atlas_fitting = []
dat_13TeV_multi_atlas = []
dat_13TeV_multi_atlas_fitting = []
multiplicity_atlas = []
phi_13TeV_multi_cms = []
phi_13TeV_multi_cms_fitting = []
dat_13TeV_multi_cms = []
dat_13TeV_multi_cms_fitting = []
err_13TeV_multi_cms = []
'''순서 : multiplicity, mean pT'''
meanpTvsnch_13TeV=[]
'''순서 : err+, err-'''
meanpTvsnch_13TeV_err=[]
'''순서 : Nch Low, Nch High'''
meanpTvsnch_13TeV_nchrange=[]
meanpTvsnch_13TeV.append(np.loadtxt(path[5]+'Table6.csv', delimiter=',', usecols=[0], skiprows=14))
meanpTvsnch_13TeV.append(np.loadtxt(path[5]+'Table6.csv', delimiter=',', usecols=[3], skiprows=14))
temp_sta1 = np.loadtxt(path[5]+'Table6.csv', delimiter=',', usecols=[4], skiprows=14)
temp_sta2 = np.loadtxt(path[5]+'Table6.csv', delimiter=',', usecols=[5], skiprows=14)
temp_sys1 = np.loadtxt(path[5]+'Table6.csv', delimiter=',', usecols=[6], skiprows=14)
temp_sys2 = np.loadtxt(path[5]+'Table6.csv', delimiter=',', usecols=[7], skiprows=14)
meanpTvsnch_13TeV_err.append((temp_sta1**2+temp_sys1**2)**0.5)
meanpTvsnch_13TeV_err.append((temp_sta2**2+temp_sys2**2)**0.5)
meanpTvsnch_13TeV_nchrange.append(np.loadtxt(path[5]+'Table6.csv', delimiter=',', usecols=[1], skiprows=14))
meanpTvsnch_13TeV_nchrange.append(np.loadtxt(path[5]+'Table6.csv', delimiter=',', usecols=[2], skiprows=14))
atlaserror = []     # Only 120<mutli<130, 130<multi


'''Multiplicity에 대한 associated yield
   순서 : ALTAS, CMS                   '''
Multi_N = []
Multi_Y = []
Multi_N.append(np.loadtxt(path[6]+'Multiplicity_yield.csv',delimiter=',',usecols=[0], skiprows=1))
Multi_Y.append(np.loadtxt(path[6]+'Multiplicity_yield.csv',delimiter=',',usecols=[1], skiprows=1))
Multi_N.append(np.loadtxt(path[2]+'Table35.csv',delimiter=',',usecols=[0],skiprows=14))
Multi_Y.append(np.loadtxt(path[2]+'Table35.csv',delimiter=',',usecols=[1],skiprows=14))
'''append atlas multiplicity data'''
def atlas_multiplicity():
    for i in range(9):
        start = 10*i + 50
        end = 10*i + 60
        if start == 130:
            phi_13TeV_multi_atlas.append(np.loadtxt(path[0]+f'{start}~.csv',delimiter=',',usecols=[0], skiprows=3, max_rows=12))
            dat_13TeV_multi_atlas.append(np.loadtxt(path[0]+f'{start}~.csv',delimiter=',',usecols=[1], skiprows=3, max_rows=12))
            '''multiplicity가 130이상이기 때문에 평균값보다 더 커서 변경할 수도 있어서 이렇게 두었음'''
            multiplicity_atlas.append((start+140)/2)
        else:
            multiplicity_atlas.append((start+end)/2)
            phi_13TeV_multi_atlas.append(np.loadtxt(path[0]+f'{start}~{end}.csv',delimiter=',',usecols=[0], skiprows=3, max_rows=12))
            dat_13TeV_multi_atlas.append(np.loadtxt(path[0]+f'{start}~{end}.csv',delimiter=',',usecols=[1], skiprows=3, max_rows=12))
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
    atlaserror.append(np.loadtxt(path[0]+f'120~130.csv',delimiter=',',usecols=[2], skiprows=3, max_rows=12))
    atlaserror.append(np.loadtxt(path[0]+f'120~130.csv',delimiter=',',usecols=[3], skiprows=3, max_rows=12))
    atlaserror.append(np.loadtxt(path[0]+f'130~.csv',delimiter=',',usecols=[2], skiprows=3, max_rows=12))
    atlaserror.append(np.loadtxt(path[0]+f'130~.csv',delimiter=',',usecols=[3], skiprows=3, max_rows=12))
atlas_multiplicity()
'''append cms multiplicity data'''
'''아래 multiplicity는 사용할지 사용 안할지 모름.'''
multiplicity_cms = [22.5, 22.5, 22.5, 22.5, 57.5, 57.5, 57.5, 57.5, 92.5, 92.5, 92.5, 92.5, 127.5, 127.5, 127.5, 127.5]
def cms_multiplicity():
    for i in range(16):
        '''80<N<105, 3<pT<4 의 데이터가 많이 이상하여 if문으로 빼서 따로 처리해야 할 수도'''
        table = 2*i+1
        phi_13TeV_multi_cms.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[0], skiprows=18, max_rows=13))
        dat_13TeV_multi_cms.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[1], skiprows=18, max_rows=13))
        err_13TeV_multi_cms.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[2], skiprows=18, max_rows=13))
        err_13TeV_multi_cms.append(np.loadtxt(path[2]+f'Table{table}.csv',delimiter=',',usecols=[3], skiprows=18, max_rows=13))
        dat_13TeV_multi_cms[i] -= min(dat_13TeV_multi_cms[i])
        check1 = int(np.where(dat_13TeV_multi_cms[i]==0)[0][0])
        check2 = int(len(dat_13TeV_multi_cms[i])-check1)-1
        if check1<check2:
            phi_13TeV_multi_cms_fitting.append(phi_13TeV_multi_cms[i][check1:check2+1])
            dat_13TeV_multi_cms_fitting.append(dat_13TeV_multi_cms[i][check1:check2+1])
        elif check1>check2:
            phi_13TeV_multi_cms_fitting.append(phi_13TeV_multi_cms[i][check2:check1+1])
            dat_13TeV_multi_cms_fitting.append(dat_13TeV_multi_cms[i][check2:check1+1])
cms_multiplicity()


'''pt dependence'''
phi_13TeV_ptdep = []
dat_13TeV_ptdep = []
err_13TeV_ptdep = []
phi_07TeV_ptdep = []
dat_07TeV_ptdep = []
err_07TeV_ptdep = []
fitting_error = []
# 순서 : alice, alice, alice, ..., cms, cms, cms, ....., atlas, alice, cms
# 마지막 두개는 Yridge
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

# ptdep_result = []
# ptdep_result = [1.04435586e+00, 1.39798449e+00, 3.13646871e+00, 4.16055192e-05, 1.39096183e-01]

# ptdep_result = [0.9550869558401427, 1.0902337006471705, 2.7422234903350926, 0.35144493617188255, 9.865787114868575e-06]
# ptdep_result_07 = [1.624901759777593, 0.9302427715844408, 5.0117954889744185, 0.8297385370001579, 0.737376419008668]
ptdep_result = []
ptdep_result_07 = []
ptdep_result_cm = []
ptdep_error_cm = []
multi_atlas_result = []
multi_cms_result = []
ptdep_Rsq = []

total_boundary = ((0.5, .5, 0, 0, 0),(5, 4., 10, 10, 10))
total_initial = (1.,  1., 2., 0, 0)
# total_initial = (0.955, 1.09, 2.74)
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
    del phi_13TeV_ptdep_fitting[-1]
    del dat_13TeV_ptdep_fitting[-1]
    del ptloww[-1]
    del pthigh[-1]
    ptdep = classes.Fitting_gpu(13000, phi_13TeV_ptdep_fitting, dat_13TeV_ptdep_fitting, (ptloww, pthigh), None, ptf, etaf, boundary, initial, "pTdependence")
    result, ptdep_error = ptdep.fitting(None, None)                  # error를 고려하지 않으려는 경우
    print("pp 13TeV Fitting result : ", result)
    print("pp 13TeV error", ptdep_error)
    if len(result) == 4:
        result_temp = [result[0], 1.514, result[1], result[2], result[3]]
    elif len(result) == 5:
        result_temp = [result[0], result[1], result[2], result[3], result[4]]
    ptdep_result.extend(result_temp)
    # ptdep_result = [9.62260664e-01, 1.08168335e+00, 2.61768147e+00, 2.84798146e-01, 4.17974797e-04]

    '''ptf 개수가 결국 phi array 개수이다. 이렇게 해야 자동으로 Yridge를 제외하고 배열을 대입한다.'''
    for i in range(len(ptf)):
        # print(i, phi_13TeV_ptdep_fitting[i], dat_13TeV_ptdep_fitting[i], ptf[i], etaf[i])
        ptdep_error = classes.Error(13000, phi_13TeV_ptdep_fitting[i], dat_13TeV_ptdep_fitting[i], ptf[i], etaf[i])
        # Error_Rsq = ptdep_error.R_squared("pTdependence", *ptdep_result)
        # ptdep_Rsq.append(Error_Rsq)
    print(ptdep_Rsq)

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
    result, ptdep_error_07 = ptdep.fitting(None, None)                 # error를 고려하지 않으려는 경우
    print(result)
    print(ptdep_error_07)
    # ptdep_result_07.append(result)
    ptdep_result_07.extend(result)

'''multiplicity fitting'''
'''multiplicity파일 안에 에서는 multiplicity에 따른 associated yield 그래프를 이용해서 fitting한 후에 delta phi correlation에 적용만 하는 상태.'''
'''이 파일에서는 multiplicity에 따른 mean pT를 확인하여 이에 따른 T를 계산하고, fitting할 것이다.'''
def fit_multipl():
    global ptdep_result
    # boundary = (0,20)       # fitting 개수 1개인 경우
    # initial = (1)           # fitting 개수 1개인 경우
    boundary = ((0.1, 0), (5, 100))
    initial = (0.5, 5)
    print("High multiplicity results : ", ptdep_result)
    highmulti_Temp = ptdep_result[1]        # 만약, 13TeV fitting을 안돌릴 경우 여기에 상수를 대입해야 함.
    Fixed_Temperature = classes.Fitting_gpu.Fixed_Temp(meanpTvsnch_13TeV[0], meanpTvsnch_13TeV[1], highmulti_Temp)
    Fixed_Temperature_fitting = []
    Fixed_Temperature_fitting.extend([Fixed_Temperature[54], Fixed_Temperature[62], Fixed_Temperature[67], Fixed_Temperature[72]])
    Fixed_Temperature_fitting.append((Fixed_Temperature[75] + Fixed_Temperature[76])/2)
    Fixed_Temperature_fitting.extend([Fixed_Temperature[77], Fixed_Temperature[78], Fixed_Temperature[79], Fixed_Temperature[80]])
    '''multiplicity에 대한 associated yield만 가지고 fitting하고 phi correlation그리기'''
    multi_atlas = classes.Fitting_gpu(13000, phi_13TeV_multi_atlas_fitting, dat_13TeV_multi_atlas_fitting, None, multiplicity_atlas, (0.5, 5), (2, 5), boundary, initial, "Multiplicity")

    # You can choice fitting mode : "free kick", "free Tem", "free fRNk xx", "Free kick, fRNk xx_FixedTem", "Free kick, fRNk xx"
    multiplicity_fittingmode = "Free kick, fRNk xx"
    # multi_atlas.multiplicity_fitting_mode(multiplicity_fittingmode)                                                                        # Temperature를 pT mean으로 결정하는 경우
    # result, multi_atlas_error = multi_atlas.fitting(None, (Fixed_Temperature_fitting, ptdep_result[2:]))                  # Temperature를 pT mean으로 결정하는 경우

    print("Temperature", Fixed_Temperature_fitting)
    # fitting = []
    # fitting.append(ptdep_result[0])
    # fitting.append(list(Fixed_Temperature_fitting))
    # fitting.extend([ptdep_result[2], ptdep_result[3], ptdep_result[4]])
    fitting = [ptdep_result[0], Fixed_Temperature_fitting, ptdep_result[2], ptdep_result[3], ptdep_result[4]]
    multi_atlas.multiplicity_fitting_mode(multiplicity_fittingmode)                                   # 각 파라미터들을 고정시켜가며 어떤게 가장 dominant한지 확인하는 작업
    # result, multi_atlas_error = multi_atlas.fitting(None, ptdep_result)                  # 각 파라미터들을 고정시켜가며 어떤게 가장 dominant한지 확인하는 작업
    result, multi_atlas_error = multi_atlas.fitting(None, fitting)                          # fitting Mode가 Free kick, fRNk xx인 경우에 사용


    # kick = 0.798958702
    # Tem = 0.840820694
    # yy = 1.49370324
    # zz = 3.51939886e-12
    # multi_atlas_result = np.array([kick, Tem, multi_atlas_result[0], yy, zz, multi_atlas_result[1], multi_atlas_result[2]])
    '''multi_atlas_result : Kick, xx, yy, zz'''
    # print('ATLAS results : ', result)

    print('ATLAS results :')
    for i in range(len(result)):
        #            q                      T                   xx              yy          zz
        if (multiplicity_fittingmode == "Nothing"):
            temp = [result[i][0], Fixed_Temperature_fitting[i], ptdep_result[2], ptdep_result[3], ptdep_result[4]]              # Temperature를 pT mean으로 결정하는 경우

        # 각 파라미터들을 고정시켜가며 어떤게 가장 dominant한지 확인하는 작업
        elif (multiplicity_fittingmode == "Free kick"):
            temp = [result[i][0], ptdep_result[1], ptdep_result[2], ptdep_result[3], ptdep_result[4]]
        elif (multiplicity_fittingmode == "Free Tem"):
            temp = [ptdep_result[0], result[i][0], ptdep_result[2], ptdep_result[3], ptdep_result[4]]
        elif (multiplicity_fittingmode == "Free fRNk xx"):
            temp = [ptdep_result[0], ptdep_result[1], result[i][0], ptdep_result[3], ptdep_result[4]]
        elif (multiplicity_fittingmode == "Free kick, fRNk xx_FixedTem"):
            temp = [result[i][0], ptdep_result[1], result[i][1], ptdep_result[3], ptdep_result[4]]
        elif (multiplicity_fittingmode == "Free kick, fRNk xx"):
            temp = [result[i][0], Fixed_Temperature_fitting[i], result[i][1], ptdep_result[3], ptdep_result[4]]
        print(temp)
        multi_atlas_result.append(temp)
    # print('ATLAS results : ', multi_atlas_result)
    print('ATLAS error : ', multi_atlas_error)
    '''
            ***Results : [0.798958702    0.840820694    5.38517847    1.49370324    3.51939886e-12    19.4957273e    117.807870]***
    '''
    
'''
    multi_cms = classes.Fitting_gpu(13000, phi_13TeV_multi_cms_fitting, dat_13TeV_multi_cms_fitting, None, multiplicity_cms, (0.5, 5), (2, 4), boundary, initial, "Multiplicity")
    fitting = [ptdep_result[0], Fixed_Temperature_fitting, ptdep_result[2], ptdep_result[3], ptdep_result[4]]
    multi_cms.multiplicity_fitting_mode(multiplicity_fittingmode)                                   # 각 파라미터들을 고정시켜가며 어떤게 가장 dominant한지 확인하는 작업
    # result, multi_cms_error = multi_atlas.fitting(None, ptdep_result)                  # 각 파라미터들을 고정시켜가며 어떤게 가장 dominant한지 확인하는 작업
    result, multi_cms_error = multi_cms.fitting(None, fitting)                          # fitting Mode가 Free kick, fRNk xx인 경우에 사용
    print('CMS results :')
    for i in range(len(result)):
        #            q                      T                   xx              yy          zz
        if (multiplicity_fittingmode == "Nothing"):
            temp = [result[i][0], Fixed_Temperature_fitting[i], ptdep_result[2], ptdep_result[3], ptdep_result[4]]              # Temperature를 pT mean으로 결정하는 경우

        # 각 파라미터들을 고정시켜가며 어떤게 가장 dominant한지 확인하는 작업
        elif (multiplicity_fittingmode == "Free kick"):
            temp = [result[i][0], ptdep_result[1], ptdep_result[2], ptdep_result[3], ptdep_result[4]]
        elif (multiplicity_fittingmode == "Free Tem"):
            temp = [ptdep_result[0], result[i][0], ptdep_result[2], ptdep_result[3], ptdep_result[4]]
        elif (multiplicity_fittingmode == "Free fRNk xx"):
            temp = [ptdep_result[0], ptdep_result[1], result[i][0], ptdep_result[3], ptdep_result[4]]
        elif (multiplicity_fittingmode == "Free kick, fRNk xx_FixedTem"):
            temp = [result[i][0], ptdep_result[1], result[i][1], ptdep_result[3], ptdep_result[4]]
        elif (multiplicity_fittingmode == "Free kick, fRNk xx"):
            temp = [result[i][0], Fixed_Temperature_fitting[i], result[i][1], ptdep_result[3], ptdep_result[4]]
        print(temp)
        multi_cms_result.append(temp)
    # print('ATLAS results : ', multi_atlas_result)
    print('CMS error : ', multi_cms_error)
'''

'''To Check Center of mass Energy dependence'''
''' Multiplicity : 90~100'''
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
        result, error = ptdep.fitting(None, None)
        print(result)
        ptdep_result_cm.append(result)
        ptdep_error_cm.append(error)
        print(f"\n**********\n sqrSnn = {sqrSnn_str[i]} End \n\n**********\n")
    print(ptdep_result_cm)
    print(ptdep_error_cm)


# fit_13tev()
# fit_7tev()
# fit_multipl()
# fit_cmenerg()


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

# md=q 로 두고 fitting
# ptdep_result = [0.9550869558401427, 1.0902337006471705, 2.7422234903350926, 0.35144493617188255, 9.865787114868575e-06]
# ptdep_result_07 = [1.624901759777593, 0.9302427715844408, 5.0117954889744185, 0.8297385370001579, 0.737376419008668]

# md=1 로 두고 fitting
# ptdep_result = [0.9619064848652367, 1.0806152707987788, 2.6181236203620877, 0.2866504816022142, 4.83643115879628e-06]
# ptdep_result_07 = [1.6594350487174188, 1.254920842633187, 6.8002216962428035, 0.8677859196950383, 0.8214136167443336]

# Nk 제거, md=1 fitting
ptdep_result = [9.61853703e-01, 1.08047173e+00, 2.61871339e+00, 2.87068889e-01, 1.00000000e-10]
ptdep_result_07 = [1.04583978e+00, 5.02452357e-01, 9.88785638e+00, 4.05148986e+00, 1.00000000e-10]

# fig1, axes1 = plt.subplots(nrows=1, ncols=5,figsize=(125,20))
#그래프 그리기
# alice = classes.Drawing_Graphs(13000, (1.6, 1.8), *ptdep_result, None, None)
# cms = classes.Drawing_Graphs(13000, (2, 4), *ptdep_result, None, None)
# cms_07 = classes.Drawing_Graphs(7000, (2, 4), *ptdep_result_07, None, None)
# atlas = classes.Drawing_Graphs(13000, (2, 5), *ptdep_result, None, None)

'''pT dependence phi correlation graph'''
def drawgraph_ptdep_phicorr():
    fig1, axes1 = plt.subplots(nrows=1, ncols=5,figsize=(125,20))
    print("13TeV : ", ptdep_result)
    print("7TeV : ", ptdep_result_07)
    alice = classes.Drawing_Graphs(13000, (1.6, 1.8), *ptdep_result, None, None, 'ALICE')
    cms = classes.Drawing_Graphs(13000, (2, 4), *ptdep_result, None, None, 'CMS')
    cms_07 = classes.Drawing_Graphs(7000, (2, 4), *ptdep_result_07, None, None, 'CMS')
    atlas = classes.Drawing_Graphs(13000, (2, 5), *ptdep_result, None, None, 'ATLAS')
    for i in range(5):
        if i==0:
            ptf = (0.1, 1)
            '''cms plot 13TeV'''
            cms_result = cms.result_plot("pTdependence", None, ptf, (min(phi_13TeV_ptdep[i+3]), max(phi_13TeV_ptdep[i+3])))
            axes1[i].plot(cms_result[0], cms_result[1], color = "black", linewidth=7, linestyle='-')
            axes1[i].errorbar(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3], yerr=(abs(err_13TeV_ptdep[2*(i+3)+1]),err_13TeV_ptdep[2*(i+3)]), color="black", linestyle=' ', linewidth=7, capthick=3, capsize=15)
            axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3], edgecolors="black", s=800, marker='o', facecolors='none', linewidths=7)
            axes1[i].scatter(phi_13TeV_ptdep[i+3], dat_13TeV_ptdep[i+3], s=800, marker='+', facecolors='black', linewidths=7)
            # '''cms plot 7TeV'''
            # cms_07result = cms_07.result_plot("pTdependence", None, ptf, (min(phi_07TeV_ptdep[i]), max(phi_07TeV_ptdep[i])))
            # axes1[i].plot(cms_07result[0], cms_07result[1], color = "grey", linewidth=7, linestyle='-')
            # axes1[i].errorbar(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i], yerr=(abs(err_07TeV_ptdep[2*i+1]),err_07TeV_ptdep[2*i]), color="grey", linestyle=' ', linewidth=7, capthick=3, capsize=15)
            # axes1[i].scatter(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i], edgecolors="grey", s=800, marker='o', facecolors='none', linewidths=7)
            # axes1[i].scatter(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i], s=800, marker='+', facecolors='grey', linewidths=7)
            # axes1[i].set_title(r'$0.1<p_{T, \, \mathrm{trig(assoc)}}<1$', size = 70, pad=30)
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
            # '''cms plot 7TeV'''
            # cms_07result = cms_07.result_plot("pTdependence", None, ptf, (min(phi_07TeV_ptdep_fitting[i-1]), max(phi_07TeV_ptdep_fitting[i-1])))
            # axes1[i].plot(cms_07result[0], cms_07result[1], color = "grey", linewidth=7, linestyle='-')
            # axes1[i].errorbar(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i], yerr=(abs(err_07TeV_ptdep[2*i+1]),err_07TeV_ptdep[2*i]), color="grey", linestyle=' ', linewidth=7, capthick=3, capsize=15)
            # axes1[i].scatter(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i], edgecolors="grey", s=800, marker='o', facecolors='none', linewidths=7)
            # axes1[i].scatter(phi_07TeV_ptdep[i], dat_07TeV_ptdep[i], s=800, marker='+', facecolors='grey', linewidths=7)
            st = i
            en = i+1
            # axes1[i].set_title(str(st)+r'$<p_{T, \, \mathrm{trig(assoc)}}<$'+str(en), size = 70, pad=30)
            axes1[i].set_title(str(st)+r'$<p_{T, \, \mathrm{trig(assoc)}}<$'+str(en), size = 70, pad=30)
        # axes1[1].text(-1.18, 0.0165, fr"ALICE R-squared : {round(ptdep_Rsq[0], 3)}", size = 60)
        # axes1[1].text(-1.18, 0.0155, fr" \ CMS \ R-squared : {round(ptdep_Rsq[3], 3)}", size = 60)
        # axes1[2].text(-1.18, 0.0089, fr"ALICE R-squared : {round(ptdep_Rsq[1], 3)}", size = 60)
        # axes1[2].text(-1.18, 0.0083, fr" \ CMS \ R-squared : {round(ptdep_Rsq[4], 3)}", size = 60)
        # axes1[3].text(-1.18, 0.0042, fr"ALICE R-squared : {round(ptdep_Rsq[2], 3)}", size = 60)
        # axes1[3].text(-1.18, 0.0039, fr"\ CMS \ R-squared : {round(ptdep_Rsq[5], 3)}", size = 60)

        axes1[i].set_xlabel(r'$\Delta\phi$', size=70)
        axes1[i].minorticks_on()
        axes1[i].tick_params(axis='both',which='major',direction='in',width=2,length=30,labelsize=45, top = 'true', right='true')
        axes1[i].tick_params(axis='both',which='minor',direction='in',width=2,length=15,labelsize=45, top = 'true', right='true')
        axes1[i].grid(color='silver',linestyle=':',linewidth=3)

    axes1[0].set_ylabel(r'$\frac{1}{N_{\mathrm{trig}}}\frac{dN^{\mathrm{pair}}}{d\Delta\phi}-C_{\mathrm{ZYAM}}$', size=70)
    fig1.tight_layout(h_pad=-1)

    fig1.savefig('./Results/phiCorr_Test.png')
    fig1.savefig('/home/jaesung/Dropbox/ohno/phiCorr_Test.png')

'''pT dependence Y^ridge graph'''
def drawgraph_ptdep_Yridge():
    fig2, axis2 = plt.subplots(nrows=1, ncols=1,figsize=(40,20))
    alice = classes.Drawing_Graphs(13000, (1.6, 1.8), *ptdep_result, None, None, 'ALICE')
    cms = classes.Drawing_Graphs(13000, (2, 4), *ptdep_result, None, None, 'CMS')
    cms_07 = classes.Drawing_Graphs(7000, (2, 4), *ptdep_result_07, None, None, 'CMS')
    atlas = classes.Drawing_Graphs(13000, (2, 5), *ptdep_result, None, None, 'ATLAS')

    '''Line'''
    alice_Yridge = alice.Yridge_line("Subtract")
    cms_Yridge = cms.Yridge_line("Subtract")
    cms_07Yridge = cms_07.Yridge_line("Subtract")
    atlas_Yridge = atlas.Yridge_line("Subtract")    # 그냥 atlas도 한번 그려보자.
    axis2.plot(alice_Yridge[0], alice_Yridge[1], color = "red", linewidth=7, linestyle='-')
    axis2.plot(cms_Yridge[0], cms_Yridge[1], color = "black", linewidth=7, linestyle='-')
    axis2.plot(atlas_Yridge[0], atlas_Yridge[1], color = "blue", linewidth=7, linestyle='-')
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
    fig2.savefig('./Results/Yridge_Test.png')
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
    pp_07TeV_Wong = [1.415411195083602, 0, 0]
    '''Hanul's pp 13TeV FrNk result plot [[pt(Average)], [FrNk]]'''
    Hanul_FrNk = [[1.5, 2.5], [0.93, 1.37]]
    Hanul_FrNk_error = [[0.5, 0.5], [0.5, 0.5]]
    ptf = np.arange(0.01,4,0.01)

    plt.plot(ptf, FrNk_func(ptf, *AuAu_200GeV), color = 'red', linewidth=7, label=r'$AuAu, \, 200\mathrm{GeV}$')
    plt.plot(ptf, FrNk_func(ptf, *PbPb_276TeV), color = 'black', linewidth=7, label=r'$PbPb, \, 2.76\mathrm{TeV}$')
    plt.plot(ptf, FrNk_func(ptf, *pp_07TeV_Wong), color = 'blue', linewidth=7, label=r'$pp, \, 7\mathrm{TeV}$')
    plt.plot(ptf, FrNk_func(ptf, *pp_13TeV), color = 'purple', linewidth=7, label=r'$pp, \, 13\mathrm{TeV}$')
    # plt.plot(ptf, FrNk_func(ptf, *pp_07TeV), color = 'grey', linewidth=7, label=r'$pp, \, 07\mathrm{TeV}$')
    plt.scatter(Hanul_FrNk[0], Hanul_FrNk[1], edgecolor = 'green', facecolors='none', s=900, marker='o', linewidths=5, zorder=2, label=r'$pp, \, 13\mathrm{TeV}$')
    plt.scatter(Hanul_FrNk[0], Hanul_FrNk[1], facecolors='green', s=900, marker='+', linewidths=5, zorder=2)
    plt.errorbar(Hanul_FrNk[0], Hanul_FrNk[1], xerr=Hanul_FrNk_error, color="green", linestyle=' ', linewidth=7, capthick=3, capsize=15)

    plt.xlabel(r'$p_T^{\mathrm{trig}},\, p_T^{\mathrm{trig}(\mathrm{assoc})}$',size=70)
    plt.ylabel(r'$f_{R} \langle N_k \rangle $',size=70)
    plt.xlim(0,4)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '${:g}$'.format(y)))
    plt.tick_params(axis='both',which='major',direction='in',width=2,length=30,labelsize=45, top='true')
    plt.tick_params(axis='both',which='minor',direction='in',width=2,length=15,labelsize=45, top='true')

    plt.grid(color='silver',linestyle=':',linewidth=5, zorder=0)
    plt.legend(fontsize=45, loc='upper left')

    plt.tight_layout()

    fig3.savefig('./Results/FrNk_Test.png')
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

    fig3.savefig('./Results/cmdep_phicorr.png')
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

    fig3.savefig('./Results/CMenerg_FrNk.png')
    fig3.savefig('/home/jaesung/Dropbox/ohno/CMenerg_FrNk.png')

def drawgraph_multi_phicorr():
    fig1, axes1 = plt.subplots(nrows=3, ncols=3,figsize=(90,90),sharey='row', sharex='col')
    #그래프 그리기
    # alice = classes.Drawing_Graphs((1.6, 1.8), *ptdep_alice_result, None, None)
    # cms = classes.Drawing_Graphs((2, 4), *multi_cms_result)
    # atlas = classes.Drawing_Graphs((2, 5), *multi_atlas_result, None, None)
    for j in range(3):
        axes1[j][0].set_ylabel(r'$\frac{1}{N_{\mathrm{trig}}}\frac{dN^{\mathrm{pair}}}{d\Delta\phi}-C_{\mathrm{ZYAM}}$', size = 150)
        for i in range(3):
            atlas = classes.Drawing_Graphs(13000, (2, 5), *multi_atlas_result[i+3*j], None, None, 'ATLAS')
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

    axes1[0][0].text(-0.9, 0.0145, r'$50 \leq N^{\mathrm{rec}}_{\mathrm{ch}} < 60$', size = 150)
    axes1[0][1].text(-0.9, 0.0145, r'$60 \leq N^{\mathrm{rec}}_{\mathrm{ch}} < 70$', size = 150)
    axes1[0][2].text(-0.9, 0.0145, r'$70 \leq N^{\mathrm{rec}}_{\mathrm{ch}} < 80$', size = 150)
    axes1[1][0].text(-0.9, 0.032, r'$80 \leq N^{\mathrm{rec}}_{\mathrm{ch}} < 90$', size = 150)
    axes1[1][1].text(-0.9, 0.032, r'$90 \leq N^{\mathrm{rec}}_{\mathrm{ch}} < 100$', size = 150)
    axes1[1][2].text(-0.9, 0.032, r'$100 \leq N^{\mathrm{rec}}_{\mathrm{ch}} < 110$', size = 150)
    axes1[2][0].text(-0.9, 0.0625, r'$110 \leq N^{\mathrm{rec}}_{\mathrm{ch}} < 120$', size = 150)
    axes1[2][1].text(-0.9, 0.0625, r'$120 \leq N^{\mathrm{rec}}_{\mathrm{ch}} < 130$', size = 150)
    axes1[2][2].text(-0.9, 0.0625, r'$130 \leq N^{\mathrm{rec}}_{\mathrm{ch}}$', size = 150)

    axes1[0][0].set_ylim(-0.001,0.0159)
    axes1[1][0].set_ylim(-0.001,0.0349)
    axes1[2][0].set_ylim(-0.001,0.0699)

    fig1.tight_layout(h_pad = -1)
    fig1.savefig('./Results/rezero_atlas_Nk.png')

    ''' 파라미터들 정리해서 그림 형태로 저장'''
    fig1 = plt.figure()
    ax1 = plt.axes()
    ax2 = ax1.twinx()
    fig1.set_size_inches(35, 16.534, forward=True)
    multiplicity_atlas = [55, 65, 75, 85, 95, 105, 115, 125, 135]
    multiplicity_range = [[5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5]]
    q = []; T = []
    for list in multi_atlas_result:
        q.append(list[0]);  T.append(list[1])

    lns1 = ax1.scatter(multiplicity_atlas, q, edgecolor = 'blue', facecolors='none', s=900, marker='o', linewidths=5, zorder=2, label=r'$q\quad (\mathrm{left \, axis})$')
    ax1.errorbar(multiplicity_atlas, q, xerr=multiplicity_range, color="blue", linestyle=' ', linewidth=7, capthick=3, capsize=15)
    ax1.scatter(multiplicity_atlas, q, facecolors='blue', s=900, marker='+', linewidths=5, zorder=2)
    lns2 = ax2.scatter(multiplicity_atlas, T, edgecolor = 'red', facecolors='none', s=900, marker='o', linewidths=5, zorder=2, label=r'$T\quad (\mathrm{right \, axis})$')
    ax2.scatter(multiplicity_atlas, T, facecolors='red', s=900, marker='+', linewidths=5, zorder=2)
    ax2.errorbar(multiplicity_atlas, T, xerr=multiplicity_range, color="red", linestyle=' ', linewidth=7, capthick=3, capsize=15)

    ax1.set_xlabel(r'$N^{\mathrm{rec}}_{\mathrm{ch}}$',size=70)
    ax1.set_xlim(50, 140)
    ax1.set_ylabel(r'$ q(\mathrm{GeV}) $',size=70)
    ax1.set_ylim(0.3, 1.8)
    ax2.set_ylabel(r'$ T(\mathrm{GeV}) $',size=70)
    # ax2.set_ylim(1.325, 1.7)      # T 기준을 AuAu 200GeV로 할 경우
    ax2.set_ylim(1.01, 1.16)        # T 기준을 pp 13TeV로 할 경우

    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '${:g}$'.format(y)))
    ax1.tick_params(axis='both',which='major',direction='in',width=2,length=30,labelsize=45, top='true')
    ax1.tick_params(axis='both',which='minor',direction='in',width=2,length=15,labelsize=45, top='true')
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '${:g}$'.format(y)))
    ax2.tick_params(axis='both',which='major',direction='in',width=2,length=30,labelsize=45, top='true')
    ax2.tick_params(axis='both',which='minor',direction='in',width=2,length=15,labelsize=45, top='true')

    ax1.grid(color='silver',linestyle=':',linewidth=5, zorder=0)

    lns = [lns1, lns2]
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, fontsize=50, loc='upper left')

    plt.tight_layout()
    fig1.savefig('./Results/parameters_multiplicity_dep_ATLAS.png')


'''
    fig1, axes1 = plt.subplots(nrows=4, ncols=4,figsize=(120,120),sharey='row', sharex='col')
    #그래프 그리기
    # alice = classes.Drawing_Graphs((1.6, 1.8), *ptdep_alice_result, None, None)
    # cms = classes.Drawing_Graphs((2, 4), *multi_cms_result)
    # atlas = classes.Drawing_Graphs((2, 5), *multi_atlas_result, None, None)
    for j in range(3):
        axes1[j][0].set_ylabel(r'$\frac{1}{N_{\mathrm{trig}}}\frac{dN^{\mathrm{pair}}}{d\Delta\phi}-C_{\mathrm{ZYAM}}$', size = 150)
        multiplicity = j*35 + 22.5
        for i in range(3):
            cms = classes.Drawing_Graphs(13000, (2, 5), *multi_atlas_result[i+3*j], None, None)
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

    axes1[0][0].text(-0.9, 0.0145, r'$50 \leq N^{\mathrm{rec}}_{\mathrm{ch}} < 60$', size = 150)
    axes1[0][1].text(-0.9, 0.0145, r'$60 \leq N^{\mathrm{rec}}_{\mathrm{ch}} < 70$', size = 150)
    axes1[0][2].text(-0.9, 0.0145, r'$70 \leq N^{\mathrm{rec}}_{\mathrm{ch}} < 80$', size = 150)
    axes1[1][0].text(-0.9, 0.032, r'$80 \leq N^{\mathrm{rec}}_{\mathrm{ch}} < 90$', size = 150)
    axes1[1][1].text(-0.9, 0.032, r'$90 \leq N^{\mathrm{rec}}_{\mathrm{ch}} < 100$', size = 150)
    axes1[1][2].text(-0.9, 0.032, r'$100 \leq N^{\mathrm{rec}}_{\mathrm{ch}} < 110$', size = 150)
    axes1[2][0].text(-0.9, 0.0625, r'$110 \leq N^{\mathrm{rec}}_{\mathrm{ch}} < 120$', size = 150)
    axes1[2][1].text(-0.9, 0.0625, r'$120 \leq N^{\mathrm{rec}}_{\mathrm{ch}} < 130$', size = 150)
    axes1[2][2].text(-0.9, 0.0625, r'$130 \leq N^{\mathrm{rec}}_{\mathrm{ch}}$', size = 150)

    axes1[0][0].set_ylim(-0.001,0.0159)
    axes1[1][0].set_ylim(-0.001,0.0349)
    axes1[2][0].set_ylim(-0.001,0.0699)

    fig1.tight_layout(h_pad = -1)
    fig1.savefig('./Results/rezero_atlas_Nk.png')

    # 파라미터들 정리해서 그림 형태로 저장
    fig1 = plt.figure()
    ax1 = plt.axes()
    ax2 = ax1.twinx()
    fig1.set_size_inches(35, 16.534, forward=True)
    multiplicity_atlas = [55, 65, 75, 85, 95, 105, 115, 125, 135]
    multiplicity_range = [[5,5,5,5,5,5,5,5,5],[5,5,5,5,5,5,5,5,5]]
    q = []; T = []
    for list in multi_atlas_result:
        q.append(list[0]);  T.append(list[1])

    lns1 = ax1.scatter(multiplicity_atlas, q, edgecolor = 'blue', facecolors='none', s=900, marker='o', linewidths=5, zorder=2, label=r'$q\quad (\mathrm{left \, axis})$')
    ax1.errorbar(multiplicity_atlas, q, xerr=multiplicity_range, color="blue", linestyle=' ', linewidth=7, capthick=3, capsize=15)
    ax1.scatter(multiplicity_atlas, q, facecolors='blue', s=900, marker='+', linewidths=5, zorder=2)
    lns2 = ax2.scatter(multiplicity_atlas, T, edgecolor = 'red', facecolors='none', s=900, marker='o', linewidths=5, zorder=2, label=r'$T\quad (\mathrm{right \, axis})$')
    ax2.scatter(multiplicity_atlas, T, facecolors='red', s=900, marker='+', linewidths=5, zorder=2)
    ax2.errorbar(multiplicity_atlas, T, xerr=multiplicity_range, color="red", linestyle=' ', linewidth=7, capthick=3, capsize=15)

    ax1.set_xlabel(r'$N^{\mathrm{rec}}_{\mathrm{ch}}$',size=70)
    ax1.set_xlim(50, 140)
    ax1.set_ylabel(r'$ q(\mathrm{GeV}) $',size=70)
    ax1.set_ylim(0.3, 1.8)
    ax2.set_ylabel(r'$ T(\mathrm{GeV}) $',size=70)
    # ax2.set_ylim(1.325, 1.7)      # T 기준을 AuAu 200GeV로 할 경우
    ax2.set_ylim(1.01, 1.16)        # T 기준을 pp 13TeV로 할 경우

    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '${:g}$'.format(y)))
    ax1.tick_params(axis='both',which='major',direction='in',width=2,length=30,labelsize=45, top='true')
    ax1.tick_params(axis='both',which='minor',direction='in',width=2,length=15,labelsize=45, top='true')
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '${:g}$'.format(y)))
    ax2.tick_params(axis='both',which='major',direction='in',width=2,length=30,labelsize=45, top='true')
    ax2.tick_params(axis='both',which='minor',direction='in',width=2,length=15,labelsize=45, top='true')

    ax1.grid(color='silver',linestyle=':',linewidth=5, zorder=0)

    lns = [lns1, lns2]
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, fontsize=50, loc='upper left')

    plt.tight_layout()
    fig1.savefig('./Results/parameters_multiplicity_dep_CMS.png')
'''

drawgraph_ptdep_phicorr()
time_phicorr = time.time()
print(f"Graph, Phi correlation end : {time_phicorr-time_calculate:.3f} sec")
# drawgraph_ptdep_Yridge()
time_yridge = time.time()
print(f"Graph, Yridge end : {time_yridge-time_phicorr:.3f} sec")
drawgraph_ptdep_frnk()
time_frnk = time.time()
print(f"FrNk end : {time_frnk-time_yridge:.3f} sec")
# drawgraph_cmdep_phicorr()
time_multi = time.time()
print(f"Graph, Multiplicity end : {time_multi-time_frnk:.3f} sec")
# drawgraph_multi_phicorr()
time_ptdist = time.time()
print(f"Graph, pT distribution end : {time_ptdist-time_multi:.3f} sec")


time_end = time.time()
print(f"Total end : {time_end-time_start:.3f} sec")