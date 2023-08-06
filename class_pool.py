from math import dist
from scipy.optimize import curve_fit
import numpy as np
import cupy as cp
import scipy
import pprint as pp
import multiprocessing
import time

import Function_cpu as cpu
import Function_gpu as gpu


# python에서는 private은 변수 앞에 '__'을 붙여서 표현한다.
class Fitting_gpu:
    '''fitting 하기 위해 함수가 실행된 횟수'''
    __count = 0 

    __yi = (-10,10)
    __pti = (0,10)
    __a = .5    #fall off parameter
    __m = 0.13957018  #m == mpi
    __mb = __m #mb==mpi, GeV
    # __md = 1.   #GeV
    __mp = 0.938272046 #Proton mass, GeV
    __Yridge_phif_start = -1.18
    __Yridge_phif_end = 1.18
    __error_temp = []
    Mode = "Nothing"
    def multiplicity_fitting_mode(self, mode):
        self.Mode = mode

    def __init__(self, sqrSnn, phi_array, data, pt_range, multi, ptf, etaf, boundary, initial, mode):
        self.sqrSnn = sqrSnn
        # array or list (2d)
        if mode == "CMenergy":
            self.array_length = 1
            self.phi_array = phi_array
            self.data = data

        else:
            self.Number_of_Array = len(phi_array)      # Number of Arrays
            array_length = []
            total_length = 0
            dist_phi_array = np.array([])
            dist_dat_array = np.array([])
            for i in range(len(phi_array)):
                array_length.append(len(phi_array[i]))
                total_length += len(phi_array[i])
                if i==0:
                    dist_phi_array = phi_array[i]
                    dist_dat_array = data[i]
                else:
                    dist_phi_array = np.concatenate((dist_phi_array, phi_array[i]))
                    dist_dat_array = np.concatenate((dist_dat_array, data[i]))
            self.array_length = array_length
            self.phi_array = dist_phi_array
            self.data = dist_dat_array
        
        # Multiplicity (tuple or list or array)
        self.multi = multi
        # [ptf_start, ptf_end] (tuple or list)
        # multiplicity인 경우, [0.5, 5]
        # in pTdependence, give etaf in 2d array. ex) [[1, 2], [2, 3]]
        self.ptf = ptf
        # [etaf_start, etaf_end]
        # in pTdependence, give etaf in 2d array. ex) [[1.6, 1.8], [2, 4]]
        self.etaf = etaf
        # boundary condition (tuple)
        self.boundary = boundary
        # initial parameters (tuple)
        self.initial = initial
        # mode : Multiplciity, pTdependence, Both
        self.mode = mode
        # Yridge ([pt_low : array], [pt_high : array]) : tuple
        if pt_range is None:
            self.ptdis_number = None

        else:
            self.pt_range = pt_range
            '''number of pt distribution data ex) 13TeV : ALICE + CMS = 2, 7 TeV : CMS = 1'''
            self.ptdis_number = len(pt_range[0])


        '''데이터 길이 확인하기 위함'''
        if mode == "CMenergy":
            if (len(self.phi_array) != len(self.data)):
                print("phi array and data array have to be same length")
                quit()
        else:
            if (len(self.phi_array) != len(self.data)) and (len(self.data) != len(self.ptf)) and (len(self.ptf) != len(self.etaf)):
                print("phi array and data array have to be same length")
                quit()
    
    def __FrNk(self, xx, yy, zz, pt):
        # return xx+yy*pt*pt
        # return xx*cp.exp(yy*pt)
        # return xx*cp.exp(-yy/pt-zz*pt)
        # ''' 13 TeV에서 zz항이 의미 없어진 것을 확인 -> 7 TeV에서도 zz항을 없애버림 '''
        # return xx*cp.exp(-yy/pt)
        ''' frnk 상수로 두면 어떻게 되려나?'''
        return xx
        # return 1.415411195083602
    def __Fr(self, xx, yy, pt):
        return xx+yy*pt*pt
    def __Nk(self, A, B, multi):
        return A*cp.exp(-B/multi)
    def __MultiNk(self, Free1, Free2, Free3, multi):
        return Free1*(Free2+Free3*multi)
        # return Free1*(Free2+Free3*multi*multi)
        # return Free1 + Free2*np.exp(-Free3/multi)

    def fitting(self, error, Fixed_parameters):
        if error is None:
            if self.mode == "pTdependence":
                popt, pcov = scipy.optimize.curve_fit(self.fitting_func, xdata = self.phi_array, ydata = self.data, bounds=self.boundary, p0 = self.initial)
                return popt, self.chisq_error
            elif self.mode == "Multiplicity":

                print(f'Mode : {self.Mode}')
                print(Fixed_parameters[1])
                # 각 파라미터들을 고정시켜가며 어떤게 가장 dominant한지 확인하는 작업
                # self.Mode = "Free kick"
                if (self.Mode == "Free kick"):
                    # dist = Fixed_parameters[1]
                    # self.Fixed_Temperature = [dist, dist, dist, dist, dist, dist, dist, dist, dist]
                    self.Fixed_Temperature = Fixed_parameters[1]
                    self.Fixed_xx = Fixed_parameters[2]
                    self.Fixed_yy = Fixed_parameters[3]
                    self.Fixed_zz = Fixed_parameters[4]
                elif(self.Mode == "Free Tem"):
                    # Temperature dependence는 물리적으로 말이 안되기 때문에 앞으로 쓸 일 없을 듯.
                    self.Fixed_kick = Fixed_parameters[0]
                    self.Fixed_xx = Fixed_parameters[2]
                    self.Fixed_yy = Fixed_parameters[3]
                    self.Fixed_zz = Fixed_parameters[4]
                elif(self.Mode == "Free fRNk xx"):
                    # dist = Fixed_parameters[1]
                    # self.Fixed_Temperature = [dist, dist, dist, dist, dist, dist, dist, dist, dist]
                    print(Fixed_parameters)
                    self.Fixed_Temperature = Fixed_parameters[1]
                    self.Fixed_kick = Fixed_parameters[0]
                    self.Fixed_yy = Fixed_parameters[3]
                    self.Fixed_zz = Fixed_parameters[4]

                elif(self.Mode == "Final"):
                    print(Fixed_parameters)
                    self.Fixed_Temperature = Fixed_parameters[1]
                    self.Fixed_kick = Fixed_parameters[0]
                    self.Fixed_yy = Fixed_parameters[3]
                    self.Fixed_zz = Fixed_parameters[4]

                elif(self.Mode == "Free kick and fRNk xx_FixedTem"):
                    self.Fixed_Temperature = Fixed_parameters[1]
                    self.Fixed_yy = Fixed_parameters[3]
                    self.Fixed_zz = Fixed_parameters[4]
            
                elif(self.Mode == "Free kick and fRNk xx"):
                    self.Fixed_Temperature = Fixed_parameters[1]
                    self.Fixed_yy = Fixed_parameters[3]
                    self.Fixed_zz = Fixed_parameters[4]

                elif(self.Mode == "Nothing"):
                    # Temperature를 pT mean으로 결정하는 경우
                    self.Fixed_Temperature = Fixed_parameters[0]
                    self.Fixed_xx = Fixed_parameters[1][0]
                    self.Fixed_yy = Fixed_parameters[1][1]
                    self.Fixed_zz = Fixed_parameters[1][2]

                elif(self.Mode == "Final_ATLAS"):
                    print(Fixed_parameters)
                    self.Fixed_Temperature = Fixed_parameters[1]
                    self.Fixed_kick = Fixed_parameters[0]
                    self.Fixed_yy = Fixed_parameters[3]
                    self.Fixed_zz = Fixed_parameters[4]

                else:
                    print("Error!! Mode 스펠링 체크 필요")
                    print("Mode : ", self.Mode)
                    exit()

                totalresult = []
                phi_array_sep = []; data_sep = []
                start = 0
                # print(self.phi_array)
                for i in range(self.Number_of_Array):
                    number = self.array_length[i]
                    phi_array_sep.append(self.phi_array[start : start + number])
                    data_sep.append(self.data[start : start + number])
                    start += number
                self.data_sep = data_sep
                # print(self.data_sep)

                if (self.Mode == "Final_ATLAS"):
                    result_temp, pcov = scipy.optimize.curve_fit(self.fitting_func_multi_final_ATLAS, xdata = self.phi_array, ydata = self.data, bounds=self.boundary, p0 = self.initial)
                    popt = result_temp

                else:
                    for i in range(len(phi_array_sep)):
                        '''Fitting하는 번호'''
                        self.separate_number = i
                        self.__count = 0
                        if(self.Mode == "Free kick and fRNk xx_FixedTem" or self.Mode == "Free kick and fRNk xx"):
                            print(i, "\t Temperature : ", self.Fixed_Temperature)
                            result_temp, pcov = scipy.optimize.curve_fit(self.fitting_func_multi_double, xdata = phi_array_sep[i], ydata = data_sep[i], bounds=self.boundary, p0 = self.initial, method='trf')
                        # elif(self.Mode == "Final"):
                        #     print(i, "\t Temperature : ", self.Fixed_Temperature[i])
                        #     # result_temp, pcov = scipy.optimize.curve_fit(self.fitting_func_multi, xdata = phi_array_sep[i], ydata = data_sep[i], bounds=self.boundary, p0 = self.initial, method='trf')
                        #     result_temp, pcov = scipy.optimize.curve_fit(self.fitting_func_multi, xdata = self.phi_array, ydata = self.data, bounds=self.boundary, p0 = self.initial, method='trf')
                        else:
                            print(i, "\t Temperature : ", self.Fixed_Temperature)

                            result_temp, pcov = scipy.optimize.curve_fit(self.fitting_func_multi, xdata = phi_array_sep[i], ydata = data_sep[i], bounds=self.boundary, p0 = self.initial, method='trf')
                        print(result_temp)
                        totalresult.append(result_temp)
                        popt = totalresult
                print(popt)
                print("Error : ", self.__error_temp)
                

            elif self.mode == "CMenergy":
                popt, pcov = scipy.optimize.curve_fit(self.fitting_func, xdata = self.phi_array, ydata = self.data, bounds=self.boundary, p0 = self.initial)
            elif self.mode == "Both":
                pass    #temporarily

            else:
                print("Typo!")
                quit()

        # '''error가 있는 경우지만, 지금은 사용하지 않음(만들다 말았음)'''
        else:
            error_array = np.array([])
            for i in range(len(error)):
                if i==0:
                    error_array = error[i]
                else:
                    error_array = np.concatenate((error_array, error[i]))
            if self.mode == "pTdependence":
                popt, pcov = scipy.optimize.curve_fit(self.fitting_func, xdata = self.phi_array, ydata = self.data, bounds=self.boundary, p0 = self.initial, sigma = error_array, absolute_sigma = True)
            elif self.mode == "Multiplicity":
                # self.Fixed_Temperature = Fixed_Temperature
                self.Fixed_Temperature = Fixed_parameters[0]
                self.Fixed_xx = Fixed_parameters[1][0]
                self.Fixed_yy = Fixed_parameters[1][1]
                self.Fixed_zz = Fixed_parameters[1][2]
                popt, pcov = scipy.optimize.curve_fit(self.fitting_func_multi, xdata = self.phi_array, ydata = self.data, bounds=self.boundary, p0 = self.initial, sigma = error_array, absolute_sigma = True)
            elif self.mode == "CMenergy":
                popt, pcov = scipy.optimize.curve_fit(self.fitting_func, xdata = self.phi_array, ydata = self.data, bounds=self.boundary, p0 = self.initial, sigma = error_array, absolute_sigma = True)
            elif self.mode == "Both":
                pass    #temporarily

            else:
                print("Typo!")
                quit()
        return popt, np.sqrt(np.diag(pcov))

    # Free1(Free2+Free3*multiplicity)
    # def fitting_func_multi_final_ATLAS(self, phi_array, Free1, Free2, Free3):
    def fitting_func_multi_final_ATLAS(self, phi_array, Free1):
    # def fitting_func_multi_final_ATLAS(self, phi_array, Free1, Free2):
        self.__md = 1.
        # 함수 외부에서 쪼개서 각각을 fitting 했었지만, 이번에는 내부에서 데이터를 쪼개야 함.
        multi_data = [55, 65, 75, 85, 95, 105, 115, 125, 135]
        multi_slope = [0.00023138684077866737, 0.00032327077444578383, 0.0004426004230247412, 0.0004974045871900281, 0.0005646750762371896, 0.0005071353528608789, 0.0005299309414008545, 0.0006254010939585516, 0.0006254010939585516]
        # Free1 = 1
        Free2 = -0.03495215
        Free3 = 0.00065111


        start = 0
        phi_array_sep = []; data_sep = []
        for i in range(self.Number_of_Array):
            number = self.array_length[i]
            phi_array_sep.append(phi_array[start : start + number])
            data_sep.append(self.data[start : start + number])
            start += number
        '''xx에 1을 넣어서 xx가 없는 것으로 하자.'''
        Tem = self.Fixed_Temperature
        kick = self.Fixed_kick
        # kick = Free3
        # xx = 2.243448959076882
        xx = 1
        yy = self.Fixed_yy
        zz = self.Fixed_zz
        Aridge_bin = 1000
        pti, yi = np.meshgrid(np.linspace(self.__pti[0], self.__pti[1], Aridge_bin), np.linspace(self.__yi[0], self.__yi[1], Aridge_bin))
        dyi = (self.__pti[1] - self.__pti[0])/Aridge_bin
        dpti = (self.__yi[1] - self.__yi[0])/Aridge_bin

        totalresult = []
        for i in range(len(phi_array_sep)):
            Aridge = cp.asarray(1/np.sum(cpu.Aridge(pti, yi, Tem[i], self.__m, self.__md, self.__a, self.sqrSnn, self.__mp)*dyi*dpti*2*np.pi))
            # result = self.__MultiNk(Free1, Free2, Free3, multi_data[i]) * self.__multiplicity(phi_array_sep[i], self.etaf, Aridge, kick, Tem[i], xx, yy, zz)
            result = self.__MultiNk(Free1, Free2, Free3, multi_data[i]) * self.__multiplicity(phi_array_sep[i], self.etaf, Aridge, kick, Tem[i], xx, yy, zz)
            temp = result - np.min(result)
            totalresult.extend(temp.tolist())

        totalresult = np.array(totalresult)
        self.__count = self.__count + 1

        if self.__count == 1:
            print("Count \t kick \t xx\t yy\t zz\t Free1 \t\t Free2 \t\t Free3 \t\t Error")
            print(f"{self.__count}회", kick, xx, yy, zz, Free1, Free2, Free3, np.sum((totalresult-self.data)**2))
        elif self.__count%5==0:
            print(f"{self.__count}회", kick, xx, yy, zz, Free1, Free2, Free3, np.sum((totalresult-self.data)**2))
        # if self.__count%10==0:
        #     print(totalresult)
            # print("deviation")
            # print(totalresult-self.data)

        self.__error_temp = np.sum((totalresult-self.data)**2)
        return totalresult

    def fitting_func_multi(self, phi_array, Free):
        multi_data = [55, 65, 75, 85, 95, 105, 115, 125, 135]
        AAA = -0.03495215; BBB = 0.00065111
        self.__md = 1.
        # xx = 5.3; yy = 8.5*10**(-35); zz = 0.22
        if (self.Mode == "Nothing"):
            xx = self.Fixed_xx; yy = self.Fixed_yy; zz = self.Fixed_zz
            Tem = self.Fixed_Temperature
        elif (self.Mode=="Free kick"):
            xx = self.Fixed_xx; yy = self.Fixed_yy; zz = self.Fixed_zz
            Tem = self.Fixed_Temperature
            kick = Free
        elif (self.Mode=="Free Tem"):
            xx = self.Fixed_xx; yy = self.Fixed_yy; zz = self.Fixed_zz
            kick = self.Fixed_kick
            Tem = Free
        elif(self.Mode == "Free fRNk xx"):
            xx = Free
            yy = self.Fixed_yy; zz = self.Fixed_zz
            kick = self.Fixed_kick
            Tem = self.Fixed_Temperature[self.separate_number]          

        elif(self.Mode == "Final"):
            xx = Free
            yy = self.Fixed_yy; zz = self.Fixed_zz
            kick = self.Fixed_kick
            Tem = self.Fixed_Temperature[self.separate_number]
        Aridge_bin = 1000
        pti, yi = np.meshgrid(np.linspace(self.__pti[0], self.__pti[1], Aridge_bin), np.linspace(self.__yi[0], self.__yi[1], Aridge_bin))
        dyi = (self.__pti[1] - self.__pti[0])/Aridge_bin
        dpti = (self.__yi[1] - self.__yi[0])/Aridge_bin
        number = self.separate_number
        # Aridge = cp.asarray(1/np.sum(cpu.Aridge(pti, yi, Tem[number], self.__m, self.__md, self.__a, self.sqrSnn, self.__mp)*dyi*dpti*2*np.pi))
        # result = self.__multiplicity(phi_array, self.etaf, Aridge, kick, Tem[number], xx, yy, zz)
        Aridge = cp.asarray(1/np.sum(cpu.Aridge(pti, yi, Tem, self.__m, self.__md, self.__a, self.sqrSnn, self.__mp)*dyi*dpti*2*np.pi))
        result = self.__MultiNk(1, AAA, BBB, multi_data[self.separate_number]) * self.__multiplicity(phi_array, self.etaf, Aridge, kick, Tem, xx, yy, zz)
        result = result - np.min(result)
        self.__count = self.__count + 1
        if self.__count == 1:
            print("Multiplicity : ", multi_data[self.separate_number])
            print("Count \t Kick \t\t Tem \t\t xx \t\t yy \t zz \t\t Free2, \t\t Free3\t\t Error")
            print(f"{self.__count}회", kick, Tem, xx, yy, zz, AAA, BBB, np.sum((result-self.data_sep[number])**2))
            self.__error_temp.append(np.sum((result-self.data_sep[number])**2))
        elif self.__count%5==0:
            print(f"{self.__count}회", kick, Tem, xx, yy, zz, AAA, BBB, np.sum((result-self.data_sep[number])**2))
            print(result)
            print(self.data_sep[number])
        self.__error_temp[number] = np.sum((result-self.data_sep[number])**2)
        return result

    def fitting_func_multi_double(self, phi_array, Free1, Free2):       
        self.__md = 1.
        xx = Free2
        yy = self.Fixed_yy; zz = self.Fixed_zz
        kick = Free1
        if(self.Mode == "Free kick and fRNk xx_FixedTem"):
            Tem = self.Fixed_Temperature
        if(self.Mode == "Free kick and fRNk xx"):
            Tem = self.Fixed_Temperature[self.separate_number]
        Aridge_bin = 1000
        pti, yi = np.meshgrid(np.linspace(self.__pti[0], self.__pti[1], Aridge_bin), np.linspace(self.__yi[0], self.__yi[1], Aridge_bin))
        dyi = (self.__pti[1] - self.__pti[0])/Aridge_bin
        dpti = (self.__yi[1] - self.__yi[0])/Aridge_bin
        number = self.separate_number
        # Aridge = cp.asarray(1/np.sum(cpu.Aridge(pti, yi, Tem[number], self.__m, self.__md, self.__a, self.sqrSnn, self.__mp)*dyi*dpti*2*np.pi))
        # result = self.__multiplicity(phi_array, self.etaf, Aridge, kick, Tem[number], xx, yy, zz)
        Aridge = cp.asarray(1/np.sum(cpu.Aridge(pti, yi, Tem, self.__m, self.__md, self.__a, self.sqrSnn, self.__mp)*dyi*dpti*2*np.pi))
        result = self.__multiplicity(phi_array, self.etaf, Aridge, kick, Tem, xx, yy, zz)
        result = result - np.min(result)
        self.__count = self.__count + 1
        if self.__count == 1:
            print("Count \t Kick \t\t Tem \t\t xx \t\t yy \t zz \t\t Error")
            print(f"{self.__count}회", kick, Tem, xx, yy, zz, np.sum((result-self.data_sep[number])**2))
            self.__error_temp.append(np.sum((result-self.data_sep[number])**2))
        elif self.__count%5==0:
            print(f"{self.__count}회", kick, Tem, xx, yy, zz, np.sum((result-self.data_sep[number])**2))
            # print(result)
            # print(self.data_sep[number])
        self.__error_temp[number] = np.sum((result-self.data_sep[number])**2)
        return result

    def __multiplicity(self, phi_array, etaf_range, Aridge, kick, Tem, xx, yy, zz):
        bin = 300
        ptf, etaf, phif = cp.meshgrid(cp.linspace(self.ptf[0], self.ptf[1], bin), cp.linspace(etaf_range[0], etaf_range[1], bin), cp.asarray(phi_array))
        dptf = (self.ptf[1] - self.ptf[0])/bin
        detaf = (etaf_range[1]-etaf_range[0])/bin
        delta_Deltaeta = 2*(etaf_range[1]-etaf_range[0])
        # dist = cp.sum((4/3)*self.__Fr(A, B, multi)*self.__Nk(xx, yy, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a), axis=(0))*dptf*detaf/delta_Deltaeta
        # deltapt = 1/(ptf_dist[1] - ptf_dist[0])       #pt normalize
        '''ATLAS는 deltapt 없는듯. ATLAS로 시작하고 있으니 일단 1로 두자.'''
        deltapt = 1
        dist = deltapt*cp.sum(self.__FrNk(xx, yy, zz, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a), axis=0)*dptf*detaf/delta_Deltaeta
        return (4/3)*cp.asnumpy(cp.sum(dist, axis=(0)))
    

    ''' fitting parameters : kick, Tem, xx, yy, zz
        zz는 frnk에 추가적으로 들어갈 수 있는 parameter '''

    def fitting_func(self, given_array, kick, Tem, xx, yy, zz):
        '''md를 q로 한번 두고 fitting 해보자.'''
        # self.__md = kick
        self.__md = 1.
        if self.ptdis_number is None:
            phi_array = given_array
        else:
            ''' delete Yridge array'''
            numberof_ptdis = 0
            for i in range(self.ptdis_number):
                numberof_ptdis += len(self.pt_range[0][i])
            given_array = given_array[0:-numberof_ptdis]
            phi_array = []
            start = 0
            for i in range(self.Number_of_Array - self.ptdis_number):       # fitting에 사용하는 데이터가 Yridge가 포함되어 있는 경우 활성화
                end = start + self.array_length[i]
                phi_array.append(given_array[start : end])
                start += self.array_length[i]
        Aridge_bin = 1000
        pti, yi = np.meshgrid(np.linspace(self.__pti[0], self.__pti[1], Aridge_bin), np.linspace(self.__yi[0], self.__yi[1], Aridge_bin))
        dpti = (self.__pti[1] - self.__pti[0])/Aridge_bin
        dyi = (self.__yi[1] - self.__yi[0])/Aridge_bin
        Aridge = cp.asarray(1/np.sum(cpu.Aridge(pti, yi, Tem, self.__m, self.__md, self.__a, self.sqrSnn, self.__mp)*dyi*dpti*2*np.pi))

        if self.ptdis_number is None and self.array_length == 1:
            ''' CM energy dependence'''            
            # deltapt = self.ptf[0][1] - self.ptf[0][0]
            deltapt = 1
            yy = zz = 0
            result_dist = deltapt*self.__ptdep(phi_array, self.etaf[0], self.ptf[0], Aridge, kick, Tem, xx, yy, zz)
            result = result_dist - min(result_dist)
            self.__count = self.__count + 1
            
        else:
            ''' Only pT dependence'''
            result = []
            result_dist = []
            for i in range(len(phi_array)):
                dist = self.__ptdep(phi_array[i], self.etaf[i], self.ptf[i], Aridge, kick, Tem, xx, yy, zz)
                result_dist.append(dist-min(dist))
                if i==0:
                    result = result_dist[i]
                else:
                    result = np.concatenate((result, result_dist[i]))
            result = np.concatenate((result, self.Yridge(Aridge, kick, Tem, xx, yy, zz)))       # fitting에 사용하는 데이터가 Yridge가 포함되어 있는 경우 활성화
            self.__count = self.__count + 1
        if self.__count == 1 or self.__count%10==0:
            print(f"{self.__count}회", kick, Tem, xx, yy, zz, np.sum((result-self.data)**2))
        self.chisq_error = np.sum((result-self.data)**2)
        return result
    
    def __ptdep(self, phi_array, etaf, ptf_dist, Aridge, kick, Tem, xx, yy, zz):
        bin = 300
        detaf = (etaf[1]-etaf[0])/bin
        delta_Deltaeta = 2*(etaf[1]-etaf[0])
        ptf, etaf, phif = cp.meshgrid(cp.linspace(ptf_dist[0], ptf_dist[1], bin), cp.linspace(etaf[0], etaf[1], bin), cp.asarray(phi_array))
        dptf = (ptf_dist[1]-ptf_dist[0])/bin
        # deltapt = 1/(ptf_dist[1] - ptf_dist[0])       #pt normalize
        deltapt = 1
        dist = deltapt*cp.sum(self.__FrNk(xx, yy, zz, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a), axis=0)*dptf*detaf/delta_Deltaeta
        return (4/3)*cp.asnumpy(cp.sum(dist, axis=(0)))

    def Yridge(self, Aridge, kick, Tem, xx, yy, zz):
        results = np.array([])
        CZYAM = np.array([])
        etaf_range = [(1.6, 1.8), (2, 4)]
        bin = 300

        ''' Yridge 계산 최적화(1회 계산당 약 0.5초 절약)'''
        for i in range(len(self.pt_range[0])):
            for j in range(len(self.pt_range[0][i])):
                ptf, phif, etaf = cp.meshgrid(cp.linspace(self.pt_range[0][i][j], self.pt_range[1][i][j], bin), cp.linspace(self.__Yridge_phif_start, self.__Yridge_phif_end, bin), cp.linspace(etaf_range[i][0], etaf_range[i][1], bin))
                dptf = (self.pt_range[1][i][j] - self.pt_range[0][i][j])/bin
                detaf = (etaf_range[i][1] - etaf_range[i][0])/bin
                deltapt = 1/(self.pt_range[1][i][j] - self.pt_range[0][i][j])       #pt normalize
                dphif = (self.__Yridge_phif_end - self.__Yridge_phif_start)/bin
                delta_Deltaeta = 2*(etaf_range[i][1] - etaf_range[i][0])
                result = deltapt*(4/3)*cp.sum(self.__FrNk(xx, yy, zz, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a), axis=(2,1))*dptf*detaf*dphif/delta_Deltaeta
                result = result - min(result);    result = cp.sum(result)
                results = np.append(results, cp.asnumpy(result))                
        return results

    def Fixed_Temp(multi, meanpT, pp13_highmulti_Temp):
        # AuAu : 200GeV
        # AuAu_meanpT = 0.39
        # AuAu_Temp = 0.5
        '''Associated Yield를 이용해 도출한 mean pT'''
        # pp13_highmulti_meanpT = 1.186
        ''' 단순히 multiplicity로 도출'''
        pp13_highmulti_meanpT = 1.209
        # return (meanpT/AuAu_meanpT)*AuAu_Temp
        return (meanpT/pp13_highmulti_meanpT)*pp13_highmulti_Temp



class Drawing_Graphs:
    '''Yridge를 포함하여 fitting하는 경우, class Drawing_Graphs가 제대로 작동하지 않는 문제가 있다. 어차피 fitting도 잘 안되기 때문에 Yridge를 fitting에서 제외할 것이다.'''
    __yi = (-10,10)
    __pti = (0,10)
    __a = .5    #fall off parameter
    __m = 0.13957018  #m == mpi
    __mb = __m #mb==mpi, GeV
    # __md = 1.   #GeV
    # __md = 9.61957351e-01
    # __sqrSnn = 13000.
    # __sqrSnn = 7000.
    __mp = 0.938272046 #Proton mass, GeV
    __Yridge_phif_start = -1.18
    __Yridge_phif_end = 1.18
    
    def __init__(self, sqrSnn, etaf, kick, Tem, xx, yy, zz, AA, BB, type):
        # self.multi = multi
        # self.ptf = ptf
        self.etaf = etaf
        # self.phif = phif
        '''만약 그래프를 그릴때 에러가 발생한다면 아래 print부분을 출력해보자.'''
        # print(type(kick), type(Tem), type(xx), type(yy), type(zz))      

        self.kick = kick
        self.Tem = Tem
        self.xx = xx
        self.yy = yy
        self.AA = AA
        self.BB = BB
        self.zz = zz
        self.sqrSnn = sqrSnn
        self.type = type
        #mode : Multiplciity, pTdependence, Both
        # self.mode = mode
    
    def __Aridge(self):
        Aridge_bin = 5000
        pti, yi = np.meshgrid(np.linspace(self.__pti[0], self.__pti[1], Aridge_bin), np.linspace(self.__yi[0], self.__yi[1], Aridge_bin))
        dyi = (self.__pti[1] - self.__pti[0])/Aridge_bin
        dpti = (self.__yi[1] - self.__yi[0])/Aridge_bin
        return cp.asarray(1/np.sum(cpu.Aridge(pti, yi, cp.asnumpy(self.Tem), self.__m, self.__md, self.__a, self.sqrSnn, self.__mp)*dyi*dpti*2*np.pi))

    def __FrNk(self, xx, yy, zz, pt):
        # return xx+yy*pt*pt
        # return xx*cp.exp(yy*pt)
        # return xx*cp.exp(-yy/pt-zz*pt)
        # return xx*cp.exp(-yy/pt)
        # return 1.415411195083602
        ''' frnk 상수로 두면 어떻게 되려나?'''
        return xx
    def __Fr(self, xx, yy, pt):
        return xx+yy*pt*pt
    def __Nk(self, A, B, multi):
        return A*cp.exp(-B/multi)

    def result_plot(self, mode, multi, ptf, phif):
        if mode == "Multiplicity":
            return self.__Ridge_Multi(multi, ptf, phif)
        elif mode == "pTdependence":
            return self.__Ridge_ptdep(ptf, phif)
        elif mode == "Multiplicity_FinalATLAS":
            return self.__Ridge_Multi_FinalATLAS(multi, ptf, phif)
        # elif mode == "CMenergy":
        #     pass
            # return self.__Ridge_ptdep(sqrsnn, ptf, phif)
        elif mode == "Both":
            pass
        else:
            print("Maybe Typo")
            quit()
    
    def MultiNk_param(self, Free1, Free2, Free3):
        self.Free1 = Free1
        self.Free2 = Free2
        self.Free3 = Free3

    def __MultiNk_func(self, Free1, Free2, Free3, multi):
        return Free1*(Free2+Free3*multi)
        # return Free1*(Free2+Free3*multi*multi)
        # return Free1 + Free2*np.exp(-Free3/multi)

    def __Ridge_Multi_FinalATLAS(self, multi, ptf, phif_range):
        # self.__md = self.kick
        self.__md = 1.
        kick = self.kick; Tem = self.Tem; xx = self.xx; yy = self.yy; zz = self.zz; AA = self.AA; BB = self.BB; etaf_range = self.etaf
        Aridge = self.__Aridge()
        bin = 300
        delta_Deltaeta = 2*(etaf_range[1]-etaf_range[0])
        ptf_range = (0.5, 5)
        dptf = (ptf_range[1]-ptf_range[0])/bin
        ptf, etaf, phif = cp.meshgrid(cp.linspace(ptf_range[0], ptf_range[1], bin), cp.linspace(etaf_range[0], etaf_range[1], bin), cp.linspace(phif_range[0], phif_range[1], bin))
        detaf = (etaf_range[1]-etaf_range[0])/bin
        ridge_integrate = ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a)
        ridge_integrate = ridge_integrate*self.__FrNk(xx, yy, zz, ptf)*self.__MultiNk_func(self.Free1, self.Free2, self.Free3, multi)
        # dist_integrate = cp.sum(ridge_integrate, axis = 1)*(4/3)*dptf*detaf/delta_Deltaeta
        if self.type == 'ATLAS':
            dist_integrate = cp.sum(ridge_integrate, axis = 1)*(4/3)*dptf*detaf/delta_Deltaeta
        else:
            dist_integrate = cp.sum(ridge_integrate, axis = 1)*(4/3)*dptf*detaf/(delta_Deltaeta*(ptf_range[1]-ptf_range[0]))
        Ridge_phi = cp.asnumpy(cp.sum(dist_integrate, axis = 0))
        return cp.asnumpy(phif[0][0]), Ridge_phi-min(Ridge_phi)

    def __Ridge_Multi(self, multi, ptf, phif_range):
        # self.__md = self.kick
        self.__md = 1.
        kick = self.kick; Tem = self.Tem; xx = self.xx; yy = self.yy; zz = self.zz; AA = self.AA; BB = self.BB; etaf_range = self.etaf
        Aridge = self.__Aridge()
        bin = 300
        delta_Deltaeta = 2*(etaf_range[1]-etaf_range[0])
        ptf_range = (0.5, 5)
        dptf = (ptf_range[1]-ptf_range[0])/bin
        ptf, etaf, phif = cp.meshgrid(cp.linspace(ptf_range[0], ptf_range[1], bin), cp.linspace(etaf_range[0], etaf_range[1], bin), cp.linspace(phif_range[0], phif_range[1], bin))
        detaf = (etaf_range[1]-etaf_range[0])/bin
        ridge_integrate = ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a)
        ridge_integrate = ridge_integrate*self.__FrNk(xx, yy, zz, ptf) * self.__MultiNk_func(1, -0.03495215, 0.00065111, multi)
        # dist_integrate = cp.sum(ridge_integrate, axis = 1)*(4/3)*dptf*detaf/delta_Deltaeta
        if self.type == 'ATLAS':
            dist_integrate = cp.sum(ridge_integrate, axis = 1)*(4/3)*dptf*detaf/delta_Deltaeta
        else:
            dist_integrate = cp.sum(ridge_integrate, axis = 1)*(4/3)*dptf*detaf/(delta_Deltaeta*(ptf_range[1]-ptf_range[0]))
        Ridge_phi = cp.asnumpy(cp.sum(dist_integrate, axis = 0))
        return cp.asnumpy(phif[0][0]), Ridge_phi-min(Ridge_phi)


    def __Ridge_ptdep(self, ptf_range, phif_range):
        # self.__md = self.kick
        self.__md = 1.

        kick = self.kick; Tem = self.Tem; xx = self.xx; yy = self.yy; zz = self.zz
        Aridge = self.__Aridge()
        bin = 300
        delta_Deltaeta = 2*(self.etaf[1]-self.etaf[0])
        dptf = (ptf_range[1]-ptf_range[0])/bin
        # deltapt = 1/(ptf_range[1] - ptf_range[0])       #pt normalize
        deltapt = 1
        ptf, etaf, phif = cp.meshgrid(cp.linspace(ptf_range[0], ptf_range[1], bin), cp.linspace(self.etaf[0], self.etaf[1], bin), cp.linspace(phif_range[0], phif_range[1], bin))
        detaf = (self.etaf[1]-self.etaf[0])/bin
        dist = deltapt*(4/3)*cp.sum(self.__FrNk(xx, yy, zz, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a), axis = 1)*dptf*detaf/delta_Deltaeta
        # Ridge_phi = cp.asnumpy(cp.sum(dist, axis=0))
        
        '''아래 경우는 pt에 대해서 normalization한 경우임. pt range에 따라서 나누어져 있기에 normalize 하면 안됨'
        if self.type == 'ATLAS':
            Ridge_phi = cp.asnumpy(cp.sum(dist, axis=0))*(ptf_range[1]-ptf_range[0])
            # Ridge_phi = cp.asnumpy(cp.sum(dist, axis=0))*(5-0.5)
        else:
            Ridge_phi = cp.asnumpy(cp.sum(dist, axis=0))
        '''
        Ridge_phi = cp.asnumpy(cp.sum(dist, axis=0))
        
        return cp.asnumpy(phif[0][0]), Ridge_phi-min(Ridge_phi)
        '''아래 경우는 CZYAM을 제거하여 제대로 pt range가 나눠져 있는 결과와 합쳐져 있는 결과의 적분값이 동일한지 확인하기 위함. 정상적으로 그리려면 위에 있는 코드 사용.'''
        # return cp.asnumpy(phif[0][0]), Ridge_phi
    

    # Yridge를 데이터와 동일하게 '점'으로 표현할 경우
    def Yridge(self, pt_range, czyam):
        # self.__md = self.kick
        self.__md = 1.
        if czyam == "Subtract":      # ALICE
            kick = self.kick; Tem = self.Tem; xx = self.xx; yy = self.yy; zz = self.zz
            Aridge = self.__Aridge()
            results = np.array([])
            CZYAM = np.array([])
            ptrange_avg = np.array([])
            bin = 300

            '''CZYAM calculate'''
            for i in range(len(pt_range[0])):
                phif = self.__Yridge_phif_end
                ptf, etaf = cp.meshgrid(cp.linspace(pt_range[0][i], pt_range[1][i], bin), cp.linspace(self.etaf[0], self.etaf[1], bin))
                dptf = (pt_range[1][i] - pt_range[0][i])/bin
                detaf = (self.etaf[1] - self.etaf[0])/bin
                delta_Deltaeta = 2*(self.etaf[1]-self.etaf[0])
                result = (4/3)*cp.sum(self.__FrNk(xx, yy, zz, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a))*dptf*detaf/delta_Deltaeta
                CZYAM = np.append(CZYAM, cp.asnumpy(result*phif*2))
                ptrange_avg = np.append(ptrange_avg, (pt_range[0][i]+pt_range[1][i])/2)
            '''Yridge calculate'''
            for i in range(len(pt_range[0])):
                ptf, etaf, phif = cp.meshgrid(cp.linspace(pt_range[0][i], pt_range[1][i], bin), cp.linspace(self.etaf[0], self.etaf[1], bin), cp.linspace(self.__Yridge_phif_start, self.__Yridge_phif_end, bin))
                dptf = (pt_range[1][i] - pt_range[0][i])/bin
                detaf = (self.etaf[1] - self.etaf[0])/bin
                dphif = (self.__Yridge_phif_end - self.__Yridge_phif_start)/bin
                delta_Deltaeta = 2*(self.etaf[1]-self.etaf[0])
                result = (4/3)*cp.sum(self.__FrNk(xx, yy, zz, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a))*dptf*detaf*dphif/delta_Deltaeta
                results = np.append(results, cp.asnumpy(result))
            return ptrange_avg, results-CZYAM

        elif czyam == "Remove":     # CMS
            kick = self.kick; Tem = self.Tem; xx = self.xx; yy = self.yy; zz = self.zz
            Aridge = self.__Aridge()
            results = np.array([])
            ptrange_avg = np.array([])
            bin = 300
            '''Yridge calculate'''
            for i in range(len(pt_range[0])):
                ptf, etaf, phif = cp.meshgrid(cp.linspace(pt_range[0][i], pt_range[1][i], bin), cp.linspace(self.etaf[0], self.etaf[1], bin), cp.linspace(self.__Yridge_phif_start, self.__Yridge_phif_end, bin))
                dptf = (pt_range[1][i] - pt_range[0][i])/bin
                detaf = (self.etaf[1] - self.etaf[0])/bin
                dphif = (self.__Yridge_phif_end - self.__Yridge_phif_start)/bin
                delta_Deltaeta = 2*(self.etaf[1]-self.etaf[0])
                result = (4/3)*cp.sum(self.__FrNk(xx, yy, zz, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a))*dptf*detaf*dphif/delta_Deltaeta
                results = np.append(results, cp.asnumpy(result))
                ptrange_avg = np.append(ptrange_avg, (pt_range[0][i]+pt_range[1][i])/2)
            return ptrange_avg, results



    # 앞선 함수와 다르게 Yridge를 실선으로 그리고자 할 경우
    def Yridge_line(self, czyam):
        # self.__md = self.kick
        self.__md = 1.
        if czyam == "Subtract":      # ALICE

            ptbin = 150
            ptstart = 0.1
            ptend = 11.1
            pt_normal = (ptend-ptstart)/ptbin
            pt_range = [cp.linspace(ptstart, ptend, ptbin), cp.linspace(ptstart+pt_normal, ptend+pt_normal, ptbin)]
            pt_range_cpu = [np.linspace(ptstart, ptend, ptbin), np.linspace(ptstart+pt_normal, ptend+pt_normal, ptbin)]

            # pt_normal = pt_normal * 2         # pt range 0.5로 normalization

            kick = self.kick; Tem = self.Tem; xx = self.xx; yy = self.yy; zz = self.zz
            Aridge = self.__Aridge()
            results = np.array([])
            CZYAM = np.array([])
            ptrange_avg = np.array([])
            bin = 300
            
            '''CZYAM calculate'''
            for i in range(len(pt_range[0])):
                phif = self.__Yridge_phif_end
                ptf, etaf = cp.meshgrid(cp.linspace(pt_range[0][i], pt_range[1][i], bin), cp.linspace(self.etaf[0], self.etaf[1], bin))
                dptf = (pt_range[1][i] - pt_range[0][i])/bin
                detaf = (self.etaf[1] - self.etaf[0])/bin
                delta_Deltaeta = 2*(self.etaf[1]-self.etaf[0])
                result = (4/(3*pt_normal))*cp.sum(self.__FrNk(xx, yy, zz, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a))*dptf*detaf/delta_Deltaeta
                CZYAM = np.append(CZYAM, cp.asnumpy(result*phif*2))
                ptrange_avg = np.append(ptrange_avg, (pt_range_cpu[0][i]+pt_range_cpu[1][i])/2)
            '''Yridge calculate'''
            for i in range(len(pt_range[0])):
                ptf, etaf, phif = cp.meshgrid(cp.linspace(pt_range[0][i], pt_range[1][i], bin), cp.linspace(self.etaf[0], self.etaf[1], bin), cp.linspace(self.__Yridge_phif_start, self.__Yridge_phif_end, bin))
                dptf = (pt_range[1][i] - pt_range[0][i])/bin
                detaf = (self.etaf[1] - self.etaf[0])/bin
                dphif = (self.__Yridge_phif_end - self.__Yridge_phif_start)/bin
                delta_Deltaeta = 2*(self.etaf[1]-self.etaf[0])
                result = (4/(3*pt_normal))*cp.sum(self.__FrNk(xx, yy, zz, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a))*dptf*detaf*dphif/delta_Deltaeta
                results = np.append(results, cp.asnumpy(result))

            return ptrange_avg, results-CZYAM
            # return ptrange_avg, results

        elif czyam == "Subtract_CMS":      # CMS (divide 1.18)

            ptbin = 150
            ptstart = 0.1
            ptend = 11.1
            pt_normal = (ptend-ptstart)/ptbin
            pt_range = [cp.linspace(ptstart, ptend, ptbin), cp.linspace(ptstart+pt_normal, ptend+pt_normal, ptbin)]
            pt_range_cpu = [np.linspace(ptstart, ptend, ptbin), np.linspace(ptstart+pt_normal, ptend+pt_normal, ptbin)]
            # pt_normal = pt_normal * 2         # pt range 0.5로 normalization
            

            kick = self.kick; Tem = self.Tem; xx = self.xx; yy = self.yy; zz = self.zz
            Aridge = self.__Aridge()
            results = np.array([])
            CZYAM = np.array([])
            ptrange_avg = np.array([])
            bin = 300

            '''CZYAM calculate'''
            for i in range(len(pt_range[0])):
                phif = self.__Yridge_phif_end
                ptf, etaf = cp.meshgrid(cp.linspace(pt_range[0][i], pt_range[1][i], bin), cp.linspace(self.etaf[0], self.etaf[1], bin))
                dptf = (pt_range[1][i] - pt_range[0][i])/bin
                detaf = (self.etaf[1] - self.etaf[0])/bin
                delta_Deltaeta = 2*(self.etaf[1]-self.etaf[0])
                result = (4/(3*pt_normal))*cp.sum(self.__FrNk(xx, yy, zz, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a))*dptf*detaf/delta_Deltaeta
                CZYAM = np.append(CZYAM, cp.asnumpy(result*phif*2))
                ptrange_avg = np.append(ptrange_avg, (pt_range_cpu[0][i]+pt_range_cpu[1][i])/2)
            '''Yridge calculate'''
            for i in range(len(pt_range[0])):
                ptf, etaf, phif = cp.meshgrid(cp.linspace(pt_range[0][i], pt_range[1][i], bin), cp.linspace(self.etaf[0], self.etaf[1], bin), cp.linspace(self.__Yridge_phif_start, self.__Yridge_phif_end, bin))
                dptf = (pt_range[1][i] - pt_range[0][i])/bin
                detaf = (self.etaf[1] - self.etaf[0])/bin
                dphif = (self.__Yridge_phif_end - self.__Yridge_phif_start)/bin
                delta_Deltaeta = 2*(self.etaf[1]-self.etaf[0])
                result = (4/(3*pt_normal))*cp.sum(self.__FrNk(xx, yy, zz, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a))*dptf*detaf*dphif/delta_Deltaeta
                results = np.append(results, cp.asnumpy(result))
            return ptrange_avg, (results-CZYAM)/1.18





        elif czyam == "Remove":     # CMS
            kick = self.kick; Tem = self.Tem; xx = self.xx; yy = self.yy; zz = self.zz
            Aridge = self.__Aridge()
            results = np.array([])
            ptrange_avg = np.array([])
            bin = 300
            '''Yridge calculate'''
            for i in range(len(pt_range[0])):
                ptf, etaf, phif = cp.meshgrid(cp.linspace(pt_range[0][i], pt_range[1][i], bin), cp.linspace(self.etaf[0], self.etaf[1], bin), cp.linspace(self.__Yridge_phif_start, self.__Yridge_phif_end, bin))
                dptf = (pt_range[1][i] - pt_range[0][i])/bin
                detaf = (self.etaf[1] - self.etaf[0])/bin
                dphif = (self.__Yridge_phif_end - self.__Yridge_phif_start)/bin
                delta_Deltaeta = 2*(self.etaf[1]-self.etaf[0])
                result = (4/3)*cp.sum(self.__FrNk(xx, yy, zz, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a))*dptf*detaf*dphif/delta_Deltaeta
                results = np.append(results, cp.asnumpy(result))
                ptrange_avg = np.append(ptrange_avg, (pt_range[0][i]+pt_range[1][i])/2)
            return ptrange_avg, results



        else:
            print("Error! in Drawing Yridge Graph, Maybe Typo")

'''여러 에러들 계산 ex) R-squared'''
'''각 데이터를 따로 입력해야 함!'''
''' 보류 '''
class Error:
    __yi = (-10,10)
    __pti = (0,10)
    __a = .5    #fall off parameter
    __m = 0.13957018  #m == mpi
    __mb = __m #mb==mpi, GeV
    __md = 1.   #GeV
    __mp = 0.938272046 #Proton mass, GeV

    def __init__(self, sqrSnn, phi_array, data, ptf, etaf):
        # data
        self.phi_array = phi_array
        self.data = data
        self.ptf = ptf
        self.etaf = etaf
        self.sqrSnn = sqrSnn


    def __FrNk(self, xx, yy, zz, pt):
        # return xx*cp.exp(-yy/pt-zz*pt)
        ''' frnk 상수로 두면 어떻게 되려나?'''
        return xx

    def R_squared(self, datatype, kick, Tem, xx, yy, zz):
        if datatype == "pTdependence":
            mean = np.mean(self.data)
            fitting_result = self.fitting_func(self.phi_array, kick, Tem, xx, yy, zz)
            SSR = np.sum((self.data-fitting_result)*(self.data-fitting_result))
            SST = np.sum((self.data-mean)*(self.data-mean))
            Rsq = 1-(SSR/SST)
        elif datatype == "Multiplicity":
            '''일단 루프만 만들어 두기는 했지만 의미 없을듯?'''
            mean = np.mean(self.data)
            fitting_result = self.fitting_func_multi(self.phi_array, kick, Tem, xx, yy, zz)
            SSR = np.sum((self.data-fitting_result)*(self.data-fitting_result))
            SST = np.sum((self.data-mean)*(self.data-mean))
            Rsq = 1-(SSR/SST)            
        return Rsq

        
    def fitting_func_multi(self, phi_array, kick):
        pass
        # Aridge_bin = 1000
        # pti, yi = np.meshgrid(np.linspace(self.__pti[0], self.__pti[1], Aridge_bin), np.linspace(self.__yi[0], self.__yi[1], Aridge_bin))
        # dyi = (self.__pti[1] - self.__pti[0])/Aridge_bin
        # dpti = (self.__yi[1] - self.__yi[0])/Aridge_bin
        # Aridge = cp.asarray(1/np.sum(cpu.Aridge(pti, yi, Tem[number], self.__m, self.__md, self.__a, self.sqrSnn, self.__mp)*dyi*dpti*2*np.pi))
        # result = self.__multiplicity(phi_array, self.etaf, Aridge, kick, Tem[number], xx, yy, zz)
        # result = result - np.min(result)
        # self.__count = self.__count + 1
        # if self.__count == 1:
        #     print("Count \t Kick \t\t xx \t\t yy \t zz \t\t Error")
        #     print(f"{self.__count}회", kick, xx, yy, zz, np.sum((result-self.data_sep[number])**2))
        # elif self.__count%5==0:
        #     print(f"{self.__count}회", kick, xx, yy, zz, np.sum((result-self.data_sep[number])**2))
        # return result


    def __multiplicity(self, phi_array, etaf_range, Aridge, kick, Tem, xx, yy, zz):
        bin = 300
        ptf, etaf, phif = cp.meshgrid(cp.linspace(self.ptf[0], self.ptf[1], bin), cp.linspace(etaf_range[0], etaf_range[1], bin), cp.asarray(phi_array))
        dptf = (self.ptf[1] - self.ptf[0])/bin
        detaf = (etaf_range[1]-etaf_range[0])/bin
        delta_Deltaeta = 2*(etaf_range[1]-etaf_range[0])
        # dist = cp.sum((4/3)*self.__Fr(A, B, multi)*self.__Nk(xx, yy, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a), axis=(0))*dptf*detaf/delta_Deltaeta
        # deltapt = 1/(ptf_dist[1] - ptf_dist[0])       #pt normalize
        '''ATLAS는 deltapt 없는듯. ATLAS로 시작하고 있으니 일단 1로 두자.'''
        deltapt = 1
        dist = deltapt*cp.sum(self.__FrNk(xx, yy, zz, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a), axis=0)*dptf*detaf/delta_Deltaeta
        return (4/3)*cp.asnumpy(cp.sum(dist, axis=(0)))
    

    ''' fitting parameters : kick, Tem, xx, yy, zz
        zz는 frnk에 추가적으로 들어갈 수 있는 parameter '''

    def fitting_func(self, given_array, kick, Tem, xx, yy, zz):
        ptf_dist = self.ptf
        Aridge_bin = 1000
        pti, yi = np.meshgrid(np.linspace(self.__pti[0], self.__pti[1], Aridge_bin), np.linspace(self.__yi[0], self.__yi[1], Aridge_bin))
        dpti = (self.__pti[1] - self.__pti[0])/Aridge_bin
        dyi = (self.__yi[1] - self.__yi[0])/Aridge_bin
        Aridge = cp.asarray(1/np.sum(cpu.Aridge(pti, yi, Tem, self.__m, self.__md, self.__a, self.sqrSnn, self.__mp)*dyi*dpti*2*np.pi))
        deltapt = 1/(ptf_dist[1] - ptf_dist[0])
        result = deltapt*self.__ptdep(given_array, self.etaf, self.ptf, Aridge, kick, Tem, xx, yy, zz)
        return result-min(result)
    
    def __ptdep(self, phi_array, etaf, ptf_dist, Aridge, kick, Tem, xx, yy, zz):
        bin = 300
        detaf = (etaf[1]-etaf[0])/bin
        delta_Deltaeta = 2*(etaf[1]-etaf[0])
        ptf, etaf, phif = cp.meshgrid(cp.linspace(ptf_dist[0], ptf_dist[1], bin), cp.linspace(etaf[0], etaf[1], bin), cp.asarray(phi_array))
        dptf = (ptf_dist[1]-ptf_dist[0])/bin
        deltapt = 1/(ptf_dist[1] - ptf_dist[0])       #pt normalize
        dist = deltapt*cp.sum(self.__FrNk(xx, yy, zz, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a), axis=0)*dptf*detaf/delta_Deltaeta
        return (4/3)*cp.asnumpy(cp.sum(dist, axis=(0)))

    def Yridge(self, Aridge, kick, Tem, xx, yy, zz):
        results = np.array([])
        CZYAM = np.array([])
        etaf_range = [(1.6, 1.8), (2, 4)]
        bin = 300

        ''' Yridge 계산 최적화(1회 계산당 약 0.5초 절약)'''
        for i in range(len(self.pt_range[0])):
            for j in range(len(self.pt_range[0][i])):
                ptf, phif, etaf = cp.meshgrid(cp.linspace(self.pt_range[0][i][j], self.pt_range[1][i][j], bin), cp.linspace(self.__Yridge_phif_start, self.__Yridge_phif_end, bin), cp.linspace(etaf_range[i][0], etaf_range[i][1], bin))
                dptf = (self.pt_range[1][i][j] - self.pt_range[0][i][j])/bin
                detaf = (etaf_range[i][1] - etaf_range[i][0])/bin
                deltapt = 1/(self.pt_range[1][i][j] - self.pt_range[0][i][j])       #pt normalize
                dphif = (self.__Yridge_phif_end - self.__Yridge_phif_start)/bin
                delta_Deltaeta = 2*(etaf_range[i][1] - etaf_range[i][0])
                result = deltapt*(4/3)*cp.sum(self.__FrNk(xx, yy, zz, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a), axis=(2,1))*dptf*detaf*dphif/delta_Deltaeta
                result = result - min(result);    result = cp.sum(result)
                results = np.append(results, cp.asnumpy(result))                
        return results






'''알고리즘 직접 만든 코드. -> 최적화 하기 귀찮아서 일단 방치'''
class optimize_gpu:
    __yi = (-10,10)
    __pti = (0,10)
    __a = .5    #fall off parameter
    __m = 0.13957018  #m == mpi
    __mb = __m #mb==mpi, GeV
    __md = 1.   #GeV
    __sqrSnn = 13000.
    __mp = 0.938272046 #Proton mass, GeV

    def __init__(self, phi_array, data, multi, ptf, etaf, boundary, initial, mode):
        # data
        self.phi_array = phi_array
        self.data = data
        # Multiplicity (tuple or list or array)
        self.multi = multi
        # [ptf_start, ptf_end] (tuple or list)
        # multiplicity인 경우, [0.5, 5]
        # in pTdependence, give etaf in 2d array. ex) [[1, 2], [2, 3]]
        self.ptf = ptf
        # [etaf_start, etaf_end]
        # in pTdependence, give etaf in 2d array. ex) [[1.6, 1.8], [2, 4]]
        self.etaf = etaf
        # boundary condition (tuple)
        self.boundary = boundary
        # initial parameters (tuple)
        self.initial = initial
        #mode : Multiplciity, pTdependence, Both
        self.mode = mode

    def __FrNk(self, xx, yy, pt):
        return xx+yy*pt*pt
    def __Fr(self, xx, yy, pt):
        return xx+yy*pt*pt
    def __Nk(self, A, B, multi):
        return A*cp.exp(-B/multi)

    def fitting(self):
        if self.mode == "pTdependence":
            popt, pcov = self.LevMar(self.phi_array, self.data, self.fitting_func, self.initial)
        elif self.mode == "Multiplicity":
            popt, pcov = self.LevMar(self.phi_array, self.data, self.fitting_func_multi, self.initial)
        elif self.mode == "Both":
            pass    #temporarily

        else:
            print("Typo!")
            quit()
        # return popt, np.sqrt(np.diag(pcov))
        return popt, pcov

    #   fitting parameters : kick, Tem, xx, yy, AA, BB
    def fitting_func__multi(self, phi_array, kick, Tem, xx, yy, AA, BB):
        Aridge_bin = 1000
        pti, yi = np.meshgrid(np.linspace(self.__pti[0], self.__pti[1], Aridge_bin), np.linspace(self.__yi[0], self.__yi[1], Aridge_bin))
        dyi = (self.__pti[1] - self.__pti[0])/Aridge_bin
        dpti = (self.__yi[1] - self.__yi[0])/Aridge_bin
        Aridge = cp.asarray(1/np.sum(cpu.Aridge(pti, yi, cp.asnumpy(Tem), self.__m, self.__md, self.__a, self.__sqrSnn, self.__mp)*dyi*dpti*2*np.pi))
        result = []
        for i in range(len(phi_array)):
            multi = 10*i + 95
            dist = self.__multiplicity(multi, phi_array[i], self.etaf, Aridge, kick, Tem, xx, yy, AA, BB)
            result.append(dist-min(dist))
        return cp.asarray(np.array(result))
    
    def __multiplicity(self, multi, phi_array, etaf, Aridge, kick, Tem, xx, yy, A, B):
        bin = 300
        ptf, etaf, phif = cp.meshgrid(cp.linspace(self.ptf[0], self.ptf[1], bin), cp.linspace(etaf[0], etaf[1], bin), cp.asarray(phi_array))
        dptf = (self.ptf[1] - self.ptf[0])/bin
        detaf = (etaf[1]-etaf[0])/bin
        delta_Deltaeta = 2*(etaf[1]-etaf[0])
        dist = cp.sum((4/3)*self.__Fr(A, B, multi)*self.__Nk(xx, yy, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.__sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a), axis=(0))*dptf*detaf/delta_Deltaeta
        return cp.asnumpy(cp.sum(dist, axis=(0)))

    #   fitting parameters : kick, Tem, xx, yy
    def fitting_func(self, etaf, ptf, phi_array, kick, Tem, xx, yy):
        Aridge_bin = 1000
        pti, yi = np.meshgrid(np.linspace(self.__pti[0], self.__pti[1], Aridge_bin), np.linspace(self.__yi[0], self.__yi[1], Aridge_bin))
        dyi = (self.__pti[1] - self.__pti[0])/Aridge_bin
        dpti = (self.__yi[1] - self.__yi[0])/Aridge_bin
        Aridge = cp.asarray(1/np.sum(cpu.Aridge(pti, yi, cp.asnumpy(Tem), self.__m, self.__md, self.__a, self.__sqrSnn, self.__mp)*dyi*dpti*2*np.pi))
        result = self.__ptdep(phi_array, etaf, ptf, Aridge, kick, Tem, xx, yy)
        return result
    
    def __ptdep(self, phi_array, etaf, ptf_dist, Aridge, kick, Tem, xx, yy):
        bin = 300
        ptf, etaf, phif = cp.meshgrid(cp.linspace(ptf_dist[0], ptf_dist[1], bin), cp.linspace(etaf[0], etaf[1], bin), cp.asarray(phi_array))
        dptf = (ptf_dist[1]-ptf_dist[0])/bin
        detaf = (etaf[1]-etaf[0])/bin
        delta_Deltaeta = 2*(etaf[1]-etaf[0])
        dist = cp.sum((4/3)*self.__FrNk(xx, yy, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.__sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a), axis=(0))*dptf*detaf/delta_Deltaeta
        return cp.sum(dist, axis=(0))


    def Jacobian_transpose(self, func, constants, xarray, params, h):
        clear = cp.zeros((len(params), len(func(*constants, xarray, *params))))
        for i in range(len(params)):
            dist = cp.zeros(len(params))
            dist[i] = h
            params_high = params + dist
            params_low = params - dist
            clear[i] = (func(*constants, xarray, *params_high)-func(*constants, xarray, *params_low))/(2*h)
        # return cp.transpose(clear)
        return clear

    #Levenberg - Marquardt method
    def LevMar(self, xarray_ndarray, datarray_ndarray, func, initial_params):
        xarray = []
        datarray = []
        for i in range(len(xarray_ndarray)):
            xarray.append(cp.asarray(xarray_ndarray[i]))
            datarray.append(cp.asarray(datarray_ndarray[i]))
        iteration = 0
        eps = 1e-10
        params = cp.array(initial_params)
        residual = []
        Error = 0
        for i in range(len(xarray)):
            residual.append(func(self.etaf[i], self.ptf[i], xarray[i], *params) - datarray[i])
            Error += cp.sum(residual[i]*residual[i])
            residual[i] = residual[i].tolist()
        # print(residual)
        residual = cp.array(sum(residual, []))
        params_matrix = cp.reshape(cp.asarray(params), (len(initial_params), -1))
        while(Error>eps):
            print(Error, params)
            step = 0.00000000000001      # 나중에 손 봐야 함
            datalength = 0
            for i in range(len(xarray_ndarray)):
                datalength += len(xarray_ndarray[i])
            print("1")
            jacT = cp.empty((len(initial_params), 0))
            for i in range(len(xarray)):
                jacT = cp.append(jacT, self.Jacobian_transpose(func, [self.etaf[i], self.ptf[i]], xarray[i], params, 1e-10), axis=1)
                
            print("2")
            jacTjac = jacT @ cp.transpose(jacT)
            params = params - cp.linalg.inv((jacTjac + step*cp.diagonal(jacTjac))) @ jacT @ cp.transpose(residual)
            Error_old = Error
            Error = 0
            residual_dist = []
            for i in range(len(xarray)):
                residual_dist.append(func(self.etaf[i], self.ptf[i], xarray[i], *params) - datarray[i])
                Error += cp.sum(residual_dist[i]*residual_dist[i])
                residual_dist[i] = residual_dist[i].tolist()
            residual = cp.array(sum(residual_dist,[]))
            print("3")
            if cp.abs(Error_old-Error) < eps:
                break
            if iteration>100000:
                quit("Do not Converge")
            iteration += 1
        return params, Error
