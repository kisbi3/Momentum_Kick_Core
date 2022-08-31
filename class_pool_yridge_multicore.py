from math import dist
from scipy.optimize import curve_fit
import numpy as np
import cupy as cp
import scipy
import pprint as pp
import multiprocessing

import Function_cpu as cpu
import Function_gpu as gpu

'''Yridge를 멀티코어로 돌려보려고 코딩한 파일.'''


# python에서는 private은 변수 앞에 '__'을 붙여서 표현한다.
class Fitting_gpu:
    __yi = (-10,10)
    __pti = (0,10)
    __a = .5    #fall off parameter
    __m = 0.13957018  #m == mpi
    __mb = __m #mb==mpi, GeV
    __md = 1.   #GeV
    __sqrSnn = 13000.
    __mp = 0.938272046 #Proton mass, GeV
    __Yridge_phif_start = -1.18
    __Yridge_phif_end = 1.18

    def __init__(self, phi_array, data, pt_range, multi, ptf, etaf, boundary, initial, mode):
        # array or list (2d)
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

        # self.data = np.array(data).flatten().tolist()
        self.data = dist_dat_array
        if len(self.phi_array) != len(self.data):
            print("phi array and data array have to be same length")
            quit()
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
        self.pt_range = pt_range
        dist_ptrange_loww = np.array([])
        dist_ptrange_high = np.array([])
        # checking for length of alice and cms Yridge data
        self.ptrange_index = (len(pt_range[0][0]), len(pt_range[0][1]))
        for i in range(len(pt_range[0])):
            if i==0:
                dist_ptrange_loww = pt_range[0][i]
                dist_ptrange_high = pt_range[1][i]
            else:
                dist_ptrange_loww = np.concatenate((dist_ptrange_loww, pt_range[0][i]))
                dist_ptrange_high = np.concatenate((dist_ptrange_high, pt_range[1][i]))
        self.ptrange_loww = dist_ptrange_loww
        self.ptrange_high = dist_ptrange_high
        # print(dist_ptrange_loww, dist_ptrange_high)

    
    def __FrNk(self, xx, yy, pt):
        return xx+yy*pt*pt
    def __Fr(self, xx, yy, pt):
        return xx+yy*pt*pt
    def __Nk(self, A, B, multi):
        return A*cp.exp(-B/multi)

    def fitting(self, error):
        if error is None:
            if self.mode == "pTdependence":
                popt, pcov = scipy.optimize.curve_fit(self.fitting_func, xdata = self.phi_array, ydata = self.data, bounds=self.boundary, p0 = self.initial)
            elif self.mode == "Multiplicity":
                popt, pcov = scipy.optimize.curve_fit(self.fitting_func__multi, xdata = self.phi_array, ydata = self.data, bounds=self.boundary, p0 = self.initial)
            elif self.mode == "Both":
                pass    #temporarily

            else:
                print("Typo!")
                quit()

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
                popt, pcov = scipy.optimize.curve_fit(self.fitting_func__multi, xdata = self.phi_array, ydata = self.data, bounds=self.boundary, p0 = self.initial, sigma = error_array, absolute_sigma = True)
            elif self.mode == "Both":
                pass    #temporarily

            else:
                print("Typo!")
                quit()
        return popt, np.sqrt(np.diag(pcov))

    #   fitting parameters : kick, Tem, xx, yy, AA, BB
    def fitting_func__multi(self, phi_dist, kick, Tem, xx, yy, AA, BB):
        phi_array = phi_dist.reshape((self.Number_of_Array,-1))
        phi_array = []
        for i in range(self.Number_of_Array):
            phi_array.append(phi_dist[self.array_length[i]*i : self.array_length[i]*(i+1)])
        Aridge_bin = 1000
        pti, yi = np.meshgrid(np.linspace(self.__pti[0], self.__pti[1], Aridge_bin), np.linspace(self.__yi[0], self.__yi[1], Aridge_bin))
        dyi = (self.__pti[1] - self.__pti[0])/Aridge_bin
        dpti = (self.__yi[1] - self.__yi[0])/Aridge_bin
        Aridge = cp.asarray(1/np.sum(cpu.Aridge(pti, yi, Tem, self.__m, self.__md, self.__a, self.__sqrSnn, self.__mp)*dyi*dpti*2*np.pi))
        result = np.array([])
        result_dist = []
        for i in range(len(phi_array)):
            multi = 10*i + 95
            dist = self.__multiplicity(multi, phi_array[i], self.etaf, Aridge, kick, Tem, xx, yy, AA, BB)
            result_dist.append(dist-min(dist))
            if i ==0:
                result = result_dist[i]
            else:
                result = np.concatenate((result, result_dist[i]))
        print(kick, Tem, xx, yy, np.sum((result-self.data)**2))
        return result

    def __multiplicity(self, multi, phi_array, etaf, Aridge, kick, Tem, xx, yy, A, B):
        bin = 500
        ptf, etaf, phif = cp.meshgrid(cp.linspace(self.ptf[0], self.ptf[1], bin), cp.linspace(etaf[0], etaf[1], bin), cp.asarray(phi_array))
        dptf = (self.ptf[1] - self.ptf[0])/bin
        detaf = (etaf[1]-etaf[0])/bin
        delta_Deltaeta = 2*(etaf[1]-etaf[0])
        dist = cp.sum((4/3)*self.__Fr(A, B, multi)*self.__Nk(xx, yy, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.__sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a), axis=(0))*dptf*detaf/delta_Deltaeta
        return cp.asnumpy(cp.sum(dist, axis=(0)))
    

    #   fitting parameters : kick, Tem, xx, yy
    def fitting_func(self, given_array, kick, Tem, xx, yy):
    # def fitting_func(self, kick, Tem, xx, yy):
        # phi_array = phi_dist.reshape((self.Number_of_Array,-1))
        ''' delete Yridge array'''
        given_array = given_array[0:-14]      # fitting에 사용하는 데이터가 Yridge가 포함되어 있는 경우
        phi_array = []
        start = 0
        end = self.array_length[0]
        ''' Yridge 추가할 예정 '''
        '''Yridge를 gpu에 할당, 나머지를 cpu에 할당할 예정'''
        # 아직 안함
        '''마지막 두개는 Yridge이므로 이 두개를 제외해야 함'''
        '''Yridge는 너무 오차가 커서 fitting이 잘 안되는 것으로 보임'''
        for i in range(self.Number_of_Array-2):       # fitting에 사용하는 데이터가 Yridge가 포함되어 있는 경우
        # for i in range(self.Number_of_Array):
            phi_array.append(given_array[start : end])
            start += self.array_length[i]
            end += self.array_length[i]
        Aridge_bin = 500
        pti, yi = np.meshgrid(np.linspace(self.__pti[0], self.__pti[1], Aridge_bin), np.linspace(self.__yi[0], self.__yi[1], Aridge_bin))
        dyi = (self.__pti[1] - self.__pti[0])/Aridge_bin
        dpti = (self.__yi[1] - self.__yi[0])/Aridge_bin
        Aridge = cp.asarray(1/np.sum(cpu.Aridge(pti, yi, Tem, self.__m, self.__md, self.__a, self.__sqrSnn, self.__mp)*dyi*dpti*2*np.pi))
        
        # import time
        # time_start = time.time()
        result = []
        result_dist = []
        for i in range(len(phi_array)):
            dist = self.__ptdep(phi_array[i], self.etaf[i], self.ptf[i], Aridge, kick, Tem, xx, yy)
            result_dist.append(dist-min(dist))
            if i==0:
                result = result_dist[i]
            else:
                result = np.concatenate((result, result_dist[i]))
        # time_end = time.time()
        # print(f"Phi Corr : {time_end-time_start:.3f} sec")
        
        # time_start = time.time()
        result = np.concatenate((result, self.Yridge(Aridge, kick, Tem, xx, yy)))       # fitting에 사용하는 데이터가 Yridge가 포함되어 있는 경우
        # time_end = time.time()
        # print(f"Yridge : {time_end-time_start:.3f} sec")
        print(kick, Tem, xx, yy, np.sum((result-self.data)**2))
        return result
    
    def __ptdep(self, phi_array, etaf, ptf_dist, Aridge, kick, Tem, xx, yy):
        bin = 500
        ptf, etaf, phif = cp.meshgrid(cp.linspace(ptf_dist[0], ptf_dist[1], bin), cp.linspace(etaf[0], etaf[1], bin), cp.asarray(phi_array))
        dptf = (ptf_dist[1]-ptf_dist[0])/bin
        detaf = (etaf[1]-etaf[0])/bin
        delta_Deltaeta = 2*(etaf[1]-etaf[0])
        dist = cp.sum(self.__FrNk(xx, yy, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.__sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a), axis=0)*dptf*detaf/delta_Deltaeta
        return (4/3)*cp.asnumpy(cp.sum(dist, axis=(0)))

    def Yridge(self, Aridge, kick, Tem, xx, yy):
        results = np.array([])
        CZYAM = np.array([])
        etaf_range = [(1.6, 1.8), (2, 4)]
        bin = 300

        '''CZAYM calculate'''
        for i in range(len(self.pt_range[0])):
            for j in range(len(self.pt_range[0][i])):
                phif = self.__Yridge_phif_end
                ptf, etaf = cp.meshgrid(cp.linspace(self.pt_range[0][i][j], self.pt_range[1][i][j], bin), cp.linspace(etaf_range[i][0], etaf_range[i][1], bin))
                dptf = (self.pt_range[1][i][j]- self.pt_range[0][i][j])/bin
                detaf = (etaf_range[i][1]-etaf_range[i][0])/bin
                delta_Deltaeta = 2*(etaf_range[i][1]-etaf_range[i][0])
                result = (4/3)*cp.sum(self.__FrNk(xx, yy, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.__sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a))*dptf*detaf/delta_Deltaeta
                CZYAM = np.append(CZYAM, cp.asnumpy(result*phif*2))

        for i in range(len(self.pt_range[0])):
            for j in range(len(self.pt_range[0][i])):
                ptf, etaf, phif = cp.meshgrid(cp.linspace(self.pt_range[0][i][j], self.pt_range[1][i][j], bin), cp.linspace(etaf_range[i][0], etaf_range[i][1], bin), cp.linspace(self.__Yridge_phif_start, self.__Yridge_phif_end, bin))
                dptf = (self.pt_range[1][i][j]- self.pt_range[0][i][j])/bin
                detaf = (etaf_range[i][1]-etaf_range[i][0])/bin
                dphif = (self.__Yridge_phif_end - self.__Yridge_phif_start)/bin
                delta_Deltaeta = 2*(etaf_range[i][1]-etaf_range[i][0])
                result = (4/3)*cp.sum(self.__FrNk(xx, yy, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.__sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a))*dptf*detaf*dphif/delta_Deltaeta
                results = np.append(results, cp.asnumpy(result))
        
        phif_start = self.__Yridge_phif_start
        phif_end = self.__Yridge_phif_end
        frnk = self.__FrNk(xx, yy, ptf)
        global Yridge_calculate
        def Yridge_calculate(etaf_loww, etaf_high, ptloww, pthigh):
            bin = 200
            ptf, etaf, phif = cp.meshgrid(cp.linspace(ptloww, pthigh, bin), cp.linspace(etaf_loww, etaf_high, bin), cp.linspace(phif_start, phif_end, bin))
            dptf = (pthigh - ptloww)/bin
            detaf = (etaf_high-etaf_loww)/bin
            dphif = (phif_end - phif_start)/bin
            delta_Deltaeta = 2*(etaf_high-etaf_loww)
            result = (4/3)*cp.sum(frnk*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.__sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a))*dptf*detaf*dphif/delta_Deltaeta
            return result
        
        etaf_range_loww = np.zeros(5)+etaf_range[0][0]
        etaf_range_loww = np.append(etaf_range_loww, np.zeros(9)+etaf_range[1][0])
        etaf_range_high = np.zeros(5)+etaf_range[0][1]
        etaf_range_high = np.append(etaf_range_high, np.zeros(9)+etaf_range[1][1])
        ptloww = self.pt_range[0][0]
        ptloww = np.append(ptloww, self.pt_range[0][1])
        pthigh = self.pt_range[1][0]
        pthigh = np.append(pthigh, self.pt_range[1][1])
        iterable = zip(etaf_range_loww, etaf_range_high, ptloww, pthigh)
        pool = multiprocessing.Pool(processes=14)
        results = pool.starmap(Yridge_calculate, iterable)
        # proc = multiprocessing.Process(target=Yridge_calculate, args=(iterable,))
        # proc.start(0)
        pool.join()
        # proc.join(0)
        # print(results-CZYAM)
        result_alice = results[0:5]-CZYAM[0:5]
        result_cms = results[5::]
        result_total = np.append(result_alice, result_cms)
        # return results-CZYAM
        return result_total

    def Yridge_calculate(self, Aridge, kick, Tem, xx, yy, etaf_loww, etaf_high, ptloww, pthigh):
        bin = 200
        ptf, etaf, phif = cp.meshgrid(cp.linspace(ptloww, pthigh, bin), cp.linspace(etaf_loww, etaf_high, bin), cp.linspace(self.__Yridge_phif_start, self.__Yridge_phif_end, bin))
        dptf = (pthigh - ptloww)/bin
        detaf = (etaf_high-etaf_loww)/bin
        dphif = (self.__Yridge_phif_end - self.__Yridge_phif_start)/bin
        delta_Deltaeta = 2*(etaf_high-etaf_loww)
        result = (4/3)*cp.sum(self.__FrNk(xx, yy, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.__sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a))*dptf*detaf*dphif/delta_Deltaeta
        return result


class Drawing_Graphs:
    '''Yridge를 포함하여 fitting하는 경우, class Drawing_Graphs가 제대로 작동하지 않는 문제가 있다. 어차피 fitting도 잘 안되기 때문에 Yridge를 fitting에서 제외할 것이다.'''
    __yi = (-10,10)
    __pti = (0,10)
    __a = .5    #fall off parameter
    __m = 0.13957018  #m == mpi
    __mb = __m #mb==mpi, GeV
    __md = 1.   #GeV
    __sqrSnn = 13000.
    __mp = 0.938272046 #Proton mass, GeV
    __Yridge_phif_start = -1.18
    __Yridge_phif_end = 1.18
    
    def __init__(self, etaf, kick, Tem, xx, yy, AA, BB):
        # self.multi = multi
        # self.ptf = ptf
        self.etaf = etaf
        # self.phif = phif
        '''만약 그래프를 그릴때 에러가 발생한다면 아래 print부분을 출력해보자.'''
        # print(type(kick), type(Tem))      

        self.kick = kick
        self.Tem = Tem
        self.xx = xx
        self.yy = yy
        self.AA = AA
        self.BB = BB
        #mode : Multiplciity, pTdependence, Both
        # self.mode = mode
    
    def __Aridge(self):
        Aridge_bin = 5000
        pti, yi = np.meshgrid(np.linspace(self.__pti[0], self.__pti[1], Aridge_bin), np.linspace(self.__yi[0], self.__yi[1], Aridge_bin))
        dyi = (self.__pti[1] - self.__pti[0])/Aridge_bin
        dpti = (self.__yi[1] - self.__yi[0])/Aridge_bin
        return cp.asarray(1/np.sum(cpu.Aridge(pti, yi, cp.asnumpy(self.Tem), self.__m, self.__md, self.__a, self.__sqrSnn, self.__mp)*dyi*dpti*2*np.pi))

    def __FrNk(self, xx, yy, pt):
        return xx+yy*pt*pt
    def __Fr(self, xx, yy, pt):
        return xx+yy*pt*pt
    def __Nk(self, A, B, multi):
        return A*cp.exp(-B/multi)

    def result_plot(self, mode, multi, ptf, phif):
        if mode == "Multiplicity":
            return self.__Ridge_Multi(multi, ptf, phif)
        elif mode == "pTdependence":
            return self.__Ridge_ptdep(ptf, phif)
        elif mode == "Both":
            pass
        else:
            print("Maybe Typo")
            quit()

    def __Ridge_Multi(self, multi, ptf, phif):
        Aridge = self.__Aridge()
        bin = 350
        delta_Deltaeta = 2*(self.etaf[1]-self.etaf[0])
        dptf = (ptf[1]-ptf[0])/bin
        ptf, etaf, phif = cp.meshgrid(cp.linspace(ptf[0], ptf[1], bin), cp.linspace(self.etaf[0], self.etaf[1], bin), cp.linspace(phif[0], phif[1], bin))
        detaf = (self.etaf[1]-self.etaf[0])/bin
        dist = cp.sum((4/3)*self.__Fr(self.AA, self.BB, multi)*self.__Nk(self.xx, self.yy, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, self.kick, self.Tem, self.__sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a), axis=(0))*dptf*detaf/delta_Deltaeta
        Ridge_phi = cp.asnumpy(cp.sum(dist, axis=0))
        return cp.asnumpy(phif[0][0]), Ridge_phi-min(Ridge_phi)
        
    def __Ridge_ptdep(self, ptf, phif):
        Aridge = self.__Aridge()
        bin = 350
        delta_Deltaeta = 2*(self.etaf[1]-self.etaf[0])
        dptf = (ptf[1]-ptf[0])/bin
        ptf, etaf, phif = cp.meshgrid(cp.linspace(ptf[0], ptf[1], bin), cp.linspace(self.etaf[0], self.etaf[1], bin), cp.linspace(phif[0], phif[1], bin))
        detaf = (self.etaf[1]-self.etaf[0])/bin
        dist = cp.sum((4/3)*self.__FrNk(self.xx, self.yy, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, self.kick, self.Tem, self.__sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a), axis = 1)*dptf*detaf/delta_Deltaeta
        Ridge_phi = cp.asnumpy(cp.sum(dist, axis=0))
        return cp.asnumpy(phif[0][0]), Ridge_phi-min(Ridge_phi)
    
    def Yridge(self, pt_range, czyam):
        if czyam == "Subtract":      # ALICE
            kick = self.kick; Tem = self.Tem; xx = self.xx; yy = self.yy
            Aridge = self.__Aridge()
            results = np.array([])
            CZYAM = np.array([])
            ptrange_avg = np.array([])
            bin = 300

            '''CZAYM calculate'''
            for i in range(len(pt_range[0])):
                phif = self.__Yridge_phif_end
                ptf, etaf = cp.meshgrid(cp.linspace(pt_range[0][i], pt_range[1][i], bin), cp.linspace(self.etaf[0], self.etaf[1], bin))
                dptf = (pt_range[1][i] - pt_range[0][i])/bin
                detaf = (self.etaf[1] - self.etaf[0])/bin
                delta_Deltaeta = 2*(self.etaf[1]-self.etaf[0])
                result = (4/3)*cp.sum(self.__FrNk(xx, yy, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.__sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a))*dptf*detaf/delta_Deltaeta
                CZYAM = np.append(CZYAM, cp.asnumpy(result*phif*2))
                ptrange_avg = np.append(ptrange_avg, (pt_range[0][i]+pt_range[1][i])/2)
            '''Yridge calculate'''
            for i in range(len(pt_range[0])):
                ptf, etaf, phif = cp.meshgrid(cp.linspace(pt_range[0][i], pt_range[1][i], bin), cp.linspace(self.etaf[0], self.etaf[1], bin), cp.linspace(self.__Yridge_phif_start, self.__Yridge_phif_end, bin))
                dptf = (pt_range[1][i] - pt_range[0][i])/bin
                detaf = (self.etaf[1] - self.etaf[0])/bin
                dphif = (self.__Yridge_phif_end - self.__Yridge_phif_start)/bin
                delta_Deltaeta = 2*(self.etaf[1]-self.etaf[0])
                result = (4/3)*cp.sum(self.__FrNk(xx, yy, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.__sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a))*dptf*detaf*dphif/delta_Deltaeta
                results = np.append(results, cp.asnumpy(result))
            return ptrange_avg, results-CZYAM

        elif czyam == "Remove":     # CMS
            kick = self.kick; Tem = self.Tem; xx = self.xx; yy = self.yy
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
                result = (4/3)*cp.sum(self.__FrNk(xx, yy, ptf)*ptf*gpu.Ridge_dist(Aridge, ptf, etaf, phif, kick, Tem, self.__sqrSnn, self.__mp, self.__m, self.__mb, self.__md, self.__a))*dptf*detaf*dphif/delta_Deltaeta
                results = np.append(results, cp.asnumpy(result))
                ptrange_avg = np.append(ptrange_avg, (pt_range[0][i]+pt_range[1][i])/2)
            return ptrange_avg, results
        
        else:
            print("Error! in Drawing Yridge Graph, Maybe Typo")


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
