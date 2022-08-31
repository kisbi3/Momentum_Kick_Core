import numpy as np

def rapidityinitt(pt, eta, mb):     #P_T Initial, Eta_initial, mb(Pion mass)
    root=np.sqrt((pt*pt*np.cosh(eta)*np.cosh(eta))+mb*mb)
    a = root+pt*np.sinh(eta)
    b = root-pt*np.sinh(eta)
    try:
        return (np.log(a/b))/2
    except:
        print("Error in Function_cpu, rapidityinitt")

def lightcone(pti, yi, sqrSnn, mp, m):  #P_T Initial, Rapidity Initial, sqr(Snn), proton mass, pion mass
    yb = np.arccosh(sqrSnn/(2*mp))
    squareroot=np.sqrt(m*m+pti*pti)
    yiabs = np.abs(yi)
    try:
        return (squareroot/m)*np.exp(yiabs-yb)
    except:
        print("Error in Function_cpu, lightcone")
        exit(0)

# Calculate Aridge
def Aridge(pti, yi, T, m, md, a, sqrSnn, mp):  #P_T Initial, Rapidity Initial, Temperature, pion mass, cut off parameter, fall of parameter, Squareroot(Snn), proton mass
    x = 1-lightcone(pti, yi, sqrSnn, mp, m)
    x = np.where(x>0, x, 0)
    try:
        return pti*x**a*np.exp(-np.sqrt(m*m+pti*pti)/T)/np.sqrt(md*md+pti*pti)
    except:
        print("Error in Function_cpu, Aridge")


def Aridge_result(pti, yi, T, m, md, a, sqrSnn, mp):  #P_T Initial, Rapidity Initial, Temperature, pion mass, cut off parameter, fall of parameter, Squareroot(Snn), proton mass
    x = 1-lightcone(pti, yi, sqrSnn, mp, m)
    x = np.where(x>0, x, 0)
    # squareroot = np.sqrt(m*m+pti*pti)
    try:
        return (x**a)*np.exp(-np.sqrt(m*m+pti*pti)/T)/np.sqrt(md*md+pti*pti)
    except:
        print("Error in Function_cpu, Aridge_result")


def Ridge_dist(Aridge, ptf, etaf, phif, q, T, sqrSnn, mp, m, mb, md, a):  #Aridge, P_T Final, Eta Final, Phi Final, q, Temperature, Pion mass, Beam mass(Pion)
    ptisq = ptf*ptf-2*ptf*q*np.cos(phif)+q*q    #Eta_jet 생략
    pti = np.sqrt(ptisq)
    Energy = np.sqrt(ptf*ptf*np.cosh(etaf)*np.cosh(etaf)+m*m)
    Energy_i = np.sqrt(pti*pti+ptf*ptf*np.sinh(etaf)*np.sinh(etaf)+m*m)

    yi = np.log((Energy_i+ptf*np.sinh(etaf))/(Energy_i-ptf*np.sinh(etaf)))/2
    yf = np.log((Energy+ptf*np.sinh(etaf))/(Energy-ptf*np.sinh(etaf)))/2

    x = 1-lightcone(pti, yi, sqrSnn, mp, m)
    dist = (Aridge*Aridge_result(pti, yi, T, m, md, a, sqrSnn, mp))*np.sqrt(1.-((mb*mb)/((mb*mb+ptf*ptf)*np.cosh(yf)*np.cosh(yf))))*(Energy/Energy_i)   # E/Ei 임을 명심하자.
    result = np.where(x>0, dist, 0)
    try:
        return result
    except:
        print("Error in Function_cpu, Ridge_dist")

# 0712.3282 (10)
def intergralNjet(pt, eta, phi, Njet, Tjet, ma, m, szero):   #P_T Final, Eta Final, Phi Final, ma, Pion mass, Sigma_(phi0)
    constant = Njet/(Tjet*(m+Tjet)*2*np.pi())
    sigmaphi = (szero*szero*ma*ma)/(ma*ma+pt*pt)
    try:
        return (constant/sigmaphi)*np.exp(((m-np.sqrt(m*m+pt*pt))/Tjet)-((phi*phi+eta*eta)/(2*sigmaphi)))
    except:
        print("Error in Function_cpu, integralNjet")