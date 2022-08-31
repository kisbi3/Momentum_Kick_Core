import cupy as cp

def rapidityinitt(pt, eta, mb):     #P_T Initial, Eta_initial, mb(Pion mass)
    root=cp.sqrt((pt*pt*cp.cosh(eta)*cp.cosh(eta))+mb*mb)
    a = root+pt*cp.sinh(eta)
    b = root-pt*cp.sinh(eta)
    try:
        return (cp.log(a/b))/2
    except:
        print("Error in Function_gpu, rapidityinitt")
        exit(0)

def lightcone(pti, yi, sqrSnn, mp, m):  #P_T Initial, Rapidity Initial, sqr(Snn), proton mass, pion mass
    yb = cp.arccosh(sqrSnn/(2*mp))
    squareroot=cp.sqrt(m*m+pti*pti)
    yiabs = cp.abs(yi)
    try:
        return (squareroot/m)*cp.exp(yiabs-yb)
    except:
        print("Error in Function_gpu, lightcone")
        exit(0)

# Calculate Aridge
def Aridge(pti, yi, T, m, md, a, sqrSnn, mp):  #P_T Initial, Rapidity Initial, Temperature, pion mass, cut off parameter, fall of parameter, Squareroot(Snn), proton mass
    x = 1-lightcone(pti, yi, sqrSnn, mp, m)
    x = cp.where(x>0, x, 0)
    try:
        return pti*(x**a)*cp.exp(-cp.sqrt(m*m+pti*pti)/T)/cp.sqrt(md*md+pti*pti)
    except:
        print("Error in Function_cpu, Aridge")


def Aridge_result(pti, yi, T, m, md, a, sqrSnn, mp):  #P_T Initial, Rapidity Initial, Temperature, pion mass, cut off parameter, fall of parameter, Squareroot(Snn), proton mass
    x = 1-lightcone(pti, yi, sqrSnn, mp, m)
    x = cp.where(x>0, x, 0)
    # print(x)
    try:
        return (x**a)*cp.exp(-cp.sqrt(m*m+pti*pti)/T)/cp.sqrt(md*md+pti*pti)
    except:
        print("Error in Function_gpu, Aridge_result")
        exit(0)


def Ridge_dist(Aridge, ptf, etaf, phif, q, T, sqrSnn, mp, m, mb, md, a):  #Aridge, P_T Final, Eta Final, Phi Final, q, Temperature, Pion mass, Beam mass(Pion)
    ptisq = ptf*ptf-2*ptf*q*cp.cos(phif)+q*q    #Eta_jet 생략
    pti = cp.sqrt(ptisq)
    Energy = cp.sqrt(ptf*ptf*cp.cosh(etaf)*cp.cosh(etaf)+m*m)
    Energy_i = cp.sqrt(pti*pti+ptf*ptf*cp.sinh(etaf)*cp.sinh(etaf)+m*m)

    yi = cp.log((Energy_i+ptf*cp.sinh(etaf))/(Energy_i-ptf*cp.sinh(etaf)))/2
    yf = cp.log((Energy+ptf*cp.sinh(etaf))/(Energy-ptf*cp.sinh(etaf)))/2
    
    try:
        return (Aridge*Aridge_result(pti, yi, T, m, md, a, sqrSnn, mp))*cp.sqrt(1.-((mb*mb)/((mb*mb+ptf*ptf)*cp.cosh(yf)*cp.cosh(yf))))*(Energy/Energy_i)   # E/Ei 임을 명심하자.
    except:
        print("Error in Function_gpu, Ridge_dist")
        exit(0)

# 0712.3282 (10)
def intergralNjet(pt, eta, phi, Njet, Tjet, ma, m, szero):   #P_T Final, Eta Final, Phi Final, ma, Pion mass, Sigma_(phi0)
    constant = Njet/(Tjet*(m+Tjet)*2*cp.pi())
    sigmaphi = (szero*szero*ma*ma)/(ma*ma+pt*pt)
    try:
        return (constant/sigmaphi)*cp.exp(((m-cp.sqrt(m*m+pt*pt))/Tjet)-((phi*phi+eta*eta)/(2*sigmaphi)))
    except:
        print("Error in Function_gpu, integralNjet")
        exit(0)