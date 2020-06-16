import numpy as np


def calcVelocity(H, T, antDist=0, H0 = 0.0):
    if len(T) != len(H):
        return None

    H = np.cumsum(np.array(H))
    T = np.array(T)

    H = np.insert(H, 0, H0)
    T = np.insert(T, 0, 0)

    D = np.sqrt((antDist/2)**2 + abs(H[1:])**2)
    T0 = H[1:]*T[1:]/D
    T0 = np.insert(T0, 0, 0)
    V = 2 * abs(H[1:] - H[:-1]) / abs(T0[1:] - T0[:-1])

    # print("antDist", antDist)
    # print("H", H)
    # print("T", T)
    # print("T0", T0)
    # print("V", V)

    return V



def calcVelocityRMS(V, H):
    if len(V) != len(H):
        return None
    # V = [0.11, 0.14, 0.15, 0.07]
    # H = [0.083, 0.067, 0.050, 0.2]
    V = np.array(V)
    H = np.array(H)
    VRMS = np.sqrt(np.cumsum(H*V)/np.cumsum(H/V))
    return VRMS



def calcTime(V, H, AntDist = 0):
    # Решение прямой задачи

    # V = [0.1, 0.15]
    # H = [4, 5]
    # AntDist = 0.1

    n = len(V)
    Vmin = np.min(V)
    pmax = 1/Vmin
    T = []

    for i in range(n):
        opt_dx = np.inf
        opt_t = -1
        opt_p = -1
        for p in np.arange(0, pmax, 0.01):
            sum_t = 0
            sum_x = 0
            for j in range(i+1):
                sum_t = sum_t + (2*H[j])/(V[j]*np.sqrt(1-(p**2)*(V[j]**2)))
                sum_x = sum_x + (2*p*H[j]*V[j]) / (np.sqrt(1 - (p ** 2) * (V[j] ** 2)))
            dx = np.abs(sum_x-AntDist)
            if dx < opt_dx:
                opt_dx = dx
                opt_t = sum_t
                opt_p = p
        T.append(opt_t)
    print(T)
    return T


def VelocityToEps(Velocity):
    c = 0.3
    return (c/Velocity)**2


def EpsToVelocity(eps):
    c = 0.3
    return c/np.sqrt(eps)


def calcAirDistance(t):
    c = 0.3
    return c*t/2





if __name__ == '__main__':
    V = [0.11, 0.14, 0.15, 0.07]
    H = [0.083, 0.067, 0.050, 0.2]
    calcVelocityRMS(V,H)

