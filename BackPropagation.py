import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def BP(Neuronas, F_act, P, T, alfa=0.01, tol=1e-3, Epocas=10):
    """ Función que aplica el algoritmo de retropropagación del error
        neuronas= cantidad de neuronas en cada capa
        funion_act= funcion de activacion para cada capa
        T = Tangente hiperbolica
        S = Sigmoide
        L = Lineal
        P=Base de datos con los patrones de entrenamiento
        T=Base de datos con las salidas deseadas
        tasa_aprendizaje= tasa de aprendizaje del algoritmo
        error_min=error cuadrado medio minimo para detener el algoritmo"""
    R, Q = np.shape(P)
    SM, _ = np.shape(T)
    Neuronas.insert(0, R)
    Neuronas.append(SM)
    W = []
    b = []
    Fp = []
    d = []
    ECM = []
    capas = len(Neuronas) - 1
    for i in range(capas):
        W.insert(i, np.random.normal(0, 1, (Neuronas[i + 1], Neuronas[i])))
        b.insert(i, np.random.normal(0, 1, (Neuronas[i + 1], 1)))
        Fp.insert(i, np.identity(Neuronas[i + 1]))
        d.append(np.zeros((Neuronas[i + 1], 1)))
    for epo in range(Epocas):
        E = []
        for q in range(Q):
            a = []
            a.append(P[:, q])
            for m in range(capas):
                Wt = np.matrix(W[m])
                bt = np.matrix(b[m])
                at = np.matrix(a[m])
                n = Wt @ at + bt
                if F_act[m] == "T":
                    A = np.tanh(n)
                    for ns in range(Neuronas[m + 1]):
                        Fp[m][ns, ns] = 1 - A[ns, 0] ** 2
                elif F_act[m] == "S":
                    A = 1 / (1 + np.exp(-n))
                    for ns in range(Neuronas[m + 1]):
                        Fp[m][ns, ns] = A[ns, 0] * (1 - A[ns, 0])
                elif F_act[m] == "L":
                    A = n
                elif F_act[m] == "R":
                    A = np.maximum(0, n)
                    for ns in range(Neuronas[m + 1]):
                        if A[ns, 0] < 0:
                            Fp[m][ns, ns] = 0
                a.append(A)
            error = T[:, q] - a[m + 1]
            E.append(error.T @ error)
            d[m] = -2 * Fp[m] @ error
            for m in range(capas - 2, -1, -1):
                d[m] = Fp[m] @ W[m + 1].T @ d[m + 1]
            for m in range(capas):
                W[m] = W[m] - alfa * d[m] @ a[m].T
                b[m] = b[m] - alfa * d[m]
        ecm = np.sum(E) / Q
        ECM.append(ecm)
        if ecm < tol:
            break
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(ECM))
    ytol = tol * np.ones((len(ECM)))
    ax.plot(x, ECM)
    ax.plot(x, ytol)
    plt.title("ECM Retropropagación")
    plt.show()
    return (W, b)

def Momentum(Neuronas, F_act, P, T, alfa=0.01, beta=0.9, tol=1e-3, Epocas=10):
    """ Función que aplica el algoritmo de retropropagación del error
        neuronas= cantidad de neuronas en cada capa
        funion_act= funcion de activacion para cada capa
        T = Tangente hiperbolica
        S = Sigmoide
        L = Lineal
        R = RELU
        P=Base de datos con los patrones de entrenamiento
        T=Base de datos con las salidas deseadas
        tasa_aprendizaje= tasa de aprendizaje del algoritmo
        error_min=error cuadrado medio minimo para detener el algoritmo"""
    R, Q = np.shape(P)
    SM, _ = np.shape(T)
    Neuronas.insert(0, R)
    Neuronas.append(SM)
    W = []
    b = []
    Vw = []
    Vb = []
    Fp = []
    d = []
    ECM = []
    capas = len(Neuronas) - 1
    for i in range(capas):
        W.insert(i, np.random.normal(0, 1, (Neuronas[i + 1], Neuronas[i])))
        b.insert(i, np.random.normal(0, 1, (Neuronas[i + 1], 1)))
        Vw.insert(i, np.zeros((Neuronas[i + 1], Neuronas[i])))
        Vb.insert(i, np.zeros((Neuronas[i + 1], 1)))
        Fp.insert(i, np.identity(Neuronas[i + 1]))
        d.append(np.zeros((Neuronas[i + 1], 1)))
    for epo in range(Epocas):
        E = []
        for q in range(Q):
            a = []
            a.append(P[:, q])
            for m in range(capas):
                Wt = np.matrix(W[m])
                bt = np.matrix(b[m])
                at = np.matrix(a[m])
                n = Wt @ at + bt
                if F_act[m] == "T":
                    A = np.tanh(n)
                    for ns in range(Neuronas[m + 1]):
                        Fp[m][ns, ns] = 1 - A[ns, 0] ** 2
                elif F_act[m] == "S":
                    A = 1 / (1 + np.exp(-n))
                    for ns in range(Neuronas[m + 1]):
                        Fp[m][ns, ns] = A[ns, 0] * (1 - A[ns, 0])
                elif F_act[m] == "L":
                    A = n
                elif F_act[m] == "R":
                    A = np.maximum(0, n)
                    for ns in range(Neuronas[m + 1]):
                        if A[ns, 0] < 0:
                            Fp[m][ns, ns] = 0
                a.append(A)
            error = T[:, q] - a[m + 1]
            E.append(error.T @ error)
            d[m] = -2 * Fp[m] @ error
            for m in range(capas - 2, -1, -1):
                d[m] = Fp[m] @ W[m + 1].T @ d[m + 1]
            for m in range(capas):
                gW = d[m] @ a[m].T
                gb = d[m]
                Vw[m] = beta * Vw[m] + (1 - beta) * gW
                Vb[m] = beta * Vb[m] + (1- beta ) * gb
                W[m] = W[m] - alfa * Vw[m]
                b[m] = b[m] - alfa * Vb[m]
        ecm = np.sum(E) / Q
        ECM.append(ecm)
        if ecm < tol:
            break
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(ECM))
    ytol = tol * np.ones((len(ECM)))
    ax.plot(x, ECM)
    ax.plot(x, ytol)
    plt.title("ECM Momentum")
    plt.show()
    return (W, b)

def ADAM(Neuronas, F_act, P, T, alfa=0.01, beta1=0.9,beta2=0.999, tol=1e-3, Epocas=10):
    """ Función que aplica el algoritmo de retropropagación del error
        neuronas= cantidad de neuronas en cada capa
        funion_act= funcion de activacion para cada capa
        T = Tangente hiperbolica
        S = Sigmoide
        L = Lineal
        R = RELU
        P=Base de datos con los patrones de entrenamiento
        T=Base de datos con las salidas deseadas
        tasa_aprendizaje= tasa de aprendizaje del algoritmo
        error_min=error cuadrado medio minimo para detener el algoritmo"""
    R, Q = np.shape(P)
    SM, _ = np.shape(T)
    Neuronas.insert(0, R)
    Neuronas.append(SM)
    W = []
    b = []
    Mw = []
    Mb = []
    Vw = []
    Vb = []
    Fp = []
    d = []
    ECM = []
    capas = len(Neuronas) - 1
    for i in range(capas):
        W.insert(i, np.random.normal(0, 1, (Neuronas[i + 1], Neuronas[i])))
        b.insert(i, np.random.normal(0, 1, (Neuronas[i + 1], 1)))
        Mw.insert(i, np.zeros((Neuronas[i + 1], Neuronas[i])))
        Mb.insert(i, np.zeros((Neuronas[i + 1], 1)))
        Vw.insert(i, np.zeros((Neuronas[i + 1], Neuronas[i])))
        Vb.insert(i, np.zeros((Neuronas[i + 1], 1)))
        Fp.insert(i, np.identity(Neuronas[i + 1]))
        d.append(np.zeros((Neuronas[i + 1], 1)))
    for epo in range(Epocas):
        E = []
        for q in range(Q):
            a = []
            a.append(P[:, q])
            for m in range(capas):
                Wt = np.matrix(W[m])
                bt = np.matrix(b[m])
                at = np.matrix(a[m])
                n = Wt @ at + bt
                if F_act[m] == "T":
                    A = np.tanh(n)
                    for ns in range(Neuronas[m + 1]):
                        Fp[m][ns, ns] = 1 - A[ns, 0] ** 2
                elif F_act[m] == "S":
                    A = 1 / (1 + np.exp(-n))
                    for ns in range(Neuronas[m + 1]):
                        Fp[m][ns, ns] = A[ns, 0] * (1 - A[ns, 0])
                elif F_act[m] == "L":
                    A = n
                elif F_act[m] == "R":
                    A = np.maximum(0, n)
                    for ns in range(Neuronas[m + 1]):
                        if A[ns, 0] < 0:
                            Fp[m][ns, ns] = 0
                a.append(A)
            error = T[:, q] - a[m + 1]
            E.append(error.T @ error)
            d[m] = -2 * Fp[m] @ error
            for m in range(capas - 2, -1, -1):
                d[m] = Fp[m] @ W[m + 1].T @ d[m + 1]
            for m in range(capas):
                gW=d[m] @ a[m].T
                gb=d[m]
                Mw[m] = beta1*Mw[m] + (1-beta1) * gW
                Mb[m] = beta1*Mb[m] + (1-beta1) * gb
                Vw[m] = beta2*Vw[m] + (1-beta2)*np.multiply(gW,gW)
                Vb[m] = beta2*Vb[m] + (1-beta2) * np.multiply(gb,gb)
                Mwg = Mw[m] / (1 - beta1 ** epo+1)
                Mbg = Mb[m] / (1 - beta1 ** epo+1)
                Vwg = Vw[m] / (1 - beta2 ** epo+1)
                Vbg = Vb[m] / (1 - beta2 ** epo+1)
                epsilon=1e-9
                W[m] = W[m] - alfa * Mwg / np.sqrt(Vwg + epsilon)
                b[m] = b[m] - alfa * Mbg / np.sqrt(Vbg + epsilon)
        ecm = np.sum(E) / Q
        ECM.append(ecm)
        if ecm < tol:
            break
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(ECM))
    ytol = tol * np.ones((len(ECM)))
    plt.title("ECM")
    plt.xlabel("Epoca")
    plt.ylabel("Error")
    ax.plot(x, ECM)
    ax.plot(x, ytol)
    plt.title("ECM ADAM")
    plt.show()
    return (W, b)

""" Ingreso de los datos para el entrenamiento de la red """
from sklearn.model_selection import train_test_split
BaseDatos=pd.read_csv("Base_PCA_N.csv")
Pentre, Pprueba, Tentre, Tprueba =train_test_split(
    BaseDatos[["Comp1","Comp2","Comp3","Comp4","Comp5","Comp6"]],
    BaseDatos["Peso"],test_size=0.1)
Pentre.to_csv("P_Entre.csv",header = False, index = False)
Pprueba.to_csv("P_Prueba.csv",header = False, index = False)
Tentre.to_csv("T_Entre.csv",header = False, index = False)
Tprueba.to_csv("T_Prueba.csv",header = False, index = False)

dfP =pd.read_csv("P_Entre.csv",header=None)
dfT =pd.read_csv("T_Entre.csv",header=None)
P=dfP.to_numpy().T
T=dfT.to_numpy().T
P = np.matrix(P)
T = np.matrix(T)
Neuronas=[30,5]
F_act = ["T","T","L"]
alfa = 0.001
beta1= 0.9
beta2= 0.999
tol = 5e-4
Epocas = 2500

W,b=ADAM(Neuronas,F_act,P,T,alfa,beta1,beta2,tol,Epocas)
guardar=input("Quieres guardar los datos: s, n: ").upper()
if guardar=="S":
    for i in range(len(W)):
        dfW= pd.DataFrame(W[i])
        dfW.to_csv("W" + str(i + 1) + ".csv", index=False, header=None)
        dfb = pd.DataFrame(b[i])
        dfb.to_csv("b" + str(i + 1) + ".csv", index=False, header=None)