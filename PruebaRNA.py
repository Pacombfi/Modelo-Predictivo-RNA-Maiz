import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargar los pesos para la prueba de la red
dfW1 = pd.read_csv("W1.csv", header=None)
dfb1 = pd.read_csv("b1.csv", header=None)
dfW2 = pd.read_csv("W2.csv", header=None)
dfb2 = pd.read_csv("b2.csv", header=None)
dfW3 = pd.read_csv("W3.csv", header=None)
dfb3 = pd.read_csv("b3.csv", header=None)
dfP = pd.read_csv("P_Prueba.csv", header=None)
dfT = pd.read_csv("T_Prueba.csv", header=None)

W1 = np.matrix(dfW1.to_numpy())
b1 = np.matrix(dfb1.to_numpy())
W2 = np.matrix(dfW2.to_numpy())
b2 = np.matrix(dfb2.to_numpy())
W3 = np.matrix(dfW3.to_numpy())
b3 = np.matrix(dfb3.to_numpy())
P = np.matrix(dfP.to_numpy()).T
T = np.matrix(dfT.to_numpy()).T

[R, Q] = np.shape(P)
S2 = np.shape(W2)[0]
S3 = np.shape(W3)[0]
A = np.zeros((S3, Q))

for q in range(Q):
    n1 = W1 * P[:, q] + b1
    a1 = np.tanh(n1)
    n2 = W2 * a1 + b2
    a2 = np.tanh(n2)
    n3 = W3 * a2 +b3
    a3 = n3
    for s in range(S3):
        A[s, q] = a3[s, 0]
A = np.matrix(A)

# ************* Coeficientes **********
T=T*(8.47-1.35)+1.35
A=A*(8.47-1.35)+1.35
Error=T-A
MAE=np.sum(np.abs(Error))/Q
SEC = Error @ Error.T
ECM = SEC / Q
RECM = math.sqrt(ECM[0,0])

#Obtención de la ecuación de regresión
MediaA = np.mean(A)
sumt=0
sumt2=0
suma=0
sumta=0

for q in range(Q):
    sumt=sumt+T[0,q]
    sumt2=sumt2+T[0,q]**2
    suma=suma+A[0,q]
    sumta=sumta+(T[0,q]*A[0,q])

L=np.matrix([[Q,sumt],[sumt,sumt2]])
B=np.matrix([[suma],[sumta]])
X=np.linalg.inv(L)@B

sec=0
stc=0
for q in range(Q):
    ap=X[0,0]+X[1,0]*T[0,q]
    sec=sec+(A[0,q]-ap)**2
    stc=stc+(A[0,q]-MediaA)**2
R2=1-(sec/stc)
R=np.sqrt(R2)
T=np.array(T)
A=np.array(A)
t=np.linspace(np.min(T),np.max(T),20)
fig=plt.figure(1,(8,6))
plt.scatter(T,A)
plt.plot(t,X[1,0]*t+X[0,0])
plt.title("Gráfica T vs A")
plt.xlabel("t, Peso Real [kg]")
plt.ylabel("a, Peso Estimado [kg]")
x=np.min(T)
y=np.max(T)-0.2
plt.text(x,y,"R2="+ str(round(R2*100,3))+" %",fontsize=12)
plt.text(x,y-0.3,"ECM="+ str(round(ECM[0,0],3))+" [kg^2]",fontsize=12)
plt.text(x,y-0.6,"RECM="+ str(round(RECM,3))+" [kg]",fontsize=12)
plt.text(x,y-0.9,"MAE="+ str(round(MAE,3))+" [kg]",fontsize=12)
plt.text(x,y-1.2,"a="+ str(round(X[1,0],3))+"t +"+str(round(X[0,0],3)),fontsize=12)
plt.show()

print("ECM= ",round(ECM[0,0],5))
print("RECM= ",round(RECM,5))
print("a= ",round(X[1,0],3),"t +",round(X[0,0],3))
print("R2= ",round(R2,3))
print("R= ", round(R,3))


