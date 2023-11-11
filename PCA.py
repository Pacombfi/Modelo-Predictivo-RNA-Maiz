from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

dfBase_VarEntradas=pd.read_csv("Base_Mod_C.csv")
pca=PCA().fit(dfBase_VarEntradas)
varianza=pca.explained_variance_ratio_
sumVar=np.cumsum(varianza)
plt.plot(sumVar)
plt.title("Varianza explicada")
plt.xlabel("Componentes")
plt.ylabel("Suma Varianza")
plt.show()

componentes=6
pca_pesos=PCA(n_components=6).fit(dfBase_VarEntradas)
pca_comp=PCA(n_components=componentes).fit_transform(dfBase_VarEntradas)
pca_pesos.components_.T
dfPesosPCA=pd.DataFrame(pca_pesos.components_.T,index=dfBase_VarEntradas.columns,
                        columns=[["Comp1", "Comp2","Comp3","Comp4","Comp5","Comp6"]])
dfBasePCA=pd.DataFrame(pca_comp,
                       columns=[["Comp1", "Comp2","Comp3","Comp4","Comp5","Comp6"]])

dfPesosPCA.to_csv("Pesos_PCA.csv")
dfBasePCA.to_csv("Base_PCA.csv")