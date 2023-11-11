import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
dfMaiz=pd.read_csv("BaseCompletaRNAMaiz.csv")
MatrizCorr=dfMaiz.corr(method="pearson")
plt.figure(figsize=(20,20))
plt.title("Matriz de Correlación")
sns.heatmap(MatrizCorr,annot=True)
plt.show()