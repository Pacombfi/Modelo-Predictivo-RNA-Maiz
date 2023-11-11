import pandas as pd
from sklearn.model_selection import train_test_split
BaseDatos=pd.read_csv("Base_PCA_N.csv")
Pentre, Pprueba, Tentre, Tprueba =train_test_split(
    BaseDatos[["Comp1","Comp2","Comp3","Comp4","Comp5","Comp6"]],
    BaseDatos["Peso"],test_size=0.1)
Pentre.to_csv("P_Entre.csv",header = False, index = False)
Pprueba.to_csv("P_Prueba.csv",header = False, index = False)
Tentre.to_csv("T_Entre.csv",header = False, index = False)
Tprueba.to_csv("T_Prueba.csv",header = False, index = False)
