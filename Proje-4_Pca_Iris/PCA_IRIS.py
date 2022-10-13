# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:35:46 2022

@author: zelih
"""
#Principal Component Analysis

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler #Verileri aynı scaler içine getirmek için
from sklearn.decomposition import PCA

url="pca_iris.data"
df=pd.read_csv(url,names=["Sepal length","Sepal Width","Petal Length","Petal Width","Target"])
print(df)

features=["Sepal length","Sepal Width","Petal Length","Petal Width"]
x=df[features]
y=df[["Target"]]

x=StandardScaler().fit_transform(x)

#4Boyuttan 2 boyuta indirme
pca=PCA(n_components=2)
principalComponents=pca.fit_transform(x)
principalDf=pd.DataFrame(data=principalComponents,columns=["principal component 1","principal component 2"])
print(principalDf)

final_dataframe=pd.concat([principalDf,df[["Target"]]],axis=1)
print(final_dataframe.head())

dfsetosa=final_dataframe[df.Target=="Iris-setosa"]
dfvirginica=final_dataframe[df.Target=="Iris-virginica"]
dfversicolor=final_dataframe[df.Target=="Iris-versicolor"]
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")

plt.scatter(dfsetosa["principal component 1"],dfsetosa["principal component 2"],color="green")
plt.scatter(dfvirginica["principal component 1"],dfvirginica["principal component 2"],color="red")
plt.scatter(dfversicolor["principal component 1"],dfversicolor["principal component 2"],color="blue")

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())
