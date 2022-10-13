# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:29:18 2022

@author: zelih
"""
import numpy as np
import pandas as pd
from sklearn import tree

df=pd.read_csv("DecisionTreesClassificationDataSet.csv")
print(df)

duzeltme_mapping={"Y":1,"N":0}
df["IseAlindi"]=df["IseAlindi"].map(duzeltme_mapping)
df["SuanCalisiyor?"]=df["SuanCalisiyor?"].map(duzeltme_mapping)
df["Top10 Universite?"]=df["Top10 Universite?"].map(duzeltme_mapping)
df["StajBizdeYaptimi?"]=df["StajBizdeYaptimi?"].map(duzeltme_mapping)
duzeltme_mapping_egitim={"BS":0,"MS":1,"PhD":2}
df["Egitim Seviyesi"]=df["Egitim Seviyesi"].map(duzeltme_mapping_egitim)
print(df.head())

y=df["IseAlindi"]
X=df.drop(["IseAlindi"],axis=1)
print(X.head())


#Decision Tree Oluşturma
clf=tree.DecisionTreeClassifier()
clf=clf.fit(X,y)
print("İşe Alındı mı",clf.predict([[5,1,3,0,1,0]]))

print("İşe Alındı mı",clf.predict([[2,0,7,0,1,0]]))
