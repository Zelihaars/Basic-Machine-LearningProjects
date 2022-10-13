# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 09:39:05 2022

@author: zelih
"""
import numpy as np
import pandas as pd

column_names=["user_id","item_id","rating","timestamp"]
df=pd.read_csv("users.data",sep="\t",names=column_names)
print(df.head())

movie_titles=pd.read_csv("movie_id_titles.csv")
print(len(movie_titles))

df=pd.merge(df,movie_titles,on="item_id")
print(df.head())

moviemat=df.pivot_table(index="user_id",columns="title",values="rating")
print(moviemat.head())

starwars_ratings=moviemat["Star Wars (1977)"]
print(starwars_ratings.head())

similar_to_starwars=moviemat.corrwith(starwars_ratings)
print(similar_to_starwars)

corr_starwars=pd.DataFrame(similar_to_starwars,columns=["Correlation"])
corr_starwars.dropna(inplace=True)
print(corr_starwars.head())
print(corr_starwars.sort_values("Correlation",ascending=False).head(10))

df.drop(["timestamp"],axis=1)

ratings=pd.DataFrame(df.groupby("title")["rating"].mean())
print(ratings.sort_values("rating",ascending=False).head())

ratings["rating_oy_sayisi"]=pd.DataFrame(df.groupby("title")["rating"].count())
print(ratings.head())

print(ratings.sort_values("rating_oy_sayisi",ascending=False).head())

print(corr_starwars.head())

corr_starwars=corr_starwars.join(ratings["rating_oy_sayisi"])
print(corr_starwars.head())

print(corr_starwars[corr_starwars["rating_oy_sayisi"]>100].sort_values("Correlation",ascending=False).head())





















