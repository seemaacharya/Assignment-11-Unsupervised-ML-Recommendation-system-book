# -*- coding: utf-8 -*-
"""
Created on Sat May 29 20:23:44 2021

@author: DELL
"""
#Importing the libraries
import pandas as pd
import numpy as np

#loading the dataset
book_data = pd.read_csv("book.csv",encoding='ISO-8859-1')
book_data.columns
book_data.columns=['sr_no.','user_id','book_title','book_rating']
book_data.head()
len(book_data.user_id.unique())
len(book_data.book_title.unique())

#matrix
book_data_matrix = book_data.pivot_table(index='user_id',columns='book_title',values='book_rating').reset_index(drop=True)
book_data_matrix

book_data_matrix.index = book_data.user_id.unique()
book_data_matrix.fillna(0,inplace=True)

#pairwise distances and cosine,correlation
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation
user_similarity=1-pairwise_distances(book_data_matrix.values,metric='cosine')
user_similarity

#store the results in a dataframe
user_similarity_df=pd.DataFrame(user_similarity)

#set the index and column names to user ids
user_similarity_df_index=book_data.user_id.unique()
user_similarity_df_columns=book_data.user_id.unique()

user_similarity_df.iloc[0:5,0:5]
np.fill_diagonal(user_similarity,0)
user_similarity_df.iloc[0:5,0:5]

#most similar user
user_similarity_df.idxmax(axis=1)[0:5]

#recommend the books for user==276726
user1=book_data[book_data["user_id"]==276726]
user1.book_title










