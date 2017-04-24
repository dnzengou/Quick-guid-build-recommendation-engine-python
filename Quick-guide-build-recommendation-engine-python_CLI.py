# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:15:53 2017

@author: dez
"""

import os
os.getcwd()
#os.chdir('C:/Users/dez/Documents/Python-Scripts/Quick-guide-build-recommendation-engine-python')

## Load the data from the MovieLens dataset into Python

import pandas as pd

#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp'],
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,
 encoding='latin-1')

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')

## Take a peak into the content of each file
print users.shape
users.head()

## Ratings

print ratings.shape
ratings.head()

## Items

print items.shape
items.head()

## Divide the ratings data set into test and train data for making models. 

# Load the data
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
ratings_base.shape, ratings_test.shape
# Output: ((90570, 4), (9430, 4))

## Since we’ll be using GraphLab, lets convert these in SFrames

import graphlab
train_data = graphlab.SFrame(ratings_base)
test_data = graphlab.SFrame(ratings_test)

## A. A Simple Popularity Model

popularity_model = graphlab.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')

popularity_recomm = popularity_model.recommend(users=range(1,6),k=5)
popularity_recomm.print_rows(num_rows=25)

ratings_base.groupby(by='movie_id')['rating'].mean().sort_values(ascending=False).head(20)

## B. A Collaborative Filtering Model

#Train Model
item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='pearson')

#Make Recommendations
item_sim_recomm = item_sim_model.recommend(users=range(1,6),k=5)
item_sim_recomm.print_rows(num_rows=25)

## Evaluating Recommendation Engines

# Lets compare both the models we have built till now based on precision-recall characteristics
model_performance = graphlab.compare(test_data, [popularity_model, item_sim_model])
graphlab.show_comparison(model_performance,[popularity_model, item_sim_model])

## -- END --