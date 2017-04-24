# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:15:53 2017

@author: dez
"""

## Note. A good practice is to set a virtual environment
# pip install virtualenvwrapper
# get working directory
# mkvirtualenv test
# pip install graphlab-create

import os
os.getcwd()
# or os.path.dirname(os.path.realpath(__file__))
# set wd
#os.chdir('C:/Users/dez/Documents/Python-Scripts/Quick-guide-build-recommendation-engine-python')

## Load the data from the MovieLens dataset into Python

import pandas as pd

# pass in column names for each CSV and read them using pandas. 
# Column names available in the readme file

#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
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
# => there are 943 users and we have 5 features for each namely their unique ID, age, gender, occupation and the zip code they are living in.

## Ratings
print ratings.shape
ratings.head()
# => there are 100K ratings for different user and movie combinations. Notice that each rating has a timestamp associated with it.

## Items
print items.shape
items.head()
# This dataset contains attributes of the 1682 movies. There are 24 columns out of which 19 specify the genre of a particular movie. The last 19 columns are for each genre and a value of 1 denotes movie belongs to that genre and 0 otherwise.

## Divide the ratings data set into test and train data for making models. Luckily GroupLens provides pre-divided data wherein the test data has 10 ratings for each user, i.e. 9430 rows in total.
# Load the data
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
ratings_base.shape, ratings_test.shape
# Output: ((90570, 4), (9430, 4))

## Since we’ll be using GraphLab, lets convert these in SFrames, with
#train_data: the SFrame which contains the required data, 
#user_id: the column name which represents each user ID, 
#item_id: the column name which represents each item to be recommended, 
#target: the column name representing scores/ratings given by the user
import graphlab
train_data = graphlab.SFrame(ratings_base)
test_data = graphlab.SFrame(ratings_test)

## A. A Simple Popularity Model
# i.e. the one where all the users have same recommendation based on the most popular choices,using the graphlab recommender functions popularity_recommender
popularity_model = graphlab.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')

# Lets use this model to make top 5 recommendations for first 5 users and see what comes, where
#users = range(1,6) specifies user ID of first 5 users
#k=5 specifies top 5 recommendations to be given
popularity_recomm = popularity_model.recommend(users=range(1,6),k=5)
popularity_recomm.print_rows(num_rows=25)
# Note. The recommendations for all users are same – 1500,1201,1189,1122,814 in the same order. 
# Let's verify it by checking the movies with highest mean recommendations in our ratings_base data set
ratings_base.groupby(by='movie_id')['rating'].mean().sort_values(ascending=False).head(20)
# Result confirms that all the recommended movies have an average rating of 5. Our popularity system works as expected. But it is good enough?

## B. A Collaborative Filtering Model
# works in 2 steps:
#1. Find similar items by using a similarity metric
#2. For a user, recommend the items most similar to the items (s)he already likes
# This is done by making an item-item matrix in which we keep a record of the pair of items which were rated together. In this case, an item is a movie.
# Note. there are 3 types of item similarity metrics supported by graphlab:
#1. Jaccard Similarity
#2. Cosine Similarity
#3. Pearson Similarity

#Train Model
item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='pearson')

#Make Recommendations
item_sim_recomm = item_sim_model.recommend(users=range(1,6),k=5)
item_sim_recomm.print_rows(num_rows=25)
# We can see that the recommendations are different for each user. So, personalization exists. 
# But how good is this model? We need some means of evaluating a recommendation engine. 

## Evaluating Recommendation Engines
# We can use the concept of precision-recall, with 
#Recall being the ratio of items that a user likes that were actually recommended
#Precision the ratio of items that a user actually liked out of all the recommended items.
# Our aim is to maximize both precision and recall. An ideal recommender system, the one which only recommends the items which user likes (thus precision=recall=1 here), is what we should try and get as close as possible.  This is an optimal recommender and .

# Lets compare both the models we have built till now based on precision-recall characteristics
model_performance = graphlab.compare(test_data, [popularity_model, item_sim_model])
graphlab.show_comparison(model_performance,[popularity_model, item_sim_model])

## -- END --
# 2 very quick observations:
#The item similarity model is definitely better than the popularity model (by atleast 10x)
#On an absolute level, even the item similarity model appears to have a poor performance. It is far from being a useful recommendation system.

# A couple of tips:
#Try leveraging the additional context information which we have
#Explore more sophisticated algorithms like matrix factorization

# Along with GraphLab, you can also use some other open source python packages like the following: 
#Crab
#Surprise
#Python Recsys
#MRec

## Source: https://www.analyticsvidhya.com/blog/2016/06/quick-guide-build-recommendation-engine-python/