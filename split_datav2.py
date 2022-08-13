import os
import csv
import torch
import collections
import pickle
import csv
import random

train_u=[]
train_v=[]
train_r=[]
train_t=[]

test_u=[]
test_v=[]
test_r=[]
test_t=[]

history_u_lists_train = collections.defaultdict(list)
history_v_lists_train = collections.defaultdict(list)
history_ur_lists_train = collections.defaultdict(list)
history_vr_lists_train = collections.defaultdict(list)

data_file = open("./gendata/time_tweet_dict.pickle", 'rb')
time_tweet_dict=pickle.load(data_file)

data_file = open("./gendata/poster_tweet.pickle", 'rb')
poster_tweet=pickle.load(data_file)

data_file = open("./gendata/item_tweet.pickle", 'rb')
item_tweet=pickle.load(data_file)

data_file = open("./gendata/rating_tweet.pickle", 'rb')
rating_tweet=pickle.load(data_file)

data_file = open("./gendata/time_tweet.pickle", 'rb')
time_tweet=pickle.load(data_file)

data_file = open("./gendata/poster_retweet.pickle", 'rb')
poster_retweet=pickle.load(data_file)

data_file = open("./gendata/item_retweet.pickle", 'rb')
item_retweet=pickle.load(data_file)

data_file = open("./gendata/rating_retweet.pickle", 'rb')
rating_retweet=pickle.load(data_file)

data_file = open("./gendata/history_v_lists.pickle", 'rb')
history_v_lists=pickle.load(data_file)

data_file = open("./gendata/history_vr_lists.pickle", 'rb')
history_vr_lists=pickle.load(data_file)

nrcounter=0
for tweet in time_tweet_dict.keys():
    t=time_tweet_dict[tweet]

    if float(t)<1620097138.3202522 :
        for user,rating in  zip(history_v_lists[tweet],history_vr_lists[tweet]):
            train_u.insert(0,user)
            train_v.insert(0,tweet)
            train_r.insert(0,rating)     

            if tweet not in history_u_lists_train[user]:
                history_u_lists_train[user].append(tweet)
                history_ur_lists_train[user].append(rating)
            
            if user not in history_v_lists_train[tweet]:
                history_v_lists_train[tweet].append(user)  
                history_vr_lists_train[tweet].append(rating)  

    elif float(t)>1620097138.3202522 :
        for user,rating in  zip(history_v_lists[tweet],history_vr_lists[tweet]):
            test_u.insert(0,user)
            test_v.insert(0,tweet)
            test_r.insert(0,rating)

            if user not in history_u_lists_train.keys():
                history_u_lists_train[user]=[]

print("total train_u ",len(train_u))
print("total train_v ",len(train_v))
print("total train_r ",len(train_r))

print("test_u ",len(test_u))
print("test_v ",len(test_v))
print("test_r ",len(test_r))

train_u=torch.tensor(train_u)
train_v=torch.tensor(train_v)
train_r=torch.tensor(train_r)

   
test_u=torch.tensor(test_u)
test_v=torch.tensor(test_v)
test_r=torch.tensor(test_r)
            

with open('./gendata/train_u.pickle', 'wb') as handle:
    pickle.dump(train_u, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/train_v.pickle', 'wb') as handle:
    pickle.dump(train_v, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/train_r.pickle', 'wb') as handle:
    pickle.dump(train_r, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/test_u.pickle', 'wb') as handle:
    pickle.dump(test_u, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/test_v.pickle', 'wb') as handle:
    pickle.dump(test_v, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/test_r.pickle', 'wb') as handle:
    pickle.dump(test_r, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/history_u_lists_train.pickle', 'wb') as handle:
    pickle.dump(history_u_lists_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/history_v_lists_train.pickle', 'wb') as handle:
    pickle.dump(history_v_lists_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/history_ur_lists_train.pickle', 'wb') as handle:
    pickle.dump(history_ur_lists_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/history_vr_lists_train.pickle', 'wb') as handle:
    pickle.dump(history_vr_lists_train, handle, protocol=pickle.HIGHEST_PROTOCOL)