import os
import csv
import torch
import collections
import pickle
import csv
import random
#cycle through all user item files making data in tensors along with rating 1 tensor
history_u_lists = collections.defaultdict(list)
history_v_lists = collections.defaultdict(list)
history_ur_lists = collections.defaultdict(list)
history_vr_lists = collections.defaultdict(list)

train_u=[]
train_v=[]
train_r=[]

val_u=[]
val_v=[]
val_r=[]

test_u=[]
test_v=[]
test_r=[]

posteritem=[]

tweet_to_tweeter={}
useridmap={}
tweetidmap={}


data_file = open("./gendata/social_adj_lists.pickle", 'rb')
social_adj_lists=pickle.load(data_file)


data_file = open("./gendata/useridmap.pickle", 'rb')
useridmap=pickle.load(data_file)


print("poster")
file=open('./poster_to_tweetNew.csv', 'r')
exampleReader = csv.reader(file)
row_count = sum(1 for row in exampleReader)
print(row_count)
counter=0

file=open('./poster_to_tweetNew.csv', 'r')
exampleReader = csv.reader(file)
for row in exampleReader:
    
    print(counter,"/",row_count," poster_to_tweet")
    counter=counter+1
    
    #Map ID##################################
    if int(row[0]) not in useridmap.keys():
        useridmap[int(row[0])]=len(useridmap.keys())
        userid=useridmap[int(row[0])]
    else:
        userid=useridmap[int(row[0])]

    if int(row[1]) not in tweetidmap.keys():
        tweetidmap[int(row[1])]=len(tweetidmap.keys())
        tweetid=tweetidmap[int(row[1])]
    else:
        tweetid=tweetidmap[int(row[1])]
    #Map ID##################################

    
    if (userid,tweetid) not in posteritem:
        
        tweet_to_tweeter[tweetid]=userid
        
        posteritem.append((userid,tweetid))
       
     
with open('./gendata/tweet_to_tweeter.pickle', 'wb') as handle:
    pickle.dump(tweet_to_tweeter, handle, protocol=pickle.HIGHEST_PROTOCOL)


print("finished!")


