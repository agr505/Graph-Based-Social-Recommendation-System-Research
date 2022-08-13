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


useridmap={}
tweetidmap={}


data_file = open("./gendata/social_adj_lists.pickle", 'rb')
social_adj_lists=pickle.load(data_file)


data_file = open("./gendata/useridmap.pickle", 'rb')
useridmap=pickle.load(data_file)


print("poster")
file=open('./tweeter_collection/poster_to_tweetNew.csv', 'r')
exampleReader = csv.reader(file)
row_count = sum(1 for row in exampleReader)
print(row_count)
counter=0

file=open('./tweeter_collection/poster_to_tweetNew.csv', 'r')
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
        
        
        posteritem.append((userid,tweetid))
       
     
        if float(row[2])<1620097138.3202522 :
            train_u.append(userid)      
            train_v.append(tweetid)
            train_r.append(1)


        elif float(row[2])>1620097138.3202522 :
            
            test_u.append(userid)
            test_v.append(tweetid)
            test_r.append(1)

            if userid not in history_u_lists.keys():
                history_u_lists[userid]=[]
                history_ur_lists[userid]=[]

            if tweetid not in history_v_lists.keys():
                history_v_lists[tweetid]=[]
                history_vr_lists[tweetid]=[]
file.close()



#####Create Validation Set
trainsize=len(train_u)-round(len(train_u)*0.1)
#print("len(train_u) ", len(train_u))
#print("valsize ", valsize)

train_u_noval=[]
train_v_noval=[]
train_r_noval=[]
val_u=[]
val_v=[]
val_r=[]
posteritem_val=[]
counter=0
for userid,tweetid,rating in  zip(train_u,train_v,train_r):
    
    if counter==trainsize:
        posteritem_val.append((userid,tweetid))
        val_u.append(userid)  
        val_v.append(tweetid)
        val_r.append(rating)

        if userid not in history_u_lists.keys():
            history_u_lists[userid]=[]
            history_ur_lists[userid]=[]

        if tweetid not in history_v_lists.keys():
            history_v_lists[tweetid]=[]
            history_vr_lists[tweetid]=[]
    else:
        train_u_noval.append(userid)      
        train_v_noval.append(tweetid)
        train_r_noval.append(rating)

        if tweetid not in history_u_lists[userid]:
            history_u_lists[userid].append(tweetid)
            history_ur_lists[userid].append(1)

        if userid not in history_v_lists[tweetid]:
            history_v_lists[tweetid].append(userid)
            history_vr_lists[tweetid].append(1)

        counter=counter+1


print("# of tweeters ",len(history_u_lists.keys()))
print("# of interactions ",len(posteritem))

print("train_u.size ",len(train_u))
print("val.size ",len(val_u))
print("test.size ",len(test_u))

with open('./gendata/train_u.pickle', 'wb') as handle:
    pickle.dump(train_u_noval, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/train_v.pickle', 'wb') as handle:
    pickle.dump(train_v_noval, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/train_r.pickle', 'wb') as handle:
    pickle.dump(train_r_noval, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/val_u.pickle', 'wb') as handle:
    pickle.dump(val_u, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/val_v.pickle', 'wb') as handle:
    pickle.dump(val_v, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/val_r.pickle', 'wb') as handle:
    pickle.dump(val_r, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('./gendata/test_u.pickle', 'wb') as handle:
    pickle.dump(test_u, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/test_v.pickle', 'wb') as handle:
    pickle.dump(test_v, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/test_r.pickle', 'wb') as handle:
    pickle.dump(test_r, handle, protocol=pickle.HIGHEST_PROTOCOL)



with open('./gendata/history_u_lists.pickle', 'wb') as handle:
    pickle.dump(history_u_lists, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/history_v_lists.pickle', 'wb') as handle:
    pickle.dump(history_v_lists, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/history_ur_lists.pickle', 'wb') as handle:
    pickle.dump(history_ur_lists, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/history_vr_lists.pickle', 'wb') as handle:
    pickle.dump(history_vr_lists, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('./gendata/useridmap.pickle', 'wb') as handle:
    pickle.dump(useridmap, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/tweetidmap.pickle', 'wb') as handle:
    pickle.dump(tweetidmap, handle, protocol=pickle.HIGHEST_PROTOCOL)


print("finished!")


