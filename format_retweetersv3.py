import os
import csv
import torch
import collections
import pickle
import csv
import datetime as dt

def negative_sample(train_u,train_v,train_r,test_u,test_v,test_r,history_u_lists,history_ur_lists,history_v_lists,history_vr_lists,social_adj_lists,pos_to_neg_interaction_dict,trainortest):
    negfollowers=[]
    eligible=False
    followers=social_adj_lists[userid]
    #print("len(followers) ",len(followers))

    if len(followers)<knegsample:
        eligible=False
                                    
    elif len(followers)>=knegsample:
        j=0
        k=0
        
        while k<knegsample:
            print("getting ")
            
            j=j+1
            #print("followers ",followers)
            negfollower=random.choice(followers)

            if negfollower not in history_v_lists[tweetid] and negfollower not in negfollowers:
                print("got negfollower ",negfollower)
        
                negfollowers.append(negfollower)

                if trainortest == 'train' :
                    train_u.append(negfollower)      
                    train_v.append(tweetid)
                    train_r.append(0)
                    pos_to_neg_interaction_dict[(userid,tweetid)].append((negfollower,tweetid))

                elif trainortest == 'test':
        
                    test_u.append(negfollower)
                    test_v.append(tweetid)
                    test_r.append(0)

                    if negfollower not in history_u_lists.keys():
                        history_u_lists[negfollower]=[]
                        history_ur_lists[negfollower]=[]

                    if tweetid not in history_v_lists.keys():
                        history_v_lists[tweetid]=[]
                        history_vr_lists[tweetid]=[]
                k=k+1
                j=0
            if j>=200:
                eligible=False
                break
        eligible=True

    return train_u,train_v,train_r,test_u,test_v,test_r,history_u_lists,history_ur_lists,history_v_lists,history_vr_lists,social_adj_lists,pos_to_neg_interaction_dict,eligible



allinteractions = collections.defaultdict(list)

history_u_lists = collections.defaultdict(list)
history_v_lists = collections.defaultdict(list)
history_ur_lists = collections.defaultdict(list)
history_vr_lists = collections.defaultdict(list)

posteritem=[]


useridmap={}
tweetidmap={}
train_u=[]
train_v=[]
train_r=[]

val_u=[]
val_v=[]
val_r=[]

test_u=[]
test_v=[]
test_r=[]
data_file = open("./gendata/tweet_to_tweeter.pickle", 'rb')
tweet_to_tweeter=pickle.load(data_file)

data_file = open("./gendata/social_adj_lists.pickle", 'rb')
social_adj_lists=pickle.load(data_file)


data_file = open("./gendata/useridmap.pickle", 'rb')
useridmap=pickle.load(data_file)
retweetercount=0

file3=open('./retweeter_to_tweetNew.csv', 'r')
reader = csv.reader(file3)
row_count = sum(1 for row in reader)
counter=0
file3=open('./retweeter_to_tweetNew.csv', 'r')
reader = csv.reader(file3)
zerocounter=0

onehop_allretweeters=[]
for row in reader:

    print(counter,"/",row_count," retweeter_to_tweet")
    counter=counter+1
    ################MAP id's
    if int(row[0]) not in useridmap.keys():
        useridmap[int(row[0])]=len(useridmap.keys())
        userid=useridmap[int(row[0])]
    else:
        userid=useridmap[int(row[0])]

    if int(row[3]) not in tweetidmap.keys():
        tweetidmap[int(row[3])]=len(tweetidmap.keys())
        tweetid=tweetidmap[int(row[3])]
    else:
        tweetid=tweetidmap[int(row[3])]

    ################MAP id's

    if (userid,tweetid) not in posteritem:

        posteritem.append((userid,tweetid))

        #tweeter_userid=tweet_to_tweeter[tweetid]

        #followers=social_adj_lists[tweeter_userid]
        
       

        if float(row[2])<1620097138.3202522 :
            #trainortest='train'
            #train_u,train_v,train_r,test_u,test_v,test_r,history_u_lists,history_ur_lists,history_v_lists,history_vr_lists,social_adj_lists,pos_to_neg_interaction_dict,eligible=negative_sample(train_u,train_v,train_r,test_u,test_v,test_r,history_u_lists,history_ur_lists,history_v_lists,history_vr_lists,social_adj_lists,pos_to_neg_interaction_dict,trainortest)
            
            #if eligible=True:

            train_u.append(userid)      
            train_v.append(tweetid)
            train_r.append(1)
            allinteractions[(userid,tweetid)]=[]

            followers=social_adj_lists[userid]
            onehop_allretweeters.extend(followers)

        elif float(row[2])>1620097138.3202522 :
            #trainortest='test'
            #train_u,train_v,train_r,test_u,test_v,test_r,history_u_lists,history_ur_lists,history_v_lists,history_vr_lists,social_adj_lists,pos_to_neg_interaction_dict,eligible=negative_sample(train_u,train_v,train_r,test_u,test_v,test_r,history_u_lists,history_ur_lists,history_v_lists,history_vr_lists,social_adj_lists,pos_to_neg_interaction_dict,trainortest)
            
            test_u.append(userid)
            test_v.append(tweetid)
            test_r.append(1)

            allinteractions[(userid,tweetid)]=[]
            #if userid not in history_u_lists.keys():
            history_u_lists[userid]=[]
            history_ur_lists[userid]=[]

            #if tweetid not in history_v_lists.keys():
            history_v_lists[tweetid]=[]
            history_vr_lists[tweetid]=[]
       

file3.close()

print("zero ",zerocounter)

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

        #if userid not in history_u_lists.keys():
        history_u_lists[userid]=[]
        history_ur_lists[userid]=[]

        #if tweetid not in history_v_lists.keys():
        history_v_lists[tweetid]=[]
        history_vr_lists[tweetid]=[]
    else:
        train_u_noval.append(userid)      
        train_v_noval.append(tweetid)
        train_r_noval.append(rating)

        #if tweetid not in history_u_lists[userid]:
        history_u_lists[userid].append(tweetid)
        history_ur_lists[userid].append(rating)

        #if userid not in history_v_lists[tweetid]:
        history_v_lists[tweetid].append(userid)
        history_vr_lists[tweetid].append(rating)

        counter=counter+1


print("# of retweeters ",len(history_u_lists.keys()))
print("# of interactions ",len(posteritem))

print("train_u.size ",len(train_u))
print("val.size ",len(val_u))
print("test.size ",len(test_u))

with open('./gendata/onehop_allretweeters.pickle', 'wb') as handle:
    pickle.dump(onehop_allretweeters, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

with open('./gendata/allinteractions.pickle', 'wb') as handle:
    pickle.dump(allinteractions, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('./gendata/useridmap.pickle', 'wb') as handle:
    pickle.dump(useridmap, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/tweetidmap.pickle', 'wb') as handle:
    pickle.dump(tweetidmap, handle, protocol=pickle.HIGHEST_PROTOCOL)


print("useridmap max",max(useridmap.values()))
print("finished!")


