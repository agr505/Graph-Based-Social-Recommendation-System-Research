import os
import csv
import torch
import collections
import pickle
import csv
import random

knegsamples=[5,10,25,50]
for knegsample in knegsamples:
    history_u_lists = collections.defaultdict(list)
    history_v_lists = collections.defaultdict(list)

    useridmap={}
    tweetidmap={}




    data_file = open('./gendata/social_adj_lists.pickle', 'rb')
    social_adj_lists=pickle.load(data_file)

    data_file = open('./gendata/history_u_lists'+str(knegsample)+'.pickle', 'rb')
    history_u_lists=pickle.load(data_file)




    data_file = open('./gendata/useridmap.pickle', 'rb')
    useridmap=pickle.load(data_file)



    print("poster")
    file=open('./user_to_followerNew.csv', 'r')
    exampleReader = csv.reader(file)
    row_count = sum(1 for row in exampleReader)
    print(row_count)
    counter=0

    file=open('./user_to_followerNew.csv', 'r')
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

        if int(row[1]) not in useridmap.keys():
            useridmap[int(row[1])]=len(useridmap.keys())
            follid=useridmap[int(row[1])]
        else:
            follid=useridmap[int(row[1])]
        #Map ID##################################


        if userid not in history_u_lists.keys():
            history_u_lists[userid]=[]
        if follid not in history_u_lists.keys():
            history_u_lists[follid]=[] 
            
    
    file.close()



    print("history_u_lists.size ",len(history_u_lists.keys()))
    print("useridmap max",max(useridmap.values())) 

    with open('./gendata/history_u_lists'+str(knegsample)+'.pickle', 'wb') as handle:
        pickle.dump(history_u_lists, handle, protocol=pickle.HIGHEST_PROTOCOL)



    with open('./gendata/useridmap.pickle', 'wb') as handle:
        pickle.dump(useridmap, handle, protocol=pickle.HIGHEST_PROTOCOL)



print("finished!")
