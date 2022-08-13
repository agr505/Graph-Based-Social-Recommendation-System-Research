import os
import csv
import torch
import collections
import pickle
import csv

useridmap={}
social_adj_lists = collections.defaultdict(list)
file5=open('./user_to_followerNew.csv', 'r')
reader = csv.reader(file5)
row_count = sum(1 for row in reader)
print(row_count)
counter=0
file5=open('./user_to_followerNew.csv', 'r')
reader = csv.reader(file5)
for row in reader:
    print(counter,"/",row_count," user_to_follower")
    counter=counter+1


    if int(row[0]) not in useridmap.keys():
        useridmap[int(row[0])]=len(useridmap.keys())
        userid=useridmap[int(row[0])]
    else:
        userid=useridmap[int(row[0])]

    if int(row[1]) not in useridmap.keys():
        useridmap[int(row[1])]=len(useridmap.keys())
        followerid=useridmap[int(row[1])]
    else:
        followerid=useridmap[int(row[1])]


    if followerid not in social_adj_lists[userid]:
        social_adj_lists[userid].append(followerid)
    
   
file5.close()
with open("social.csv", "a") as social:
    social.write("{}\n".format(social_adj_lists))

print("useridmap max",max(useridmap.values())) 


print("social_adj_lists.size ",len(social_adj_lists.keys()))
with open('./gendata/social_adj_lists.pickle', 'wb') as handle:
    pickle.dump(social_adj_lists, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./gendata/useridmap.pickle', 'wb') as handle:
    pickle.dump(useridmap, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Finished!")