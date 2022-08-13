import os
import csv

import collections
#import pickle
import csv
import datetime as dt
import bisect 

times=[]

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


    bisect.insort(times, float(row[2]))

file.close()

file=open('./retweeter_to_tweetNew.csv', 'r')
exampleReader = csv.reader(file)
row_count = sum(1 for row in exampleReader)
print(row_count)
counter=0

file=open('./retweeter_to_tweetNew.csv', 'r')
exampleReader = csv.reader(file)
for row in exampleReader:
    
    print(counter,"/",row_count," retweeter_to_tweet")
    counter=counter+1


    bisect.insort(times, float(row[2]))

file.close()


import numpy as np
from scipy import stats

arr = np.array(times)
std=np.std(arr)

m=np.mean(arr)

result=stats.norm.ppf(0.7,loc=m,scale=std)
cdfvalue=stats.norm.cdf(result,loc=m,scale=std)

#print("Standard Deviation: ",std )
#
#print("Split Point %=0.7: ",result)
#print("CDF value: ",cdfvalue)

less=0.0
greater=0.0

for t in times:
  
    print(counter,"/",row_count," tweeter_to_tweet")
    counter=counter+1
  
    
    #bisect.insort(times2, float(row[2]))
    if t<result:
        less=less+1
    elif t>result:
        greater=greater+1


print("Standard Deviation: ",std )
print("Split Point %=0.7: ",result)
print("Mean: ",m)
print("Range: ",max(times) - min(times))
print("Max: ",max(times) )
print("Min: ",min(times) )
print("greater count: ",greater/len(times))
print("less count: ",less/len(times))
print("FINISHED")
#with open('./gendata/times.pickle', 'wb') as handle:
#    pickle.dump(times, handle, protocol=pickle.HIGHEST_PROTOCOL)