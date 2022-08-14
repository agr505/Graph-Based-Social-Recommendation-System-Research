import os
import csv
import torch
import collections
import pickle
import csv
import random
#cycle through all user item files making data in tensors along with rating 1 tensor




knegsamples=[5,10,25,50]
#knegsamples=[5]
for knegsample in knegsamples:
    history_u_lists = collections.defaultdict(list)
    history_v_lists = collections.defaultdict(list)
    history_ur_lists = collections.defaultdict(list)
    history_vr_lists = collections.defaultdict(list)

    pos_to_neg_interaction_dict = collections.defaultdict(list)

    posteritem=[]

    history_u=[]
    history_v=[]
    history_r=[]

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
    dir="/home/areich8/API"
    data_file = open(dir+"/gendata/useridmap.pickle", 'rb')
    useridmap=pickle.load(data_file)

    data_file = open(dir+"/gendata/tweetidmap.pickle", 'rb')
    tweetidmap=pickle.load(data_file)


    data_file = open(dir+"/gendata/social_adj_lists.pickle", 'rb')
    social_adj_lists=pickle.load(data_file)

    data_file = open(dir+"/gendata/history_u.pickle", 'rb')
    history_u=pickle.load(data_file)

    data_file = open(dir+"/gendata/history_v.pickle", 'rb')
    history_v=pickle.load(data_file)

    data_file = open(dir+"/gendata/history_r.pickle", 'rb')
    history_r=pickle.load(data_file)
    
    data_file = open(dir+"/gendata/train_u.pickle", 'rb')
    train_u=pickle.load(data_file)

    data_file = open(dir+"/gendata/train_v.pickle", 'rb')
    train_v=pickle.load(data_file)

    data_file = open(dir+"/gendata/train_r.pickle", 'rb')
    train_r=pickle.load(data_file)


    data_file = open(dir+"/gendata/test_u.pickle", 'rb')
    test_u=pickle.load(data_file)

    data_file = open(dir+"/gendata/test_v.pickle", 'rb')
    test_v=pickle.load(data_file)

    data_file = open(dir+"/gendata/test_r.pickle", 'rb')
    test_r=pickle.load(data_file)


    data_file = open(dir+"/gendata/val_u.pickle", 'rb')
    val_u=pickle.load(data_file)

    data_file = open(dir+"/gendata/val_v.pickle", 'rb')
    val_v=pickle.load(data_file)

    data_file = open(dir+"/gendata/val_r.pickle", 'rb')
    val_r=pickle.load(data_file)


    data_file = open(dir+"/gendata/history_u_lists.pickle", 'rb')
    history_u_lists=pickle.load(data_file)

    data_file = open(dir+"/gendata/history_v_lists.pickle", 'rb')
    history_v_lists=pickle.load(data_file)

    data_file = open(dir+"/gendata/history_ur_lists.pickle", 'rb')
    history_ur_lists=pickle.load(data_file)

    data_file = open(dir+"/gendata/history_vr_lists.pickle", 'rb')
    history_vr_lists=pickle.load(data_file)

    data_file = open(dir+"/gendata/tweet_to_tweeter.pickle", 'rb')
    tweet_to_tweeter=pickle.load(data_file)

    data_file = open(dir+"/gendata/allinteractions.pickle", 'rb')
    allinteractions=pickle.load(data_file)

    data_file = open(dir+"/gendata/onehop_allretweeters.pickle", 'rb')
    onehop_allretweeters=pickle.load(data_file)
    
    #data_file = open("/home/areich8/API/gendata/nullentries.pickle", 'rb')
    #nullentries=pickle.load(data_file)
    
    print("intersection ",set(zip(*(train_u,train_v))).intersection(set(zip(*(history_u,history_v) ))))
    print("intersection ",set(zip(*(val_u,val_v))).intersection(set(zip(*(history_u,history_v) ))))
    print("intersection ",set(zip(*(val_u,val_v))).intersection(set(zip(*(train_u,train_v) ))))
     
 
    #print("poster")
    #file=open('./tweeter_collection/poster_to_tweetNew.csv', 'r')
    #exampleReader = csv.reader(file)
    #row_count = sum(1 for row in exampleReader)
    #print(row_count)
    counter=0

    #file=open('./tweeter_collection/poster_to_tweetNew.csv', 'r')
    #exampleReader = csv.reader(file)

    #for row in exampleReader:
    #userids,tweetids=zip(*pos_to_neg_interaction_dict.keys())
    #CORRECT THIS, HAVE TO SAMPLE FOLLOWERS FROM TWEETERS
    jbreak=0
    pos_to_neg_interaction_dict= collections.defaultdict(list)

    
    for userid,tweetid in allinteractions.keys():
        #userid,tweetid=1673380, 109

        print(counter,"/"," knegsample ",knegsample)
        counter=counter+1
        
        #Map ID##################################
        #if int(row[0]) not in useridmap.keys():
        #    useridmap[int(row[0])]=len(useridmap.keys())
        #    userid=useridmap[int(row[0])]
        #else:
        #    userid=useridmap[int(row[0])]

        #if int(row[1]) not in tweetidmap.keys():
        #    tweetidmap[int(row[1])]=len(tweetidmap.keys())
        #    tweetid=tweetidmap[int(row[1])]
        #else:
        #    tweetid=tweetidmap[int(row[1])]
        #Map ID##################################
        
        if (userid,tweetid) not in posteritem:
            
            #NEGATIVE SAMPLING################
            
           
            tweeter_userid=tweet_to_tweeter[tweetid]

            possiblefollowers=social_adj_lists[tweeter_userid]

            if len(possiblefollowers)>0:
                followers=possiblefollowers
            else:
                followers=onehop_allretweeters
            #print("len(followers) ",len(followers))

            if len(followers)<knegsample:
                j=0
                k=0
                
                while k<knegsample:
                    print("less ",len(followers))
                    
                    j=j+1
                    #print("followers ",followers)
                    negfollower=random.choice(followers)
                  
                    if negfollower in history_v_lists[tweetid]:
                        negindex=history_v_lists[tweetid].index(negfollower)
                        if history_vr_lists[tweetid][negindex] !=1:
                            eligible =True
                        else:
                            eligible=False
                    else:
                        eligible =True
                        print("got negfollower ",negfollower)
                
                        #negfollowers.append(negfollower)
                    if eligible==True:
                        if (userid,tweetid) in zip(history_u,history_v) :
                         

                            #if tweetid not in history_u_lists[negfollower]:
                            history_u_lists[negfollower].append(tweetid)
                            history_ur_lists[negfollower].append(0)

                            #if negfollower not in history_v_lists[tweetid]:
                            history_v_lists[tweetid].append(negfollower)
                            history_vr_lists[tweetid].append(0)
                            k=k+1
                        elif (userid,tweetid) in zip(train_u,train_v) :
                          
                            pos_to_neg_interaction_dict[(userid,tweetid)].append((negfollower,tweetid))

                            #if tweetid not in history_u_lists[negfollower]:
                            history_u_lists[negfollower]=[]
                            history_ur_lists[negfollower]=[]

                            #if tweetid not in history_v_lists.keys():
                            history_v_lists[tweetid]=[]
                            history_vr_lists[tweetid]=[]
                            k=k+1
                        elif (userid,tweetid) in zip(val_u,val_v):
                            val_u.append(negfollower)
                            val_v.append(tweetid)
                            val_r.append(0)

                            #if negfollower not in history_u_lists.keys():
                            history_u_lists[negfollower]=[]
                            history_ur_lists[negfollower]=[]

                            #if tweetid not in history_v_lists.keys():
                            history_v_lists[tweetid]=[]
                            history_vr_lists[tweetid]=[]
                            k=k+1
                        elif  (userid,tweetid) in zip(test_u,test_v):
                
                            test_u.append(negfollower)
                            test_v.append(tweetid)
                            test_r.append(0)

                            #if negfollower not in history_u_lists.keys():
                            history_u_lists[negfollower]=[]
                            history_ur_lists[negfollower]=[]

                            #if tweetid not in history_v_lists.keys():
                            history_v_lists[tweetid]=[]
                            history_vr_lists[tweetid]=[]
                            k=k+1
                        else:
                            print("stpop")
                        j=0
                dicttest=pos_to_neg_interaction_dict.copy()
                pairedneglist=dicttest[(userid,tweetid)]
                if tweetid in train_v and len(pairedneglist)==knegsample:
                    print("match")
                elif tweetid in val_v or tweetid in test_v or tweetid in history_v:
                    print("match")
                else:
                    print("no match", len(dicttest[(userid,tweetid)]))
                    quit()                           
            elif len(followers)>=knegsample:
                
                j=0
                k=0
                negfollowers=[]
                while k<knegsample:
                    print("interaction of interest ",(userid,tweetid))
                   
                    
                    j=j+1
                    #print("followers ",followers)
                    negfollower=random.choice(followers)

                    if negfollower in history_v_lists[tweetid]:
                        negindex=history_v_lists[tweetid].index(negfollower)
                        if history_vr_lists[tweetid][negindex] !=1:
                            eligible =True
                        else:
                            eligible=False
                    else:
                        eligible =True
                    print("eligible ",eligible)
                    if eligible==True:
                        if negfollower not in negfollowers:
                            print("got negfollower ",negfollower)
                    
                            negfollowers.append(negfollower)
                            if (userid,tweetid) in zip(history_u,history_v) :
                         

                                #if tweetid not in history_u_lists[negfollower]:
                                history_u_lists[negfollower].append(tweetid)
                                history_ur_lists[negfollower].append(0)

                                #if negfollower not in history_v_lists[tweetid]:
                                history_v_lists[tweetid].append(negfollower)
                                history_vr_lists[tweetid].append(0)
                                k=k+1
                            elif (userid,tweetid) in zip(train_u,train_v) :
                                pos_to_neg_interaction_dict[(userid,tweetid)].append((negfollower,tweetid))

                                #if tweetid not in history_u_lists[negfollower]:
                                history_u_lists[negfollower]=[]
                                history_ur_lists[negfollower]=[]

                                #if tweetid not in history_v_lists.keys():
                                history_v_lists[tweetid]=[]
                                history_vr_lists[tweetid]=[]
                                k=k+1
                            elif (userid,tweetid) in zip(val_u,val_v):
                                val_u.append(negfollower)
                                val_v.append(tweetid)
                                val_r.append(0)

                                #if negfollower not in history_u_lists.keys():
                                history_u_lists[negfollower]=[]
                                history_ur_lists[negfollower]=[]

                                #if tweetid not in history_v_lists.keys():
                                history_v_lists[tweetid]=[]
                                history_vr_lists[tweetid]=[]
                                k=k+1
                            elif  (userid,tweetid) in zip(test_u,test_v):
                                
                                test_u.append(negfollower)
                                test_v.append(tweetid)
                                test_r.append(0)

                                #if negfollower not in history_u_lists.keys():
                                history_u_lists[negfollower]=[]
                                history_ur_lists[negfollower]=[]

                                #if tweetid not in history_v_lists.keys():
                                history_v_lists[tweetid]=[]
                                history_vr_lists[tweetid]=[]
                                k=k+1
                            else:
                                print("stpop")
                            j=0
                        if j>=200:
                            print("greater ",len(followers))
                            quit()
                            jbreak=jbreak+1
                        #    j=0
                        #    break
                dicttest=pos_to_neg_interaction_dict.copy()
                pairedneglist=dicttest[(userid,tweetid)]
                if tweetid in train_v and len(pairedneglist)==knegsample:
                    print("match")
                elif tweetid in val_v or tweetid in test_v or tweetid in history_v:
                    print("match")
                else:
                    print("no match2", len(dicttest[(userid,tweetid)]))
                    quit()
            else:
                #del pos_to_neg_interaction_dict[(userid,tweetid)]
                quit()
              
            #assert that dict list len == k   for each one
    q=0 
    nullentries=[]    
    for k,v in pos_to_neg_interaction_dict.items():
        if len(v) == 0:
            q=q+1
            nullentries.append((k,v))
    #with open(dir+'/gendata/nullentries.pickle', 'wb') as handle:
    #    pickle.dump(nullentries, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("q ",q)

    print("train_u.size ",len(train_u))
    print("val.size ",len(val_u))
    print("test.size ",len(test_u))
    print("jbreak ",jbreak)

    with open('./gendata/history_u'+str(knegsample)+'.pickle', 'wb') as handle:
        pickle.dump(history_u, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./gendata/history_v'+str(knegsample)+'.pickle', 'wb') as handle:
        pickle.dump(history_v, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./gendata/history_r'+str(knegsample)+'.pickle', 'wb') as handle:
        pickle.dump(history_r, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(dir+'/gendata/train_u'+str(knegsample)+'.pickle', 'wb') as handle:
        pickle.dump(train_u, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(dir+'/gendata/train_v'+str(knegsample)+'.pickle', 'wb') as handle:
        pickle.dump(train_v, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(dir+'/gendata/train_r'+str(knegsample)+'.pickle', 'wb') as handle:
        pickle.dump(train_r, handle, protocol=pickle.HIGHEST_PROTOCOL)


    with open(dir+'/gendata/val_u'+str(knegsample)+'.pickle', 'wb') as handle:
        pickle.dump(val_u, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(dir+'/gendata/val_v'+str(knegsample)+'.pickle', 'wb') as handle:
        pickle.dump(val_v, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(dir+'/gendata/val_r'+str(knegsample)+'.pickle', 'wb') as handle:
        pickle.dump(val_r, handle, protocol=pickle.HIGHEST_PROTOCOL)


    with open(dir+'/gendata/test_u'+str(knegsample)+'.pickle', 'wb') as handle:
        pickle.dump(test_u, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(dir+'/gendata/test_v'+str(knegsample)+'.pickle', 'wb') as handle:
        pickle.dump(test_v, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(dir+'/gendata/test_r'+str(knegsample)+'.pickle', 'wb') as handle:
        pickle.dump(test_r, handle, protocol=pickle.HIGHEST_PROTOCOL)



    with open(dir+'/gendata/history_u_lists'+str(knegsample)+'.pickle', 'wb') as handle:
        pickle.dump(history_u_lists, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(dir+'/gendata/history_v_lists'+str(knegsample)+'.pickle', 'wb') as handle:
        pickle.dump(history_v_lists, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(dir+'/gendata/history_ur_lists'+str(knegsample)+'.pickle', 'wb') as handle:
        pickle.dump(history_ur_lists, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(dir+'/gendata/history_vr_lists'+str(knegsample)+'.pickle', 'wb') as handle:
        pickle.dump(history_vr_lists, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(dir+'/gendata/pos_to_neg_interaction_dict'+str(knegsample)+'.pickle', 'wb') as handle:
        pickle.dump(pos_to_neg_interaction_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    with open(dir+'/gendata/useridmap.pickle', 'wb') as handle:
        pickle.dump(useridmap, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(dir+'/gendata/tweetidmap.pickle', 'wb') as handle:
        pickle.dump(tweetidmap, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
#file.close()
print("finished!")


