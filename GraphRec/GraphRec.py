import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import f1_score
from math import sqrt
import datetime
import argparse
import os
from torchmetrics import F1Score
from ray import tune

"""
GraphRec: Graph Neural Networks for Social Recommendation. 
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]

If you use this code, please cite our paper:
```
@inproceedings{fan2019graph,
  title={Graph Neural Networks for Social Recommendation},
  author={Fan, Wenqi and Ma, Yao and Li, Qing and He, Yuan and Zhao, Eric and Tang, Jiliang and Yin, Dawei},
  booktitle={WWW},
  year={2019}
}
```

"""


class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e,negweight,dropout):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim
        self.neg_weight=negweight
        self.num_neg_samples=5
        self.drop_out=dropout

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.BCELoss()
        self.sig= nn.Sigmoid()
        self.logsig= nn.LogSigmoid()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.prob_dist0=[]
        self.prob_dist1=[]

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u,p=self.drop_out, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v,p=self.drop_out,training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x,p=self.drop_out, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x,p=self.drop_out, training=self.training)
        scores = self.w_uv3(x)
       
        scores=scores.squeeze()
        sigpreds=self.sig(scores)
            
        preds=torch.round(sigpreds)
                   
        return scores,preds


    def loss(self, nodes_u, nodes_v, labels_list,pos_to_neg_interaction_dict):
        #print("nodes_u", nodes_u.size())
        
        posscores,preds = self.forward(nodes_u, nodes_v)
        #turn list of 2d tuples into 2 vectors
        #print("Len ",pos_to_neg_interaction_dict.values())
        negunodes=[]
        negvnodes=[]
        q=0

        #for v in pos_to_neg_interaction_dict.values():
        #    if len(v) == 0:
        #        q=q+1
        #print("q ",q)
        #quit()
        for n,v in zip(nodes_u.tolist(), nodes_v.tolist()):
            #print("n v ",(n,v))
            #print(pos_to_neg_interaction_dict[(n,v)])
            #quit()
            negtuplelist=pos_to_neg_interaction_dict[(n,v)]
            if len(negtuplelist) == 0:
                print("TupleA :",negtuplelist)
            else:
                noutput1,noutput2=zip(*negtuplelist)
                negunodes.extend(noutput1)
                negvnodes.extend(noutput2)
            #    print("TupleB :",negtuplelist)
            

        #convert to tensors or array
        negunodes_t=torch.tensor(negunodes)
        negvnodes_t=torch.tensor(negvnodes)
        negscores,preds_neg = self.forward(negunodes_t, negvnodes_t)

            
        pos_loss = self.bce_loss(posscores, torch.ones_like(posscores))/posscores.shape[0]
        neg_loss = self.bce_loss(negscores, torch.zeros_like(negscores))/posscores.shape[0]
        loss = pos_loss + self.neg_weight*neg_loss
        #print(loss)
        return loss

           
def train1(model, device, train_loader, optimizer, epoch, best_f1score):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
    
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()

        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best f1score: %.6f' % (
                epoch, i, running_loss / 100, best_f1score))
            running_loss = 0.0
    return 0


def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            scores,val_output = model.forward(test_u, test_v)
            #val_output=torch.LongTensor(val_output2.cpu().numpy())
            #with open("log.txt", "a") as logg:
            #    logg.write("val_output tensor{}\n".format(val_output))
            #    logg.write("val_output {}\n".format(val_output.shape))

            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))

    #expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    #mae = mean_absolute_error(tmp_pred, target)
    #print("tmp_pred ",tmp_pred)
    #print("target ",target)
    modelscore=f1_score(target, tmp_pred)
    return modelscore

def train_graprec(config):
    D=os.getcwd()
    print(D)
    #with open("DIRLOG.csv", "a") as dirlog:
    #    dirlog.write("{}\n".format(D))
    #!workingdir="/nethome/areich8/model/GraphRec-WWW19/gendata/"

    workingdir=D+'/gendata/'

    #!knegsample=config["knegsample"]
    knegsample=5
    data_file = open(workingdir+'train_u'+str(knegsample)+'.pickle', 'rb')
    train_u=pickle.load(data_file)
    #print("FAILLLLLLLLLLLLLLLLLLLLL")
    data_file = open(workingdir+'train_v'+str(knegsample)+'.pickle', 'rb')
    train_v=pickle.load(data_file)

    data_file = open(workingdir+'train_r'+str(knegsample)+'.pickle', 'rb')
    train_r=pickle.load(data_file)

    data_file = open(workingdir+'val_u'+str(knegsample)+'.pickle', 'rb')
    val_u=pickle.load(data_file)

    data_file = open(workingdir+'val_v'+str(knegsample)+'.pickle', 'rb')
    val_v=pickle.load(data_file)

    data_file = open(workingdir+'val_r'+str(knegsample)+'.pickle', 'rb')
    val_r=pickle.load(data_file)

    data_file = open(workingdir+'test_u'+str(knegsample)+'.pickle', 'rb')
    test_u=pickle.load(data_file)

    data_file = open(workingdir+'test_v'+str(knegsample)+'.pickle', 'rb')
    test_v=pickle.load(data_file)

    data_file = open(workingdir+'test_r'+str(knegsample)+'.pickle', 'rb')
    test_r=pickle.load(data_file)

    data_file = open(workingdir+'social_adj_lists.pickle', 'rb')
    social_adj_lists=pickle.load(data_file)


    data_file = open(workingdir+'history_u_lists'+str(knegsample)+'.pickle', 'rb')
    history_u_lists=pickle.load(data_file)

    data_file = open(workingdir+'history_v_lists'+str(knegsample)+'.pickle', 'rb')
    history_v_lists=pickle.load(data_file)

    data_file = open(workingdir+'history_ur_lists'+str(knegsample)+'.pickle', 'rb')
    history_ur_lists=pickle.load(data_file)

    data_file = open(workingdir+'history_vr_lists'+str(knegsample)+'.pickle', 'rb')
    history_vr_lists=pickle.load(data_file)

    data_file = open(workingdir+'pos_to_neg_interaction_dict'+str(knegsample)+'.pickle', 'rb')
    pos_to_neg_interaction_dict=pickle.load(data_file)

    ratings_list=[0,1]
    print("total train_u ",len(train_u))
    print("total train_v ",len(train_v))
    print("total train_r ",len(train_r))

    print("test_u ",len(test_u))
    print("test_v ",len(test_v))
    print("test_r ",len(test_r))

    train_u=torch.tensor(train_u)
    train_v=torch.tensor(train_v)
    train_r=torch.tensor(train_r)   

    val_u=torch.tensor(val_u)
    val_v=torch.tensor(val_v)
    val_r=torch.tensor(val_r)

    test_u=torch.tensor(test_u)
    test_v=torch.tensor(test_v)
    test_r=torch.tensor(test_r)

   
    args=config["args"]
    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                            train_r)
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                            test_r)

    valset = torch.utils.data.TensorDataset(torch.LongTensor(val_u), torch.LongTensor(val_v),val_r)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)

    
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch_size, shuffle=True)

    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()
    data_file = open(workingdir+'useridmap.pickle', 'rb')
    useridmap=pickle.load(data_file)
    print("num_users ",num_users)
    print("useridmap max",max(useridmap.values()))
    print("num ur ",history_ur_lists.__len__()) 
    
    print("num_items ",num_items)
    #quit()
    embed_dim = args.embed_dim
    device = config["device"]
    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)

    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True)
    # neighobrs
    agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, cuda=device)
    enc_u = Social_Encoder(lambda nodes: enc_u_history(nodes).t(), embed_dim, social_adj_lists, agg_u_social,
                        base_model=enc_u_history, cuda=device)

    # item feature: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)

    # model
    #!graphrec = GraphRec(enc_u, enc_v_history, r2e,config["negweight"],config["dropout"]).to(device)
    #!optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=config["lr"], alpha=0.9)
    graphrec = GraphRec(enc_u, enc_v_history, r2e,1,0.5).to(device)
    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=0.01, alpha=0.9)

    #torch.save(graphrec.state_dict(),'model/graphrec.pt')

    best_f1score=0
    endure_count = 0

    for epoch in range(1, args.epochs + 1):
        
        print("Epoch ",epoch)
    
        graphrec.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
        
            batch_nodes_u, batch_nodes_v, labels_list = data
            optimizer.zero_grad()
    
            loss = graphrec.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device),pos_to_neg_interaction_dict)
            loss.backward(retain_graph=True)
            print("loss.item() ", loss.item())
            optimizer.step()
            running_loss =running_loss+ loss.item()
            #print("after")
            if i % 100 == 0:
                print('[%d, %5d] loss: %.3f, The best f1score: %.6f' % (
                    epoch, i, running_loss , best_f1score))
                running_loss = 0.0


        f1score = test(graphrec, device, val_loader)
        # please add the validation set to tune the hyper-parameters based on your datasets.

        # early stopping (no validation set in toy dataset)
        print("f1 shape ",f1score)
        tune.report(rayf1score=f1score)
        if f1score>best_f1score:
            best_f1score=f1score
            
            endure_count = 0
            #torch.save({
            #'epoch': epoch,
            #'model_state_dict': graphrec.state_dict(),
            #'optimizer_state_dict': optimizer.state_dict()
            
            #}, 'model/graphrec.pt')
        else:
            endure_count += 1
        print("f1score: %.4f " % (f1score))

        if endure_count > 5:
            break
def main():
    # Training settings
    import gc

    gc.collect()

    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--device', type=str, default='cuda:0', metavar='N', help='')
    args1 = parser.parse_args()

    #os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device(args1.device if use_cuda else "cpu")
    #device="cpu"
    embed_dim = args1.embed_dim
   
    #ray.init("--temp-dir={your temp path}")
    train_graprec(config={"lr": tune.grid_search([0.001,0.0001,0.01]),"negweight":tune.grid_search([1,0.5,0.2,0.1,0.01]),"dropout":tune.grid_search([0.5]), "knegsample":tune.grid_search([5]),"args":args1,"device":device})#,local_dir="./raytune/")
    #!analysis = tune.run(train_graprec, resources_per_trial={"gpu": 4},config={"lr": tune.grid_search([0.001,0.0001,0.01]),"negweight":tune.grid_search([1,0.5,0.2,0.1,0.01]),"dropout":tune.grid_search([0.5]), "knegsample":tune.grid_search([5,10,25,50]),"args":args1,"device":device},local_dir="./raytune/")

    print("Best config: ", analysis.get_best_config(metric="rayf1score"))


if __name__ == "__main__":
    main()
