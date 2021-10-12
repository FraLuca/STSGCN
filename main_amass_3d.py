import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.autograd
import matplotlib.pyplot as plt
from model import *
from utils.ang2joint import *
from utils.loss_funcs import mpjpe_error
from utils.amass_3d import *
#from utils.dpw3d import * # choose amass or 3dpw by importing the right dataset class
from utils.amass_3d_viz import visualize
from utils.parser import args




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using device: %s'%device)





model = Model(args.input_dim,args.input_n,
                           args.output_n,args.st_gcnn_dropout,args.joints_to_consider,args.n_tcnn_layers,args.tcnn_kernel_size,args.tcnn_dropout).to(device)





model_name='amass_3d_'+str(args.output_n)+'frames_ckpt'


print('total number of parameters of the network is: '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))


def train():

        optimizer=optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-05)
        
        if args.use_scheduler:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)


        train_loss = []
        val_loss = []
        model.train()
        Dataset = Datasets(args.data_dir,args.input_n,args.output_n,args.skip_rate,split=0)
        loader_train = DataLoader(
            Dataset,
            batch_size=args.batch_size,
            shuffle = True,
            num_workers=0)    

        Dataset_val = Datasets(args.data_dir,args.input_n,args.output_n,args.skip_rate,split=1)

        loader_val = DataLoader(
            Dataset_val,
            batch_size=args.batch_size,
            shuffle = True,
            num_workers=0)                          
        joint_used=np.arange(4,22)
        

        for epoch in range(args.n_epochs):
            running_loss=0
            n=0
            model.train()
            for cnt,batch in enumerate(loader_train): 
                    batch = batch.float().to(device)[:, :, joint_used] # multiply by 1000 for milimeters
                    batch_dim=batch.shape[0]
                    n+=batch_dim
                    
                    sequences_train=batch[:,0:args.input_n,:,:].permute(0,3,1,2)
                    sequences_predict_gt=batch[:,args.input_n:args.input_n+args.output_n,:,:]
                    optimizer.zero_grad()
                    sequences_predict=model(sequences_train)
                    loss=mpjpe_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000# # both must have format (batch,T,V,C)
                    if cnt % 200 == 0:
                        print('[%d, %5d]  training loss: %.3f' %(epoch + 1, cnt + 1, loss.item()))             
                    loss.backward()
                    if args.clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)
                    optimizer.step()
                    running_loss += loss*batch_dim
            train_loss.append(running_loss.detach().cpu()/n)
            model .eval()
            with torch.no_grad():
                running_loss=0
                n=0
                for cnt,batch in enumerate(loader_val): 
                    batch = batch.float().to(device)[:, :, joint_used]
                    batch_dim=batch.shape[0]
                    n+=batch_dim
                    
                    sequences_train=batch[:,0:args.input_n,:,:].permute(0,3,1,2)
                    sequences_predict_gt=batch[:,args.input_n:args.input_n+args.output_n,:,:]
                    sequences_predict=model(sequences_train)
                    loss=mpjpe_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000 # the inputs to the loss function must have shape[N,T,V,C]
                    if cnt % 200 == 0:
                        print('[%d, %5d]  validation loss: %.3f' %(epoch + 1, cnt + 1, loss.item()))                    
                    running_loss+=loss*batch_dim
                val_loss.append(running_loss.detach().cpu()/n)
                if args.use_scheduler:
                    scheduler.step()

            if (epoch+1)%10==0:
                print('----saving model-----')
                torch.save(model.state_dict(),os.path.join(args.model_path,model_name))


        
                plt.figure(1)
                plt.plot(train_loss, 'r', label='Train loss')
                plt.plot(val_loss, 'g', label='Val loss')
                plt.legend()
                plt.show()


def test():


        print('Test mode')
        model.load_state_dict(torch.load(os.path.join(args.model_path,model_name)))
        model.eval()
        accum_loss=0  
        n=0
        Dataset = Datasets(args.data_dir,args.input_n,args.output_n,args.skip_rate,split=2)
        loader_test = DataLoader(
            Dataset,
            batch_size=args.batch_size,
            shuffle =False,
            num_workers=0)
        joint_used=np.arange(4,22)
        full_joint_used=np.arange(0,22) # needed for visualization
        with torch.no_grad():
            for cnt,batch in enumerate(loader_test): 
                batch = batch.float().to(device)
                batch_dim=batch.shape[0]
                n+=batch_dim
                
                sequences_train=batch[:,0:args.input_n,joint_used,:].permute(0,3,1,2)

                sequences_predict_gt=batch[:,args.input_n:args.input_n+args.output_n,full_joint_used,:]
                
                sequences_predict=model(sequences_train).permute(0,1,3,2)
                
                
                all_joints_seq=sequences_predict_gt.clone()

                all_joints_seq[:,:,joint_used,:]=sequences_predict

                loss=mpjpe_error(all_joints_seq,sequences_predict_gt)*1000 # loss in milimeters
                accum_loss+=loss*batch_dim
        print('overall average loss in mm is: '+str(accum_loss/n))
if __name__ == '__main__':

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    elif args.mode == 'viz':
        model.load_state_dict(torch.load(os.path.join(args.model_path,model_name)))
        model.eval()
        visualize(args.input_n,args.output_n,args.visualize_from,args.data_dir,model,device,args.n_viz,args.skip_rate)


