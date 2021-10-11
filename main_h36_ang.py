import os 
from utils import h36motion as datasets
from torch.utils.data import DataLoader
from model import *
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.autograd
import torch
import numpy as np
from utils.loss_funcs import euler_error,mpjpe_error
from utils.data_utils import *
from utils.h36_ang_viz import visualize

from utils.parser import args



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)


model = Model(args.input_dim,args.input_n,
                           args.output_n,args.st_gcnn_dropout,args.joints_to_consider,args.n_tcnn_layers,args.tcnn_kernel_size,args.tcnn_dropout).to(device)


model_name='h36_ang_'+str(args.output_n)+'frames_ckpt'


print('total number of parameters of the network is: '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

def train():

  optimizer=optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-05)
  
  if args.use_scheduler:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)



  train_loss = []
  val_loss = []
  dataset = datasets.Datasets(args.data_dir,args.input_n,args.output_n,args.skip_rate, split=0)
  print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
  data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

  vald_dataset = datasets.Datasets(args.data_dir,args.input_n,args.output_n,args.skip_rate, split=1)
  print('>>> Validation dataset length: {:d}'.format(vald_dataset.__len__()))
  vald_loader = DataLoader(vald_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

  dim_used = np.array([6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28, 29, 30, 36, 37, 38, 39, 40, 41, 42,
                         43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85,
                         86])



  for epoch in range(args.n_epochs):
      running_loss=0
      n=0
      model.train()
      for cnt,batch in enumerate(data_loader): 
          batch=batch.to(device)
          batch_dim=batch.shape[0]
          n+=batch_dim
          
          sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//3,3).permute(0,3,1,2)
          sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, dim_used]



          optimizer.zero_grad() 

          sequences_predict=model(sequences_train).permute(0,1,3,2)
          
          loss=torch.mean(torch.sum(torch.abs(sequences_predict.reshape(-1,args.output_n,len(dim_used)) - sequences_gt), dim=2).view(-1))



          loss.backward()  
          if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)

          optimizer.step()
          if cnt % 200 == 0:
            print('[%d, %5d]  training loss: %.3f' %(epoch + 1, cnt + 1, loss.item()))

          
          running_loss += loss*batch_dim 

      train_loss.append(running_loss.detach().cpu()/n)     
      model .eval()
      with torch.no_grad():
          running_loss=0 
          n=0
          for cnt,batch in enumerate(vald_loader):
              batch=batch.to(device)
              batch_dim=batch.shape[0]
              n+=batch_dim
              
              sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//3,3).permute(0,3,1,2)
              sequences_gt=batch[:, args.input_n:args.input_n+args.output_n,:]

              all_joints_seq=batch.clone()[:, args.input_n:args.input_n+args.output_n,:]


              sequences_predict=model(sequences_train).permute(0,1,3,2).reshape(-1,args.output_n,len(dim_used))

              all_joints_seq[:,:,dim_used] = sequences_predict


              euler_loss=euler_error(all_joints_seq,sequences_gt)

              all_joints_seq=all_joints_seq.reshape(-1,99)
        
              sequences_gt=sequences_gt.reshape(-1,99)
        
              all_joints_seq=expmap2xyz_torch(all_joints_seq).view(-1,args.output_n,32,3)
        
              sequences_gt=expmap2xyz_torch(sequences_gt).view(-1,args.output_n,32,3)

              mpjpe_loss=mpjpe_error(all_joints_seq,sequences_gt)




              if cnt % 200 == 0:
                        print('[%d, %5d]  validation loss euler: %.3f validation loss mpjpe : %.3f' %(epoch + 1, cnt + 1, euler_loss.item(),mpjpe_loss)) 
              running_loss+=euler_loss *batch_dim
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

  model.load_state_dict(torch.load(os.path.join(args.model_path,model_name)))
  model.eval()
  accum_loss=0  
  accum_loss_mpjpe=0
  n_batches=0 # number of batches for all the sequences
  actions=define_actions(args.actions_to_consider)
  dim_used = np.array([6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28, 29, 30, 36, 37, 38, 39, 40, 41, 42,
                        43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85,
                        86])

  for action in actions:
    running_loss=0
    running_loss_mpjpe=0
    n=0
    dataset_test = datasets.Datasets(args.data_dir,args.input_n,args.output_n,args.skip_rate, split=2,actions=[action])
    print('>>> test action for sequences: {:d}'.format(dataset_test.__len__()))

    test_loader = DataLoader(dataset_test, batch_size=args.batch_size_test, shuffle=False, num_workers=0, pin_memory=True)
    for cnt,batch in enumerate(test_loader):
      with torch.no_grad():

        batch=batch.to(device)
        batch_dim=batch.shape[0]
        n+=batch_dim
        
        all_joints_seq=batch.clone()[:, args.input_n:args.input_n+args.output_n,:]

        sequences_train=batch[:, 0:args.input_n, dim_used].view(-1,args.input_n,len(dim_used)//3,3).permute(0,3,1,2)
        sequences_gt=batch[:, args.input_n:args.input_n+args.output_n, :]

        

        sequences_predict=model(sequences_train).permute(0,1,3,2).contiguous().view(-1,args.output_n,len(dim_used))

        all_joints_seq[:,:,dim_used] = sequences_predict

        loss=euler_error(all_joints_seq,sequences_gt)

        all_joints_seq=all_joints_seq.reshape(-1,99)
  
        sequences_gt=sequences_gt.reshape(-1,99)
  
        all_joints_seq=expmap2xyz_torch(all_joints_seq).view(-1,args.output_n,32,3)
  
        sequences_gt=expmap2xyz_torch(sequences_gt).view(-1,args.output_n,32,3)

        mpjpe_loss=mpjpe_error(all_joints_seq,sequences_gt)



        running_loss+=loss*batch_dim
        running_loss_mpjpe+=mpjpe_loss*batch_dim
        accum_loss+=loss*batch_dim
        accum_loss_mpjpe+=mpjpe_loss*batch_dim

    print('euler angle loss at test subject for action : '+str(action)+ ' is: '+ str(running_loss/n))
    print('mpjpe loss at test subject for action : '+str(action)+ ' is: '+ str(running_loss_mpjpe/n))
    n_batches+=n
  print('overall average loss in euler angle is: '+str(accum_loss/n_batches))
  print('overall average loss in mpjpe is: '+str(accum_loss_mpjpe/n_batches))

if __name__ == '__main__':

    if args.mode== 'train':
        train()
    elif args.mode == 'test':
        test()
    elif args.mode == 'viz':
       model.load_state_dict(torch.load(os.path.join(args.model_path,model_name)))
       model.eval()
       visualize(args.input_n,args.output_n,args.visualize_from,args.data_dir,model,device,args.n_viz,args.skip_rate,args.actions_to_consider) 



