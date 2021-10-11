#!/usr/bin/env python
# coding: utf-8


import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
#from utils.amass_3d import *
from utils.dpw3d import * # choose dataset to visualize on the dataset class that we import
from utils.loss_funcs import mpjpe_error


# In[10]:


def create_pose(ax,plots,vals,pred=True,update=False):
    connect = [
        (0, 1), (0, 2), #(0, 3),
        (1, 4), (5, 2), #(3, 6),
        (7, 4), (8, 5), #(6, 9),
        (7, 10), (8, 11), #(9, 12),
      #  (12, 13), (12, 14),
        (12, 15),
        #(13, 16), (12, 16), (14, 17), (12, 17),
        (12, 16), (12, 17),
        (16, 18), (19, 17), (20, 18), (21, 19),
      #  (22, 20), (23, 21),# wrists
    (1, 16), (2, 17)]



    LR = np.array([
        False,
        True, False,
        False,
        True, False,
        False,
        True, False,
        False,
        True, False,
        False,
        True, False,
        True, True,
        False,
        True, False,
        True, False,
        True, False,
        True, False])

# Start and endpoints of our representation
    I   = np.array([touple[0] for touple in connect])
    J   = np.array([touple[1] for touple in connect])
# Left / right indicator
    LR  = np.array([LR[a] or LR[b] for a,b in connect])
    if pred:
        lcolor = "#9b59b6"
        rcolor = "#2ecc71"
    else:
        lcolor = "#8e8e8e"
        rcolor = "#383838"

    for i in np.arange( len(I) ):
        x = np.array( [vals[I[i], 0], vals[J[i], 0]] )
        z = np.array( [vals[I[i], 1], vals[J[i], 1]] )
        y = np.array( [vals[I[i], 2], vals[J[i], 2]] )
        if not update:

            if i ==0:
                plots.append(ax.plot(x, y, z, lw=2,linestyle='--' ,c=lcolor if LR[i] else rcolor,label=['GT' if not pred else 'Pred']))
            else:
                plots.append(ax.plot(x, y, z, lw=2,linestyle='--', c=lcolor if LR[i] else rcolor))

        elif update:
            plots[i][0].set_xdata(x)
            plots[i][0].set_ydata(y)
            plots[i][0].set_3d_properties(z)
            plots[i][0].set_color(lcolor if LR[i] else rcolor)
    
    return plots
   # ax.legend(loc='lower left')


# In[11]:


def update(num,data_gt,data_pred,plots_gt,plots_pred,fig,ax):
    
    gt_vals=data_gt[num]
    pred_vals=data_pred[num]
    plots_gt=create_pose(ax,plots_gt,gt_vals,pred=False,update=True)
    plots_pred=create_pose(ax,plots_pred,pred_vals,pred=True,update=True)
    
    

    
    
    r = 0.75
    xroot, zroot, yroot = gt_vals[0,0], gt_vals[0,1], gt_vals[0,2]
    ax.set_xlim3d([-r+xroot, r+xroot])
    ax.set_zlim3d([-r+zroot, r+zroot])
    ax.set_ylim3d([-r+yroot, r+yroot])
    #ax.set_title('pose at time frame: '+str(num))
    #ax.set_aspect('equal')
 
    return plots_gt,plots_pred
    


# In[12]:


def visualize(input_n,output_n,visualize_from,path,modello,device,n_viz,skip_rate):
    
    if visualize_from=='train':
        loader=Datasets(path,input_n,output_n,skip_rate,split=0)
    elif visualize_from=='validation':
        loader=Datasets(path,input_n,output_n,skip_rate,split=1)
    elif visualize_from=='test':
        loader=Datasets(path,input_n,output_n,skip_rate,split=2)
        
    joint_used=np.arange(4,22)
    
    full_joint_used=np.arange(0,22)
        
        
    loader = DataLoader(
    loader,
    batch_size=1,
    shuffle = True,
    num_workers=0)       
    
        

    for cnt,batch in enumerate(loader): 
        batch = batch.float().to(device) # multiply by 1000 for milimeters
        sequences_train=batch[:,0:input_n,joint_used,:].permute(0,3,1,2)
        sequences_predict_gt=batch[:,input_n:input_n+output_n,full_joint_used,:]
        
        sequences_predict=modello(sequences_train).permute(0,1,3,2)
        
        all_joints_seq=sequences_predict_gt.clone()
        
        all_joints_seq[:,:,joint_used,:]=sequences_predict
        
        loss=mpjpe_error(all_joints_seq,sequences_predict_gt)*1000# # both must have format (batch,T,V,C)

        data_pred=torch.squeeze(all_joints_seq,0).cpu().data.numpy()
        data_gt=torch.squeeze(sequences_predict_gt,0).cpu().data.numpy()


        fig = plt.figure()
        ax = Axes3D(fig)
        vals = np.zeros((22, 3))
        gt_plots=[]
        pred_plots=[]

        gt_plots=create_pose(ax,gt_plots,vals,pred=False,update=False)
        pred_plots=create_pose(ax,pred_plots,vals,pred=True,update=False)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(loc='lower left')



        ax.set_xlim3d([-1, 1.5])
        ax.set_xlabel('X')

        ax.set_ylim3d([-1, 1.5])
        ax.set_ylabel('Y')

        ax.set_zlim3d([0.0, 1.5])
        ax.set_zlabel('Z')
        ax.set_title('loss in mm is: '+str(round(loss.item(),4))+' for frames= '+str(output_n))

        line_anim = animation.FuncAnimation(fig, update, output_n, fargs=(data_gt,data_pred,gt_plots,pred_plots
                                                                   ,fig,ax),interval=70, blit=False)
        plt.show()
        
     #   line_anim.save('amass_3d.gif')

        
        if cnt==n_viz-1:
            break

