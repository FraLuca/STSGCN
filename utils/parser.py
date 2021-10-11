import argparse


parser = argparse.ArgumentParser(description='Arguments for running the scripts')

#ARGS FOR LOADING THE DATASET


parser.add_argument('--data_dir',type=str,default='../datasets/',help='path to the unziped dataset directories(H36m/AMASS/3DPW)')
parser.add_argument('--input_n',type=int,default=10,help="number of model's input frames")
parser.add_argument('--output_n',type=int,default=10,help="number of model's output frames")
parser.add_argument('--skip_rate',type=int,default=1,choices=[1,5],help='rate of frames to skip,defaults=1 for H36M or 5 for AMASS/3DPW')
parser.add_argument('--joints_to_consider',type=int,default=22,choices=[16,18,22],help='number of joints to use, defaults=16 for H36M angles, 22 for H36M 3D or 18 for AMASS/3DPW')


#ARGS FOR THE MODEL

parser.add_argument('--n_stgcnn_layers',type=int,default=9,help= 'number of stgcnn layers')
parser.add_argument('--n_ccnn_layers',type=int,default=2,help= 'number of layers for the Coordinate-Channel Convolution')
parser.add_argument('--n_tcnn_layers',type=int,default=4,help= 'number of layers for the Time-Extrapolator Convolution')
parser.add_argument('--ccnn_kernel_size',type=list,default=[1,1],help= ' kernel for the C-CNN layers')
parser.add_argument('--tcnn_kernel_size',type=list,default=[3,3],help= ' kernel for the Time-Extrapolator CNN layers')
parser.add_argument('--embedding_dim',type=int,default=40,help= 'dimensions for the coordinates of the embedding')
parser.add_argument('--input_dim',type=int,default=3,help= 'dimensions of the input coordinates')
parser.add_argument('--st_gcnn_dropout',type=float,default=.1,help= 'st-gcnn dropout')
parser.add_argument('--ccnn_dropout',type=float,default=0.0,help= 'ccnn dropout')
parser.add_argument('--tcnn_dropout',type=float,default=0.0,help= 'tcnn dropout')






#ARGS FOR THE TRAINING




parser.add_argument('--mode',type=str,default='train',choices=['train','test','viz'],help= 'Choose to train,test or visualize from the model.Either train,test or viz')
parser.add_argument('--n_epochs',type=int,default=50,help= 'number of epochs to train')
parser.add_argument('--batch_size',type=int,default=256,help= 'batch size')
parser.add_argument('--batch_size_test',type=int,default=256,help= 'batch size for the test set')
parser.add_argument('--lr',type=int,default=1e-02,help= 'Learning rate of the optimizer')
parser.add_argument('--use_scheduler',type=bool,default=True,help= 'use MultiStepLR scheduler')
parser.add_argument('--milestones',type=list,default=[15,25,35,40],help= 'the epochs after which the learning rate is adjusted by gamma')
parser.add_argument('--gamma',type=float,default=0.1,help= 'gamma correction to the learning rate, after reaching the milestone epochs')
parser.add_argument('--clip_grad',type=float,default=None,help= 'select max norm to clip gradients')
parser.add_argument('--model_path',type=str,default='./checkpoints/CKPT_3D_H36M',help= 'directory with the models checkpoints ')


#FLAGS FOR THE VISUALIZATION

parser.add_argument('--visualize_from',type=str,default='test',choices =['train','val','test'],help= 'choose data split to visualize from(train-val-test)')
parser.add_argument('--actions_to_consider',default='all',help= 'Actions to visualize.Choose either all or a list of actions')
parser.add_argument('--n_viz',type=int,default='2',help= 'Numbers of sequences to visaluze for each action')




args = parser.parse_args()




