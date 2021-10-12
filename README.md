
<div align="center">
<h1>Space-Time-Separable Graph Convolutional Network for Pose Forecasting (<b>Accepted to ICCV '21</b>)</h1>
<h3> <i>Theodoros Sofianos†, Alessio Sampieri†, Luca Franco and Fabio Galasso</i></h3>
 <h4> <i>Sapienza University of Rome, Italy</i></h4>
 
 [[Paper](https://arxiv.org/abs/2110.04573)] [[Website](https://fraluca.github.io/)] [[Talk](https://youtu.be/tQIygtJrrtk)]
 

<image src="https://github.com/FraLuca/STSGCN/blob/main/pipeline-cameraready-1.png" width="600">
</div>

<div align="center"> <h3> Abstract </h3>  </div>
<div align="justify">
Human pose forecasting is a complex structured-data sequence-modelling task, which has received increasing attention, also due to numerous potential applications. Research has mainly addressed the temporal dimension as time series and the interaction of human body joints with a kinematic tree or by a graph. This has decoupled the two aspects and leveraged progress from the relevant fields, but it has also limited the understanding of the complex structural joint spatio-temporal dynamics of the human pose.

Here we propose a novel Space-Time-Separable Graph Convolutional Network (STS-GCN) for pose forecasting. For the first time, STS-GCN models the human pose dynamics only with a graph convolutional network (GCN), including the temporal evolution and the spatial joint interaction within a single-graph framework, which allows the cross-talk of motion and spatial correlations. Concurrently, STS-GCN is the first space-time-separable GCN: the space-time graph connectivity is factored into space and time affinity matrices, which bottlenecks the space-time cross-talk, while enabling full joint-joint and time-time correlations. Both affinity matrices are learnt end-to-end, which results in connections substantially deviating from the standard kinematic tree and the linear-time time series.

In experimental evaluation on three complex, recent and large-scale benchmarks, Human3.6M [Ionescu et al. TPAMI'14], AMASS [Mahmood et al. ICCV'19] and 3DPW [Von Marcard et al. ECCV'18], STS-GCN outperforms the state-of-the-art, surpassing the current best technique [Mao et al. ECCV'20] by over 32% in average in the most difficult long-term predictions, while only requiring 1.7% of its parameters. We explain the results qualitatively and illustrate the graph attention by the factored joint-joint and time-time learnt graph connections.
</div>
--------


 ### Install dependencies:
```
 $ pip install -r requirements.txt
```
 
 ### Get the data

[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).
 
Directory structure: 
```shell script
H3.6m
|-- S1
|-- S5
|-- S6
|-- ...
`-- S11
```

[AMASS](https://amass.is.tue.mpg.de/en) from their official website.
 

Directory structure:
```shell script
amass
|-- ACCAD
|-- BioMotionLab_NTroje
|-- CMU
|-- ...
`-- Transitions_mocap
```
[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) from their official website.

Directory structure: 
```shell script
3dpw
|-- imageFiles
|   |-- courtyard_arguing_00
|   |-- courtyard_backpack_00
|   |-- ...
`-- sequenceFiles
    |-- test
    |-- train
    `-- validation
```
Put the all downloaded datasets in ../datasets directory.

### Train
The arguments for running the code are defined in [parser.py](utils/parser.py). We have used the following commands for training the network,on different datasets and body pose representations(3D and euler angles):
 
```bash
 python main_h36_3d.py --input_n 10 --output_n 25 --skip_rate 1 --joints_to_consider 22 
 ```
```bash
 python main_h36_ang.py --input_n 10 --output_n 25 --skip_rate 1 --joints_to_consider 16 
  ```
```bash
  python main_amass_3d.py --input_n 10 --output_n 25 --skip_rate 5 --joints_to_consider 18 
  ```
 
 ### Test
 To test on the pretrained model, we have used the following commands:
 ```bash
 python main_h36_3d.py --input_n 10 --output_n 25 --skip_rate 1 --joints_to_consider 22 --mode test --model_path ./checkpoints/CKPT_3D_H36M
  ```
  ```bash
  python main_h36_ang.py --input_n 10 --output_n 25 --skip_rate 1 --joints_to_consider 16 --mode test --model_path ./checkpoints/CKPT_ANG_H36M
  ```
  ```bash
   python main_amass_3d.py --input_n 10 --output_n 25 --skip_rate 5 --joints_to_consider 18 --mode test --model_path ./checkpoints/CKPT_3D_AMASS
  ```
### Visualization
 For visualizing from a pretrained model, we have used the following commands:
 ```bash
  python main_h36_3d.py --input_n 10 --output_n 25 --skip_rate 1 --joints_to_consider 22 --mode viz --model_path ./checkpoints/CKPT_3D_H36M --n_viz 5
 ```
 ```bash
  python main_h36_ang.py --input_n 10 --output_n 25 --skip_rate 1 --joints_to_consider 16 --mode viz --model_path ./checkpoints/CKPT_ANG_H36M --n_viz 5
 ```
 ```bash
  python main_amass_3d.py --input_n 10 --output_n 25 --skip_rate 5 --joints_to_consider 18 --mode viz --model_path ./checkpoints/CKPT_3D_AMASS --n_viz 5
 ```

### Citing
 If you use our code,please cite our work
 
 ```
@misc{sofianos2021spacetimeseparable,
      title={Space-Time-Separable Graph Convolutional Network for Pose Forecasting}, 
      author={Theodoros Sofianos and Alessio Sampieri and Luca Franco and Fabio Galasso},
      year={2021},
      eprint={2110.04573},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
 ```
 
 ### Acknowledgments
 
 Some of our code was adapted from [HisRepsItself](https://github.com/wei-mao-2019/HisRepItself) by [Wei Mao](https://github.com/wei-mao-2019).
 
 The authors wish to acknowledge Panasonic for partially supporting this work and the project of the Italian Ministry of Education, Universities and Research (MIUR) “Dipartimenti di Eccellenza 2018-2022”.

 
 ### License 
 
 MIT license
