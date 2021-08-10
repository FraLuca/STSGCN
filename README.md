# Space-Time-Separable Graph Convolutional Network for Pose Forecasting 
### Accepted to ICCV 2021

--------

<p align="center">
<image src="https://github.com/FraLuca/STSGCN/blob/main/pipeline-cameraready.pdf" type="application/pdf" width="600">
</p>

### Abstract 

Human pose forecasting is a complex structured-data sequence-modelling task, which has received increasing attention, also due to numerous potential applications. Research has mainly addressed the temporal dimension as time series and the interaction of human body joints with a kinematic tree or by a graph. This has decoupled the two aspects and leveraged progress from the relevant fields, but it has also limited the understanding of the complex structural joint spatio-temporal dynamics of the human pose.

Here we propose a novel Space-Time-Separable Graph Convolutional Network (STS-GCN) for pose forecasting. For the first time, STS-GCN models the human pose dynamics only with a graph convolutional network (GCN), including the temporal evolution and the spatial joint interaction within a single-graph framework, which allows the cross-talk of motion and spatial correlations. Concurrently, STS-GCN is the first space-time-separable GCN: the space-time graph connectivity is factored into space and time affinity matrices, which bottlenecks the space-time cross-talk, while enabling full joint-joint and time-time correlations. Both affinity matrices are learnt end-to-end, which results in connections substantially deviating from the standard kinematic tree and the linear-time time series.

In experimental evaluation on three complex, recent and large-scale benchmarks, Human3.6M~\cite{Human36M}, AMASS~\cite{AMASS} and 3DPW~\cite{3DPW}, STS-GCN outperforms the state-of-the-art, surpassing the current best technique~\cite{Mao20DCT} by over 33\% in average in the most difficult long-term predictions, while only requiring 
2.6\% of its parameters. We explain the results qualitatively and illustrate the graph interactions by the factored joint-joint and time-time learnt graph connections.

--------
## Code will be released soon 
