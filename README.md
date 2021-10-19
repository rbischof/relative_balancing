# Multi-Objective Loss Balancing for Physics-Informed Deep Learning
Code for [ReLoBRaLo](https://www.researchgate.net/publication/355395042_Multi-Objective_Loss_Balancing_for_Physics-Informed_Deep_Learning).

## Abstract
Physics Informed Neural Networks (PINN) are algorithms from deeplearning leveraging physical laws by including partial differential equations (PDE)together with a respective set of boundary and initial conditions (BC / IC) aspenalty terms into their loss function. As the PDE, BC and IC loss function parts cansignificantly differ in magnitudes, due to their underlying physical units or stochasticityof initialisation, training of PINNs may suffer from severe convergence and efficiencyproblems, causing PINNs to stay beyond desirable approximation quality. In thiswork, we observe the significant role of correctly weighting the combination of multiplecompetitive loss functions for training PINNs effectively. To that end, we implementand evaluate different methods aiming at balancing the contributions of multipleterms of the PINNs loss function and their gradients. After review of three existingloss scaling approaches (Learning Rate Annealing, GradNorm as well as SoftAdapt),we propose a novel self-adaptive loss balancing of PINNs calledReLoBRaLo(RelativeLoss Balancing with Random Lookback). Finally, the performance of ReLoBRaLo iscompared and verified against these approaches by solving both forward as well asinverse problems on three benchmark PDEs for PINNs: Burgers’ equation, Kirchhoff’splate bending equation and Helmholtz’s equation. Our simulation studies show thatReLoBRaLo training is much faster and achieves higher accuracy than training PINNswith other balancing methods and hence is very effective and increases sustainabilityof PINNs algorithms. The adaptability of ReLoBRaLo illustrates robustness acrossdifferent PDE problem settings. The proposed method can also be employed tothe wider class of penalised optimisation problems, including PDE-constrained andSobolev training apart from the studied PINNs examples. 

## Launch Training
Example:
```
python train.py --verbose --layers 2 --nodes 32 --task helmholtz --update_rule relobralo --resample
```
The available options are the following:
* --path, default: experiments, type: str, path where to store the results
* --layers, default: 1, type: int, number of layers
* --nodes, default: 32, type: int, number of nodes
* --network, default: fc, type: str, type of network

* --optimizer, default: adam, type: str, type of optimizer
* --lr, default: 0.001, type: float, learning rate
* --patience, default: 3, type: int, how many evaluations without improvement to wait before reducing learning rate
* --factor, default: .1, type: float, multiplicative factor by which to reduce the learning rate

* --task, default: helmholtz, type: str, type of task to fit
* --inverse, action: store_true, solve inverse problem
* --inverse_var, default: None, type: float, target inverse variable
* --update_rule, default: manual, type: str, type of balancing
* --T, default: 1., type: float, temperature parameter for softmax
* --alpha, default: .999, type: float, rate for exponential decay
* --rho, default: 1., type: float, rate for exponential decay
* --aggregate_boundaries, action: store_true, aggregate all boundary terms into one before balancing

* --epochs, default: 100000, type: int, number of epochs
* --resample, action: store_true, resample datapoints or keep them fixed
* --batch_size, default: 1024, type: int, number of sampled points in a batch
* --verbose, action: store_true, print progress to terminal
