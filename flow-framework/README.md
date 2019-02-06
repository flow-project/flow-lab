# Flow: Architecture and Benchmarking for Reinforcement Learning in Traffic Control

## Abstract

Flow is a new computational framework, built to support a key need triggered by 
the rapid growth of autonomy in ground traffic: controllers for autonomous 
vehicles in the presence of complex nonlinear dynamics in traffic. Leveraging 
recent advances in deep Reinforcement Learning (RL), Flow enables the use of RL 
methods such as policy gradient for traffic control and enables benchmarking 
the performance of classical (including hand-designed) controllers with learned 
policies (control laws). Flow integrates traffic microsimulator SUMO with deep 
reinforcement learning library rllab and enables the easy design of traffic 
tasks, including different networks configurations and vehicle dynamics. We use 
Flow to develop reliable controllers for complex problems, such as controlling 
mixed-autonomy traffic (involving both autonomous and human-driven vehicles) 
in a ring road. For this, we first show that state-of-the-art hand-designed 
controllers excel when in-distribution, but fail to generalize; then, we show 
that even simple neural network policies can solve the stabilization task 
across density settings and generalize to out-of-distribution settings.

## Authors

* [Cathy Wu](https://github.com/cathywu)
* [Aboudy Kreidieh](https://github.com/AboudyKreidieh)
* [Kanaad Pravate](https://github.com/kanaadp)
* [Eugene Vinitsky](https://github.com/eugenevinitsky)
* Alexandre M. Bayen

## Resources

* Paper: [https://arxiv.org/pdf/1710.05465.pdf]()
* Website: [https://sites.google.com/view/ieee-tro-flow]()

## Usage

This sub-repository allows you to recreate the case study results from the 
article. This includes the results from the *FollowerStopper* and 
*PI with Saturation* controllers, as well as the derived controllers from 
reinforcement learning. The training operation can also be replicated from this
package.

### Installation

In order to install the correct versions of flow and all other packages, run 
the following command. **Note**: You will need to have 
[Anaconda](https://www.anaconda.com/distribution/) or 
[Miniconda](https://conda.io/en/latest/miniconda.html) installed for the setup 
to be successful.

```bash
sh ./install.sh
```

The script will prompt you to specify the operating system you are using. Type 
the number corresponding to your OS and press enter:

```
Type the number corresponding to your operating system, followed by [ENTER]:
1 - Ubuntu 14.04
2 - Ubuntu 16.04
3 - Ubuntu 18.04
4 - Mac OSX
```

The setup instructions will create a conda environment that can be used to run 
all sample scripts. In order to use this environment, run the following command:

```bash
source activate flow-framework
```

### Hand Designed Controllers

The controllers that the case study was compared against are the 
*FollowerStopper* and *PI with Saturation* controllers from the paper: Stern, 
Raphael E., et al. "Dissipation of stop-and-go waves via control of autonomous 
vehicles: Field experiments." Transportation Research Part C: Emerging 
Technologies 89 (2018): 205-221.

The *FollowerStopper* controller example can be run using the command:

```bash
python followerstopper.py
```

The *PI with Saturation* controller example can be run using the command:

```bash
python pisaturation.py
```

In both cases, the automated vehicles acts as a human driver for the first 300 
seconds, and then is controlled by the respective controller for the next 300 
seconds. The length of the network in both cases can be modified using the 
`LENGTH` attribute in each script.

### Reinforcement Learning Controllers

Training is performed using Trust Region Policy Optimization (TRPO) using an 
MLP policy with shape (3,3) and a GRU policy with shape (5,).

In order to run training on the MLP policy, run:

```bash
python train_mlp.py
```

In order to run training on the GRU policy, run:

```bash
python train_gru.py
```

## Citing

If you would like to cite this work, please use the following citation:

```
@article{wu2017flow,
  title={Flow: Architecture and benchmarking for reinforcement learning in traffic control},
  author={Wu, Cathy and Kreidieh, Aboudy and Parvate, Kanaad and Vinitsky, Eugene and Bayen, Alexandre M},
  journal={arXiv preprint arXiv:1710.05465},
  year={2017}
}
```
