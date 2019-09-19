# Lagrangian Control through Deep-RL: Applications to Bottleneck Decongestion

## Abstract
Using deep reinforcement learning, we derive
novel control policies for autonomous vehicles to improve the
throughput of a bottleneck modeled after the San Francisco-
Oakland Bay Bridge. Using Flow, a new library for applying
deep reinforcement learning to traffic micro-simulators, we
consider the problem of improving the throughput of a traffic
benchmark: a two-stage bottleneck where four lanes reduce
to two and then reduce to one. We first characterize the
inflow-outflow curve of this bottleneck without any control. We
introduce an inflow of autonomous vehicles with the intent of
improving the congestion through Lagrangian control. To handle
the varying number of autonomous vehicles in the system
we derive a per-lane variable speed limits parametrization of
the controller. We demonstrate that a 10% penetration rate of
controlled autonomous vehicles can improve the throughput of
the bottleneck by 200 vehicles per hour: a 25% improvement at
high inflows. Finally, we compare the performance of our control
policies to feedback ramp metering and show that the AV
controller provides comparable performance to ramp metering
without the need to build new ramp metering infrastructure.
Illustrative videos of the results can be found at https:
//sites.google.com/view/itsc-lagrangian-avs/home and
code and tutorials can be found at https://github.com/
flow-project/flow.

## Authors
* Eugene Vinitsky  
* Kanaad Parvate
* [Aboudy Kreidieh](https://github.com/AboudyKreidieh)
* [Cathy Wu](https://github.com/cathywu)
* Alexandre M Bayen

## Resources

* Paper: [https://flow-project.github.io/papers/LagrangianBottlenecks.pdf](https://flow-project.github.io/papers/LagrangianBottlenecks.pdf)
* Website: [https://sites.google.com/view/itsc-lagrangian-avs/home](https://sites.google.com/view/itsc-lagrangian-avs/home)

## Usage

This sub-repository allows you to recreate the case study results from the 
article. This includes the results from the *########* and 
*########* controllers, as well as the derived controllers from 
reinforcement learning. The training operation can also be replicated from this
package.

### Installation [NEEDS REVIEW]

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
## Running Experiments/Results

### Hand Designed Controllers (Baselines)
Note: No training is done in the baselines.

The *####* controller example can be run using the command:

```bash
python ###.py
```

The *####* controller example can be run using the command:

```bash
python #####.py
```

### Reinforcement Learning Controllers [NEEDS REVIEW]

Training is performed using Trust Region Policy Optimization (TRPO) using an 
MLP policy with shape (3,3). 

In order to run training on the MLP policy, run:

```bash
python ###.py
```

```bash
python #####.py
```

## Citing [NEEDS REVIEW]

If you would like to cite this work, please use the following citation:

```
@article{wu2017flow,
  title={Lagrangian Control through Deep-RL: Applications to Bottleneck Decongestion},
  author={Vinitsky, Eugene and Parvate, Kanaad and Kreidieh, Aboudy and Wu, Cathy and Bayen, Alexandre M},
  journal={ITSC},
  year={2018}
}
```
