# Dissipating Stop-and-Go Waves in Closed and Open Networks via Deep Reinforcement Learning

## Abstract
We demonstrates the ability for model free reinforcement learning (RL) techniques to generate traffic
control strategies for connected and automated vehicles (CAVs)
in various network geometries (Open Networks and Closed Networks). This method is demonstrated to
achieve near complete wave dissipation in a straight open road
network with only 10% CAV penetration, while penetration
rates as low as 2.5% are revealed to contribute greatly to
reductions in the frequency and magnitude of formed waves.
Moreover, a study of controllers generated in closed network
scenarios exhibiting otherwise similar densities and perturbing
behaviors confirms that closed network policies generalize to
open network tasks, and presents the potential role of transfer
learning in fine-tuning the parameters of these policies. Videos
of the results are available at: https://sites.google.com/view/itsc-dissipating-waves.

## Authors
* [Aboudy Kreidieh](https://github.com/AboudyKreidieh)
* [Cathy Wu](https://github.com/cathywu)
* Alexandre M Bayen

## Resources

* Paper: [https://flow-project.github.io/papers/08569485.pdf](https://flow-project.github.io/papers/08569485.pdf)
* Website: [https://sites.google.com/view/ieee-tro-flow](https://sites.google.com/view/ieee-tro-flow)

## Usage

This sub-repository allows you to recreate the case study results from the 
article. This includes the results from the *Open Network Highway - Merge* and 
*Closed Network Highway - Ring-Road* controllers, as well as the derived controllers from 
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

The *Open Network Highway - Merge* controller example can be run using the command:

```bash
python baseline.py
```

The *Closed Network Highway - Ring-Road* controller example can be run using the command:

```bash
python #####.py
```

### Reinforcement Learning Controllers [NEEDS REVIEW]

Training is performed using Trust Region Policy Optimization (TRPO) using an 
MLP policy with shape (3,3). 

In order to run training on the MLP policy, run:

```bash
python train_mlp.py
```

```bash
python #####.py <-- for ring road
```

## Citing [NEEDS REVIEW]

If you would like to cite this work, please use the following citation:

```
@article{wu2017flow,
  title={Dissipating stop-and-go waves in closed and open networks via deep reinforcement learning},
  author={Kreidieh, Aboudy and Wu, Cathy and Bayen, Alexandre M},
  journal={ITSC},
  year={2018}
}
```
