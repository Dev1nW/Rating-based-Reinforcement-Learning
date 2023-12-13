# Rating-based Reinforcement Learning

Official codebase for [Rating-based Reinforcement Learning](https://arxiv.org/pdf/2307.16348.pdf) which is based off of the [B-Pref: Benchmarking Preference-BasedReinforcement Learning](https://openreview.net/forum?id=ps95-mkHF_) codebase.


## Install

```
conda env create -f conda_env.yml
pip install -e .[docs,tests,extra]
cd custom_dmcontrol
pip install -e .
cd custom_dmc2gym
pip install -e .
pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld
pip install pybullet
```

## Run Rating-based Reinforcement Learning experiments

Experiments for Walker can be reproduced by running the following command:

```
./scripts/walker_walk/1000/equal/run_PrefPPO.sh [n = 2, 3, 4, 5, 6]

```

Experiments for Quadruped can be reproduced by adjusting the reward threshold for specific rating classes and then running the following command:

```
./scripts/quadruped_walk/2000/equal/run_PrefPPO.sh [n = 2, 3, 4, 5, 6]
```

### PPO 

Experiments can be reproduced with the following:

```
./scripts/walker_walk/run_ppo.sh 
./scripts/quadruped_walk/run_ppo.sh 
```
