import gym
import argparse
import yaml
import os

from collections import OrderedDict
from stable_baselines3 import PPO_REWARD
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_dmcontrol_env, make_vec_metaworld_env
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from stable_baselines3.common.vec_env import VecNormalize
from reward_model import RewardModel

def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="walker_walk", help="environment ID")
    parser.add_argument("-tb", "--tensorboard-log", help="Tensorboard log dir", default="logs/PrefPPO/", type=str)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=123)
    parser.add_argument("--n-envs", help="# of parallel environments", type=int, default=16)
    parser.add_argument("--n-steps", help="# of steps to run for each environment per update", type=int, default=500)
    parser.add_argument("--lr", help="learning rate", type=float, default=3e-4)
    parser.add_argument("--total-timesteps", help="total timesteps", type=int, default=2000000)
    parser.add_argument("-b", "--batch-size", help="batch size", type=int, default=64)
    parser.add_argument("--ent-coef", help="coeff for entropy", type=float, default=0.0)
    parser.add_argument("--hidden-dim", help="dim of hidden features", type=int, default=1024)
    parser.add_argument("--num-layer", help="# of layers", type=int, default=2)
    parser.add_argument("--use-sde", help="Whether to use generalized State Dependent Exploration", type=int, default=1)
    parser.add_argument("--sde-freq", help="Sample a new noise matrix every n steps", type=int, default=4)
    parser.add_argument("--gae-lambda", help="Factor for trade-off of bias vs variance", type=float, default=0.92)
    parser.add_argument("--clip-init", help="Initial value of clipping", type=float, default=0.4)
    parser.add_argument("--n-epochs", help="Number of epoch when optimizing the surrogate loss", type=int, default=20)
    parser.add_argument("--normalize", help="Normalization", type=int, default=1)
    parser.add_argument("--unsuper-step", help="# of steps for unsupervised learning", type=int, default=32000)
    parser.add_argument("--unsuper-n-epochs", help="# of steps for unsupervised learning", type=int, default=50)
    parser.add_argument("--num_ratings", help="Number of Ratings (2-6)", type=int, default=2)
    # reward learning
    parser.add_argument("--re-lr", help="Learning rate of reward fn", type=float, default=0.0003)
    parser.add_argument("--re-segment", help="Size of segment", type=int, default=50)
    parser.add_argument("--re-act", help="Last activation for reward fn", type=str, default='tanh')
    parser.add_argument("--re-num-interaction", help="# of interactions", type=int, default=16000)
    parser.add_argument("--re-num-feed", help="# of feedbacks", type=int, default=1)
    parser.add_argument("--re-batch", help="Batch size", type=int, default=128)
    parser.add_argument("--re-update", help="Gradient update of reward fn", type=int, default=100)
    parser.add_argument("--re-feed-type", help="type of feedback", type=int, default=0)
    parser.add_argument("--re-large-batch", help="size of buffer for ensemble uncertainty", type=int, default=10)
    parser.add_argument("--re-max-feed", help="# of total feedback", type=int, default=1400)
    parser.add_argument("--teacher-beta", type=float, default=-1)
    parser.add_argument("--teacher-gamma", type=float, default=1.0)
    parser.add_argument("--teacher-eps-mistake", type=float, default=0.0)
    parser.add_argument("--teacher-eps-skip", type=float, default=0.0)
    parser.add_argument("--teacher-eps-equal", type=float, default=0.0)
    args = parser.parse_args()
    
    metaworld_flag = False
    max_ep_len = 1000
    if 'metaworld' in args.env:
        metaworld_flag = True
        max_ep_len = 500
    
    env_name = args.env
        
    if args.normalize == 1:
        args.tensorboard_log += 'normalized_' + env_name
    else:
        args.tensorboard_log += env_name
    
    args.tensorboard_log += '/teacher_' + str(args.teacher_beta)
    args.tensorboard_log += '_' + str(args.teacher_gamma)
    args.tensorboard_log += '_' + str(args.teacher_eps_mistake)
    args.tensorboard_log += '_' + str(args.teacher_eps_skip)
    args.tensorboard_log += '_' + str(args.teacher_eps_equal)
    args.tensorboard_log += '/disagreement_' + str(args.re_num_interaction)
    args.tensorboard_log += '_type_' + str(args.re_feed_type)
    args.tensorboard_log += '_ratings_' + str(args.num_ratings)
    args.tensorboard_log += '/seed_' + str(args.seed) 
    
    # extra params
    if args.use_sde == 0:
        use_sde = False
    else:
        use_sde = True
    
    clip_range = linear_schedule(args.clip_init)
    
    # Parallel environments
    if metaworld_flag:
        env = make_vec_metaworld_env(
            args.env, 
            n_envs=args.n_envs, 
            monitor_dir=args.tensorboard_log,
            seed=args.seed)
    else:
        env = make_vec_dmcontrol_env(
            args.env,     
            n_envs=args.n_envs, 
            monitor_dir=args.tensorboard_log,
            seed=args.seed)
            
    # instantiating the reward model
    reward_model = RewardModel(
        env.envs[0].observation_space.shape[0],
        env.envs[0].action_space.shape[0],
        size_segment=args.re_segment,
        activation=args.re_act, 
        lr=args.re_lr,
        mb_size=args.re_batch, 
        teacher_beta=args.teacher_beta, 
        teacher_gamma=args.teacher_gamma, 
        teacher_eps_mistake=args.teacher_eps_mistake,
        teacher_eps_skip=args.teacher_eps_skip, 
        teacher_eps_equal=args.teacher_eps_equal,
        large_batch=args.re_large_batch,
        teacher_num_ratings=args.num_ratings)
    
    if args.normalize == 1:
        env = VecNormalize(env, norm_reward=False)
    
    # network arch
    net_arch = [dict(pi=[args.hidden_dim]*args.num_layer, 
                     vf=[args.hidden_dim]*args.num_layer)]
    policy_kwargs = dict(net_arch=net_arch)
    
    # train model
    model = PPO_REWARD(
        reward_model,
        MlpPolicy, env,
        tensorboard_log=args.tensorboard_log, 
        seed=args.seed, 
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        use_sde=use_sde,
        sde_sample_freq=args.sde_freq,
        gae_lambda=args.gae_lambda,
        clip_range=clip_range,
        n_epochs=args.n_epochs,
        num_interaction=args.re_num_interaction,
        num_feed=args.re_num_feed,
        feed_type=args.re_feed_type,
        re_update=args.re_update,
        metaworld_flag=metaworld_flag,
        max_feed=args.re_max_feed, 
        unsuper_step=args.unsuper_step,
        unsuper_n_epochs=args.unsuper_n_epochs,
        size_segment=args.re_segment,
        max_ep_len=max_ep_len,
        verbose=1)

    # save args
    with open(os.path.join(args.tensorboard_log, "args.yml"), "w") as f:
        ordered_args = OrderedDict([(key, vars(args)[key]) for key in sorted(vars(args).keys())])
        yaml.dump(ordered_args, f)
        
    model.learn(total_timesteps=args.total_timesteps, unsuper_flag=1)
    model.reward_model.save(args.tensorboard_log, args.total_timesteps)
