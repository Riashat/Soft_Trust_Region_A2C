import copy
import glob
import os
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from envs import make_env
from kfac import KFACOptimizer
from model import CNNPolicy, MLPPolicy
from storage import RolloutStorage
from visualize import visdom_plot
from utils import Logger

args = get_args()

assert args.algo in ['a2c']

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def main():
    print("#######")
    print("WARNING: All rewards are clipped or normalized so you need to use a monit`or (see envs.py) or visdom plot to get true rewards")
    print("#######")

    os.environ['OMP_NUM_THREADS'] = '1'

    # logger = Logger(algorithm_name = args.algo, environment_name = args.env_name, folder = args.folder)
    # logger.save_args(args)

    # print ("---------------------------------------")
    # print ('Saving to', logger.save_folder)
    # print ("---------------------------------------")    


    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None



    envs = [make_env(args.env_name, args.seed, i, args.log_dir)
                for i in range(args.num_processes)]

    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    if len(envs.observation_space.shape) == 3:
        actor_critic = CNNPolicy(obs_shape[0], envs.action_space, args.recurrent_policy)
        target_actor_critic = CNNPolicy(obs_shape[0], envs.action_space, args.recurrent_policy)

    else:
        actor_critic = MLPPolicy(obs_shape[0], envs.action_space)
        target_actor_critic = MLPPolicy(obs_shape[0], envs.action_space)

    for param, target_param in zip(actor_critic.parameters(), target_actor_critic.parameters()):
            target_param.data.copy_(param.data)

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if args.cuda:
        actor_critic.cuda()

    actor_regularizer_criterion = nn.KLDivLoss()
    optimizer = optim.RMSprop(actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size)
    current_obs = torch.zeros(args.num_processes, *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)

    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            value, action, action_log_prob, states = actor_critic.act(Variable(rollouts.observations[step], volatile=True),
                                                                      Variable(rollouts.states[step], volatile=True),
                                                                      Variable(rollouts.masks[step], volatile=True))
            cpu_actions = action.data.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs)
            rollouts.insert(step, current_obs, states.data, action.data, action_log_prob.data, value.data, reward, masks)

        next_value = actor_critic(Variable(rollouts.observations[-1], volatile=True), Variable(rollouts.states[-1], volatile=True), Variable(rollouts.masks[-1], volatile=True))[0].data

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        values, action_log_probs, dist_entropy, states, action_mean, action_std = actor_critic.evaluate_actions_mean_and_std(Variable(rollouts.observations[:-1].view(-1, *obs_shape)),
                                                                                       Variable(rollouts.states[0].view(-1, actor_critic.state_size)),
                                                                                       Variable(rollouts.masks[:-1].view(-1, 1)),
                                                                                       Variable(rollouts.actions.view(-1, action_shape)))

        target_values, target_action_log_probs, target_dist_entropy, target_states, target_action_mean, target_action_std = target_actor_critic.evaluate_actions_mean_and_std(Variable(rollouts.observations[:-1].view(-1, *obs_shape)),
                                                                                       Variable(rollouts.states[0].view(-1, actor_critic.state_size)),
                                                                                       Variable(rollouts.masks[:-1].view(-1, 1)),
                                                                                       Variable(rollouts.actions.view(-1, action_shape)))


        actor_regularizer_loss = (torch.log(action_std/target_action_std) + (action_std.pow(2) + (action_mean - target_action_mean).pow(2))/(2*target_action_std.pow(2)) - 0.5)
        #t_log_probs = Variable(target_action_log_probs.data, requires_grad=False)


        values = values.view(args.num_steps, args.num_processes, 1)
        action_log_probs = action_log_probs.view(args.num_steps, args.num_processes, 1)
        #t_log_probs = t_log_probs.view(args.num_steps, args.num_processes, 1)

        advantages = Variable(rollouts.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()

        #actor_regularizer_loss = actor_regularizer_criterion(action_log_probs, t_log_probs)

        action_loss = -(Variable(advantages.data) * action_log_probs).mean() + args.actor_lambda * actor_regularizer_loss.mean(0).sum()

        # if j > 1000 and j < 3000:
        #     args.actor_lambda *= 0.9
        # elif j > 3000:
        #     args.actor_lambda *= 1.0
            #args.actor_lambda = max(args.actor_lambda, 0.9)


        optimizer.zero_grad()
        total_loss = value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef
        total_loss.backward()

        nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)
        optimizer.step()

        ## Exponential average for target updates
        #if (j%args.target_update_interval == 0):
        for param, target_param in zip(actor_critic.parameters(), target_actor_critic.parameters()):
            target_param.data.copy_(args.target_tau * param.data + (1 - args.target_tau) * target_param.data)

        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                            hasattr(envs, 'ob_rms') and envs.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), dist_entropy.data[0],
                       value_loss.data[0], action_loss.data[0]))

            final_rewards_mean = [final_rewards.mean()]
            final_rewards_median = [final_rewards.median()]
            final_rewards_min = [final_rewards.min()]
            final_rewards_max = [final_rewards.max()]


            # logger.record_data(final_rewards_mean, final_rewards_median, final_rewards_min, final_rewards_max)
            # logger.save()  


        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name, args.algo)
            except IOError:
                pass

if __name__ == "__main__":
    main()
