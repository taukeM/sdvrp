import torch
import torch.optim as optim
import os
import time
import math
import matplotlib.pyplot as plt
from torch.nn import DataParallel


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class State(object):

    def __init__(self, batch_size, n_nodes, mask, demand, cur_load):
        self.batch_size = batch_size
        self.n_nodes = n_nodes
        self.demand = demand
        self.mask = mask
        self.cur_load = cur_load.to(device)
        self.cur_loc = torch.full((self.batch_size, 1), self.n_nodes - 1).to(device)

    def __getitem__(self, item):
        return {
            'cur_loc': self.cur_loc[item],
            'mask': self.mask[item],
            'cur_load': self.cur_load[item],
            'demand': self.demand[item]
        }

    def update(self, cur_loc, mask, demand, cur_load):
        # self.current_node = current_node[:, None]
        self.cur_loc = cur_loc.to(device)
        self.cur_load = cur_load.to(device)
        self.demand = demand
        self.mask = mask


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


class A2CAgent(object):

    def __init__(self, model, args, env, dataGen):
        self.model = model
        self.args = args
        self.env = env
        self.dataGen = dataGen
        self.test_data = dataGen.get_test_all()
        # Initialize optimizer
        self.optimizer = optim.Adam([{'params': model.parameters(), 'lr': args['actor_net_lr']}])
        # Initialize learning rate scheduler, decay by lr_decay once per epoch!
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: args['lr_decay'] ** epoch)
        out_file = open(os.path.join(args['log_dir'], 'results.txt'), 'w+')
        print("agent is initialized")

    def train_epochs(self, baseline):
        args = self.args
        model = self.model
        test_rewards = []
        best_model = 100000
        losses = []

        start_time = time.time()
        reward_epoch = torch.zeros(args['n_epochs'])
        total_demand = torch.zeros(args['n_epochs'])
        for epoch in range(args['n_epochs']):
            # [batch_size, n_nodes, 3]: entire epoch train data
            train_data = self.dataGen.get_train_next()
            # compute baseline value for the entire epoch
            baseline_data = baseline.wrap_dataset(train_data)

            # compute for each batch the rollout
            # train each batch
            print("epoch: ", epoch)
            # evaluate b_l with  new train data and old model
            data, bl_val = baseline.unwrap_batch(baseline_data[0])
            bl_val = move_to(bl_val, device) if bl_val is not None else None
            R, logs, actions = self.rollout_train(data)
            # Calculate loss
            adv = (R - bl_val).to(device)
            loss = (adv * logs).mean()
            losses.append(loss)
            # Perform backward pass and optimization step
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norms and get (clipped) gradient norms for logging
            grad_norms = clip_grad_norms(self.optimizer.param_groups, args['max_grad_norm'])
            self.optimizer.step()
            epoch_duration = time.time() - start_time
            print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))
            avg_reward = self.rollout_test(self.test_data, self.model).mean()
            print("average test reward: ", avg_reward)
            reward_epoch[epoch] = avg_reward
            total_demand[epoch] = -self.env.total_demand.mean()
            if (epoch % args['save_interval'] == 0) or epoch == args['n_epochs'] - 1:
                print('Saving model and state...')
                torch.save(
                    {
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'rng_state': torch.get_rng_state(),
                        'cuda_rng_state': torch.cuda.get_rng_state_all(),
                        'baseline': baseline.state_dict()
                    },
                    os.path.join(args['save_path'], 'epoch-{}.pt'.format(epoch))
                )
            if avg_reward > best_model:
                best_model = avg_reward
                torch.save(
                    {
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'rng_state': torch.get_rng_state(),
                        'cuda_rng_state': torch.cuda.get_rng_state_all(),
                        'baseline': baseline.state_dict()
                    },
                    os.path.join(args['save_path'], 'best_model.pt')
                )

            baseline.epoch_callback(self.model, epoch)
            test_rewards.append(avg_reward)
            # np.savetxt("trained_models/test_rewards.txt", test_rewards)
            # np.savetxt("trained_models/losses.txt", losses)
            # lr_scheduler should be called at end of epoch
            self.lr_scheduler.step()
        plt.plot(torch.arange(args['n_epochs']).numpy(), reward_epoch.cpu().numpy())
        plt.plot(torch.arange(args['n_epochs']).numpy(), total_demand.cpu().numpy())
        plt.show()
        print(reward_epoch, "reward epoch")

    def rollout_train(self, data):
        env = self.env
        model = self.model
        model.train()
        set_decode_type(self.model, "sampling")

        data, mask, demand, cur_load = env.reset(data)
        data = move_to(data, device)
        embeddings, fixed = model.embed(data)
        state = State(env.batch_size, env.n_nodes, mask, demand, cur_load)

        # print("{}: {}".format("initial state", state[0]))
        # print("{}: {}".format("mask", mask[0]))
        logs = []
        actions = []
        time_step = 0

        while time_step < self.args['decode_len']:
            log_p, idx = model(embeddings, fixed, state)
            logs.append(log_p[:, 0, :])
            actions.append(idx)
            time_step += 1
            print(time_step, " time step")
            data, cur_loc, mask, demand, cur_load, finished = env.step(idx)
            if finished:
                break
            state.update(cur_loc, mask, demand, cur_load)
            data = move_to(data, device)
            embeddings, fixed = model.embed(data)

            # print("{}: {}".format("state update", state[0]))
            # print("{}: {}".format("mask", mask[0]))
            # print("{}: {}".format("state", env.state[0]))

        R = env.reward.to(device)
        logs = torch.stack(logs, 1)
        actions = torch.stack(actions, 1)

        logs = model._calc_log_likelihood(logs, actions)

        return R, logs, actions

    def rollout_test(self, data_o, model):
        env = self.env
        model.eval()
        set_decode_type(self.model, "greedy")
        data, mask, demand, cur_load = env.reset(data_o)
        data = move_to(data, device)
        embeddings, fixed = model.embed(data)
        state = State(env.batch_size, env.n_nodes, mask, demand, cur_load)

        # print("{}: {}".format("initial state", state[0]))
        # print("{}: {}".format("mask", mask[0]))
        time_step = 0

        while time_step < self.args['decode_len']:
            log_p, idx = model(embeddings, fixed, state)
            time_step += 1
            print(time_step, " time step")
            data, cur_loc, mask, demand, cur_load, finished = env.step(idx)
            if finished:
                break
            data = move_to(data, device)
            state.update(cur_loc, mask, demand, cur_load)
            embeddings, fixed = model.embed(data)

            # print("{}: {}".format("state update", state[0]))
            # print("{}: {}".format("mask", mask[0]))
            # print("{}: {}".format("state", env.state[0]))

        R = env.reward.to(device)

        return R
