
import envwrapper
import gym
import memory
from memory import Transition
from model import DQN
import torch
import torch.nn.functional  as F
import math
import numpy as np
import random
from itertools import count
import matplotlib.pyplot as plt



steps_done = 0


def select_action(state):


    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1.* steps_done / EPS_DECAY)

    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # what's the shape of the output?
            # input_samples * 1 ?
            output = policy_net(state.to(device)).max(1)[1]
            #print('shape of select action output is', output.shape)
            return policy_net(state.to(device)).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(4)]], dtype=torch.long, device=device)

    pass




# training


def train(env, n_episodes):
    rewards = []
    for i in range(n_episodes):
        obs = env.reset()
        cur_state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = select_action(cur_state)

            obs, reward, done, _ = env.step(action)
            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None
            
            #reward = torch.tensor([reward], device=device)

            replaymemory.push(cur_state, action.to('cpu'), next_state, reward)
            cur_state = next_state

            '''
            initially we don't have enough trasition tuple to 
            form a optimization batch
            so we just use initial network to sample training tuples
            from env
            '''
            if steps_done > OPTIMIZE_THRESHOLD:
                optimize()

                '''
                to keep the training samples relatively I.I.D
                we only update the network after get some samples
                with current parameters
                '''
                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            
            if done:
                rewards.append(total_reward)
                break

        if i % 20 == 0:
            print('current episode:{}, current episode reward:{}'.format(i, total_reward))
    env.close()
    return rewards
    pass


def optimize():
    if len(replaymemory) < BATCH_SIZE:
        return
    # sample tuples
    trainsitions = replaymemory.sample(BATCH_SIZE)


    batch = Transition(*zip(*trainsitions))
    actions = tuple((map(lambda a:torch.tensor([[a]], device=device), batch.action)))
    rewards = tuple((map(lambda r:torch.tensor([r], device=device), batch.reward)))

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    next_state_batch = batch.next_state

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_batch)),dtype=torch.bool, device=device)
    non_final_next_states = torch.cat([s for s in next_state_batch if s is not None]).to(device)

    #policy net output q value for cur state
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # target net output q value for cur state
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()    

    pass

def test(env, render, model):
    pass

def get_state(obs):
    state = np.array(obs)
    #print('original state shape is', state.shape)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)


def plot_rewards(rewards):
    plt.figure()
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.plot(rewards)
    plt.savefig('/content/drive/MyDrive/pong/training-reward.jpg')
    plt.show()
    pass



if __name__ == "__main__":
    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.9
    EPS_DECAY = 1000000#200
    TARGET_UPDATE = 1000#100

    EPISODE_NUM = 1000
    BATCH_SIZE = 32


    lr = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        print('----- cuda available ----')
    else:
        print('----- cuda unavailable ----')
        

    policy_net = DQN(output=4).to(device)
    target_net = DQN(output=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())



    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    env = gym.make('PongNoFrameskip-v4')
    #env = gym.make('Pong-v0')
    env = envwrapper.make_env(env)

    # prepare memory
    OPTIMIZE_THRESHOLD =  1000
    capacity = OPTIMIZE_THRESHOLD * 10

    replaymemory = memory.ReplayMemory(capacity)

    episode_rewards = train(env, EPISODE_NUM)

    plot_rewards(episode_rewards)
    
    torch.save(policy_net, 'dqn_pong_model')
    policy_net = torch.load('dqn_pong_model')
    test(env, 1, policy_net)

'''
    print(env.action_space)

    # select action to interact with env
    for i in range(10):
        selected_action = select_action(get_state(env.reset()))
        print(selected_action.shape)
    pass

'''