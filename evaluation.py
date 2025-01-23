import numpy as np
import torch

from model import ActorCriticModel
from utils import create_env


def evaluate(model: ActorCriticModel,
             num_episodes: int,
             config: dict,
             memory: torch.Tensor,
             memory_mask: torch.Tensor,
             memory_indices: torch.Tensor,
             memory_length: int,
             device: torch.device):

    # Instantiate environment
    env = create_env(config["environment"], render=True)
    model.eval()
    
    total_episode_rewards = []
    total_episode_lengths = []
    # Run and render episode
    for i in range(num_episodes):
        done = False
        episode_rewards = []
        t = 0
        obs = env.reset()
        # TODO: Are we doing deterministic or stochastic evaluation? BIG difference!
        while not done:
            # Prepare observation and memory
            obs = torch.tensor(np.expand_dims(obs, 0), dtype=torch.float32, device=device)
            in_memory = memory[0, memory_indices[t].unsqueeze(0)]
            t_ = max(0, min(t, memory_length - 1))
            mask = memory_mask[t_].unsqueeze(0)
            indices = memory_indices[t].unsqueeze(0)
            # Render environment
            env.render()
            # Forward model
            policy, value, new_memory = model(obs, in_memory, mask, indices)
            memory[:, t] = new_memory
            # Sample action
            action = [] 
            for action_branch in policy:
                action.append(action_branch.sample().item())

            # Step environemnt
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            t += 1

        total_episode_rewards.append((sum(episode_rewards)))
        total_episode_lengths.append(t)
        print("Eval Episode", i, "length:", str(t), "reward: " + str(sum(episode_rewards)))

    score = np.mean(total_episode_rewards)
    print("Mean episode length:", str(np.mean(total_episode_lengths)))
    print("Mean episode reward: " + str(score))

    env.close()

    return score