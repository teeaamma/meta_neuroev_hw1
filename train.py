from config import *
from environment import DroneEnvironment
from q_agent import QLearningAgent
from prospect_agent import ProspectAgent
from risk_sensitive_agent import RiskSensitiveAgent
import numpy as np


def avg(a, w):
    if len(a) == 0:
        return np.array([])
    if len(a) < w:
        return np.array([np.mean(a)], dtype=float)
    k = np.ones(w, dtype=float) / w
    return np.convolve(np.array(a, dtype=float), k, mode="valid")


def train_agent(agent, env, episodes=TRAIN_EPISODES,
                eps_start=EPS_START, eps_min=EPS_MIN, eps_decay=EPS_DECAY):

    eps = float(eps_start)

    rewards, lengths, collisions, energies, successes = [], [], [], [], []

    for _ in range(episodes):
        st = env.reset()
        done = False

        total_reward = 0.0
        ep_len = 0
        ep_collisions = 0
        ep_success = False

        while not done:
            a = agent.choose_action(st, eps)
            ns, rw, done, info = env.step(a)

            agent.update(st, a, rw, ns, done)

            st = ns
            total_reward += rw
            ep_len += 1
            ep_collisions += int(info["collision"])
            ep_success = ep_success or bool(info["success"])

        rewards.append(total_reward)
        lengths.append(ep_len)
        collisions.append(ep_collisions)
        energies.append(env.total_energy)
        successes.append(ep_success)

        eps = max(eps_min, eps * eps_decay)

    return {
        "rewards": rewards,
        "lengths": lengths,
        "collisions": collisions,
        "energies": energies,
        "successes": successes,
    }


def train_all():
    np.random.seed(RANDOM_SEED)

    results = {}

    for name in ["q", "prospect", "risk"]:
        print(f"Training {name}...")

        env = DroneEnvironment()

        if name == "q":
            agent = QLearningAgent(env.state_shape)
        elif name == "prospect":
            agent = ProspectAgent(env.state_shape)
        else:
            agent = RiskSensitiveAgent(env.state_shape)

        train_res = train_agent(agent, env)

        results[name] = {
            "agent": agent,
            "train": train_res
        }

    return results