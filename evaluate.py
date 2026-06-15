from config import *
from environment import DroneEnvironment
from train import avg

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy.stats import ttest_ind


def test_agent(agent, env):
    rewards = []
    lengths = []
    collisions = []
    energies = []
    success = []

    success_times = []
    success_energies = []
    trajectories = []

    final_batteries = []
    final_positions = []

    for _ in range(TEST_EPISODES):
        state = env.reset()
        done = False

        total_reward = 0.0
        episode_length = 0
        episode_collision = 0
        episode_success = False

        while not done:
            action = agent.choose_action(state, TEST_EPSILON)
            state, reward, done, info = env.step(action)

            total_reward += reward
            episode_length += 1
            episode_collision += int(info["collision"])
            episode_success = episode_success or bool(info["success"])

        rewards.append(total_reward)
        lengths.append(episode_length)
        collisions.append(episode_collision > 0)
        energies.append(env.total_energy)
        success.append(episode_success)

        final_batteries.append(env.battery)
        final_positions.append(env.trajectory[-1])

        if len(trajectories) < 4:
            trajectories.append(list(env.trajectory))

        if episode_success:
            success_times.append(episode_length)
            success_energies.append(env.total_energy)

    return {
        "rewards": rewards,
        "lengths": lengths,
        "collisions": collisions,
        "energies": energies,
        "success": success,
        "success_times": success_times,
        "success_energies": success_energies,
        "trajectories": trajectories,
        "final_batteries": final_batteries,
        "final_positions": final_positions,
        "success_rate": float(np.mean(success) * 100.0),
        "collision_rate": float(np.mean(collisions) * 100.0),
        "mean_time": float(np.mean(success_times)) if success_times else float("nan"),
        "mean_energy": float(np.mean(energies)) if energies else float("nan"),
    }


def plot_curve(name, rewards, directory):
    plt.figure(figsize=(9, 4))

    values = avg(rewards, LEARNING_CURVE_WINDOW)
    if len(values):
        plt.plot(values)

    plt.title(f"{name} learning curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(directory, "learning_curve.png"), dpi=150)
    plt.close()


def plot_cdf(name, times, directory):
    plt.figure(figsize=(7, 4))

    if times:
        sorted_times = np.sort(np.array(times))
        cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        plt.plot(sorted_times, cdf)

    plt.title(f"{name} CDF of success time")
    plt.xlabel("Success time")
    plt.ylabel("CDF")
    plt.tight_layout()
    plt.savefig(os.path.join(directory, "cdf_success_time.png"), dpi=150)
    plt.close()


def plot_heatmap(name, positions, batteries, directory):
    plt.figure(figsize=(7, 4))

    if positions:
        xs = np.array([p[0] for p in positions])
        ys = np.array([p[1] for p in positions])
        bs = np.array(batteries)

        weighted_sum, x_edges, y_edges = np.histogram2d(xs, ys, bins=25, weights=bs)
        counts, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])

        heatmap = np.divide(
            weighted_sum,
            counts,
            out=np.zeros_like(weighted_sum),
            where=counts > 0,
        )

        plt.imshow(
            heatmap.T,
            origin="lower",
            aspect="auto",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        )
        plt.colorbar()

    plt.title(f"{name} final battery heatmap")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(os.path.join(directory, "final_battery_heatmap.png"), dpi=150)
    plt.close()


def plot_traj(name, trajectories, directory):
    fig, ax = plt.subplots(figsize=(10, 6))

    for trajectory in trajectories[:4]:
        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]

        ax.plot(xs, ys, linewidth=1.5)
        ax.scatter(xs[0], ys[0], marker="o", s=40)
        ax.scatter(xs[-1], ys[-1], marker="x", s=40)

    for x1, y1, x2, y2 in OBSTACLES:
        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=True,
                alpha=0.3,
            )
        )

    ax.add_patch(
        Circle(
            (STATION_X, STATION_Y),
            STATION_RADIUS,
            fill=False,
            linewidth=2,
        )
    )

    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, Y_MAX)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{name} trajectories")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(directory, "trajectories.png"), dpi=150)
    plt.close()


def evaluate_all(trained, ttest_metric="energy"):
    np.random.seed(RANDOM_SEED)

    results = {}
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, data in trained.items():
        directory = os.path.join(OUTPUT_DIR, name)
        os.makedirs(directory, exist_ok=True)

        env = DroneEnvironment()
        result = test_agent(data["agent"], env)
        results[name] = result

        plot_curve(name, data["train"]["rewards"], directory)
        plot_cdf(name, result["success_times"], directory)
        plot_heatmap(name, result["final_positions"], result["final_batteries"], directory)
        plot_traj(name, result["trajectories"], directory)

    if ttest_metric == "time":
        metric_values = lambda result: result["success_times"]
        metric_name = "time"
    elif ttest_metric == "energy":
        metric_values = lambda result: result["energies"]
        metric_name = "energy"
    else:
        raise ValueError("ttest_metric must be either 'time' or 'energy'")

    comparisons = {}
    pairs = [
        ("q", "prospect"),
        ("q", "risk"),
        ("prospect", "risk"),
    ]

    for first, second in pairs:
        x = metric_values(results[first])
        y = metric_values(results[second])

        if len(x) > 1 and len(y) > 1:
            statistic, p_value = ttest_ind(x, y, equal_var=False)
            comparisons[f"{first}_vs_{second}"] = {
                "metric": metric_name,
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
        else:
            comparisons[f"{first}_vs_{second}"] = {
                "metric": metric_name,
                "statistic": None,
                "p_value": None,
            }

    summary = {
        name: {
            "success_rate": result["success_rate"],
            "collision_rate": result["collision_rate"],
            "mean_time": result["mean_time"],
            "mean_energy": result["mean_energy"],
        }
        for name, result in results.items()
    }

    return results, {
        "summary": summary,
        "comparisons": comparisons,
        "ttest_metric": metric_name,
    }