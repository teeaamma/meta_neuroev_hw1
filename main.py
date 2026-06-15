import numpy as np
from config import RANDOM_SEED
from train import train_all
from evaluate import evaluate_all


def main():
    np.random.seed(RANDOM_SEED)

    trained = train_all()

    results, report = evaluate_all(trained, ttest_metric="energy")

    print("\n=== Evaluation summary ===")
    for name, m in results.items():
        print(
            f"{name} | "
            f"success={m['success_rate']:.2f}% | "
            f"collision={m['collision_rate']:.2f}% | "
            f"mean_time={m['mean_time']} | "
            f"mean_energy={m['mean_energy']}"
        )

    print("\n=== t-test ===")
    for pair, stat in report["comparisons"].items():
        print(f"{pair} | t={stat['statistic']} | p={stat['p_value']}")


if __name__ == "__main__":
    main()