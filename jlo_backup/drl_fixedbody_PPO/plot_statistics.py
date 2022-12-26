import pandas as pd
from matplotlib import pyplot as plt
import argparse


def plot(database: str):
    data = pd.read_csv(database + '/statistics.csv')

    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(1, 2, 1)
    ax.title.set_text("Mean reward for each action")
    plt.plot(data['mean_rew'])
    ax = fig.add_subplot(1, 2, 2)
    ax.title.set_text("Mean value for each state")
    plt.plot(data['mean_val'])
    plt.show()

def main() -> None:
    """Run the program."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "database",
        type=str,
        help="The database to plot.",
    )
    args = parser.parse_args()

    plot(args.database)


if __name__ == "__main__":
    main()