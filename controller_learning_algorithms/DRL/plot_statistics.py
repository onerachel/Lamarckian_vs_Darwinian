import pandas as pd
from matplotlib import pyplot as plt
import argparse


def plot(database: str):
    data = pd.read_csv(database + '/statistics.csv')
    num_stats = len(data.columns)

    fig = plt.figure(figsize=(7*num_stats, 6))
    for i, stat in enumerate(data.columns):
        ax = fig.add_subplot(1, num_stats, i+1)
        ax.title.set_text(stat)
        plt.plot(data[stat])
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