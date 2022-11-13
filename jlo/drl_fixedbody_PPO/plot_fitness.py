import pandas as pd
from matplotlib import pyplot as plt
import argparse

def plot(database: str):
    data = pd.read_csv(database + '/fitnesses.csv')

    plt.plot(data['fitness'][:])
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