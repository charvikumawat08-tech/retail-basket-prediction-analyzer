"""Command-line entry point: `python -m app.cli --help`."""
import argparse

from app.core import milestone_1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="retail-basket-prediction-analyzer", description="Retail analysts use this to compare next-item prediction accuracy between an n-gram model and a small LM, both trained on simulated shopping baskets data.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    sub.add_parser("generate", help="generate the synthetic shopping basket dataset")
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Retail Basket Prediction Analyzer is ready. Open README.md for your milestones.")
    elif args.command == "generate":
        dataset = milestone_1()
        print(
            f"Generated {len(dataset['train_baskets'])} train and "
            f"{len(dataset['test_baskets'])} test baskets; "
            f"vocabulary size: {len(dataset['vocab'])}."
        )


if __name__ == "__main__":
    main()
