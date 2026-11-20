"""Command-line entry point: `python -m app.cli --help`."""
import argparse

from app.core import milestone_1, milestone_2


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="retail-basket-prediction-analyzer", description="Retail analysts use this to compare next-item prediction accuracy between an n-gram model and a small LM, both trained on simulated shopping baskets data.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    sub.add_parser("generate", help="generate the synthetic shopping basket dataset")
    sub.add_parser("train-ngram", help="train the n-gram next-item predictor")
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
    elif args.command == "train-ngram":
        summary = milestone_2()
        sample = summary["sample_prediction"]
        print(
            f"Trained n-gram model: vocabulary size {summary['vocab_size']}, "
            f"{summary['bigram_context_count']} bigram contexts, "
            f"{summary['trigram_context_count']} trigram contexts."
        )
        print(
            f"Sample prediction for {sample['prefix']}: {sample['item']} "
            f"(probability {sample['probability']:.3f})."
        )


if __name__ == "__main__":
    main()
