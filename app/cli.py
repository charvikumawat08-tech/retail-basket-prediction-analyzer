"""Command-line entry point: `python -m app.cli --help`."""
import argparse

from app.core import milestone_1, milestone_2, milestone_3, milestone_4, milestone_5


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="retail-basket-prediction-analyzer", description="Retail analysts use this to compare next-item prediction accuracy between an n-gram model and a small LM, both trained on simulated shopping baskets data.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    sub.add_parser("generate", help="generate the synthetic shopping basket dataset")
    sub.add_parser("train-ngram", help="train the n-gram next-item predictor")
    sub.add_parser("train-lm", help="pretrain the small neural basket language model")
    sub.add_parser("evaluate", help="evaluate next-item prediction on the test baskets")
    sub.add_parser("compare", help="present the retail analyst comparison report")
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
    elif args.command == "train-lm":
        summary = milestone_3()
        print(
            f"Training loss: {summary['initial_loss']:.4f} -> "
            f"{summary['final_loss']:.4f} over {summary['epochs']} epochs."
        )
        sample = summary["sample_prediction"]
        print(
            f"Sample prediction for {sample['prefix']}: {sample['item']} "
            f"(probability {sample['probability']:.3f})."
        )
    elif args.command == "evaluate":
        evaluation = milestone_4()
        for model_name, results in evaluation.items():
            print(
                f"{model_name}: Hit@1={results['top1_accuracy']:.3f}, "
                f"Hit@3={results['top3_accuracy']:.3f}, "
                f"Hit@5={results['top5_accuracy']:.3f}, "
                f"MRR={results['mrr']:.3f}, "
                f"mean rank={results['mean_rank']:.2f}"
            )
            for segment in ("planted_itemset", "general"):
                metrics = results[segment]
                if metrics["count"]:
                    print(
                        f"  {segment}: Hit@1={metrics['top1_accuracy']:.3f}, "
                        f"Hit@3={metrics['top3_accuracy']:.3f}, "
                        f"Hit@5={metrics['top5_accuracy']:.3f}, "
                        f"MRR={metrics['mrr']:.3f}, "
                        f"mean rank={metrics['mean_rank']:.2f} "
                        f"(n={metrics['count']})"
                    )
    elif args.command == "compare":
        report = milestone_5()
        print(report["formatted_report"])


if __name__ == "__main__":
    main()
