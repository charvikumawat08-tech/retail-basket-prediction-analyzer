"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

# Milestone 1: Generate a synthetic dataset of shopping baskets with planted itemsets.
def test_milestone_1(tmp_path):
    from app.core import milestone_1
    from app.storage import Store

    store = Store(str(tmp_path / "app.json"))
    dataset = milestone_1(num_baskets=1000, train_ratio=0.8, seed=7, store=store)

    assert len(dataset["train_baskets"]) == 800
    assert len(dataset["test_baskets"]) == 200
    assert dataset["seed"] == 7
    assert {"<PAD>", "<BOS>", "<EOS>"}.issubset(dataset["vocab"])
    assert len(dataset["vocab"]) >= 30
    assert len(dataset["planted_itemsets"]) >= 6

    vocabulary = set(dataset["vocab"])
    all_baskets = dataset["train_baskets"] + dataset["test_baskets"]
    assert all(
        basket[0] == "<BOS>"
        and basket[-1] == "<EOS>"
        and set(basket).issubset(vocabulary)
        for basket in all_baskets
    )

    retail_vocabulary_size = len(vocabulary) - 3
    random_chance = 1 / retail_vocabulary_size**3
    for itemset in dataset["planted_itemsets"]:
        occurrence_rate = sum(
            any(
                basket[index:index + len(itemset)] == itemset
                for index in range(1, len(basket) - len(itemset))
            )
            for basket in all_baskets
        ) / len(all_baskets)
        assert occurrence_rate > random_chance * 100
        assert occurrence_rate > 0.05

    assert store.get("dataset") == dataset


# Milestone 2: Implement the n-gram model to predict next items in a basket.
def test_milestone_2(tmp_path):
    from app.core import NGramBasketPredictor, milestone_1, milestone_2
    from app.storage import Store

    store = Store(str(tmp_path / "app.json"))
    dataset = milestone_1(num_baskets=1000, seed=13, store=store)
    summary = milestone_2(smoothing=0.5, store=store)

    serialized_model = store.get("ngram_model")
    assert serialized_model is not None
    predictor = NGramBasketPredictor.from_dict(serialized_model)
    assert summary["vocab_size"] == len(dataset["vocab"])
    assert summary["bigram_context_count"] > 0
    assert summary["trigram_context_count"] > 0

    prefix = ["<BOS>", "pasta", "marinara_sauce"]
    probabilities = predictor.predict_proba(prefix)
    assert set(probabilities) == set(dataset["vocab"]) - {"<PAD>", "<BOS>"}
    assert all(probability >= 0 for probability in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)
    ranked_items = sorted(probabilities, key=probabilities.get, reverse=True)
    assert "parmesan" in ranked_items[:2]
    assert all(
        probabilities["parmesan"] > probabilities[item]
        for item in ("apples", "bananas", "spinach")
    )
    assert summary["sample_prediction"]["item"] == "parmesan"

    bigram_probabilities = predictor.predict_proba(["parmesan"])
    assert ("parmesan",) in predictor.bigram_counts
    assert sum(bigram_probabilities.values()) == pytest.approx(1.0)

    unseen_probabilities = predictor.predict_proba(
        ["<BOS>", "never_before_seen_item"]
    )
    assert sum(unseen_probabilities.values()) == pytest.approx(1.0)
    assert all(probability > 0 for probability in unseen_probabilities.values())

    empty_store = Store(str(tmp_path / "generated.json"))
    generated_summary = milestone_2(store=empty_store)
    assert empty_store.get("dataset") is not None
    assert empty_store.get("ngram_model") is not None
    assert generated_summary["vocab_size"] == len(empty_store.get("dataset")["vocab"])


# Milestone 3: Pretrain a small language model (like in the drill) on the basket sequences.
def test_milestone_3(tmp_path):
    from app.core import SmallBasketLM, milestone_1, milestone_3
    from app.storage import Store

    store = Store(str(tmp_path / "app.json"))
    dataset = milestone_1(num_baskets=1000, seed=23, store=store)
    summary = milestone_3(store=store)

    assert summary["final_loss"] < summary["initial_loss"]
    assert summary["epochs"] == 25
    assert summary["vocab_size"] == len(dataset["vocab"])
    serialized_model = store.get("small_lm_model")
    assert serialized_model is not None

    model = SmallBasketLM.from_dict(serialized_model)
    prefix = ["<BOS>", "pasta", "marinara_sauce"]
    probabilities = model.predict_proba(prefix)
    assert set(probabilities) == set(dataset["vocab"]) - {"<PAD>", "<BOS>"}
    assert all(probability >= 0 for probability in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)
    ranked_items = sorted(probabilities, key=probabilities.get, reverse=True)
    assert "parmesan" in ranked_items[:2]
    assert summary["sample_prediction"]["item"] == "parmesan"

    restored_model = SmallBasketLM.from_dict(model.to_dict())
    assert restored_model.predict_proba(prefix) == probabilities

    empty_store = Store(str(tmp_path / "generated.json"))
    generated_summary = milestone_3(store=empty_store, epochs=2)
    assert empty_store.get("dataset") is not None
    assert empty_store.get("small_lm_model") is not None
    assert generated_summary["vocab_size"] == len(empty_store.get("dataset")["vocab"])


# Milestone 4: Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).
def test_milestone_4(tmp_path):
    from app.core import milestone_4
    from app.storage import Store

    store = Store(str(tmp_path / "app.json"))
    evaluation = milestone_4(store=store)

    assert store.get("dataset") is not None
    assert store.get("ngram_model") is not None
    assert store.get("small_lm_model") is not None
    assert store.get("evaluation") == evaluation

    for model_name in ("ngram", "small_lm"):
        results = evaluation[model_name]
        for metric in (
            "top1_accuracy",
            "top3_accuracy",
            "top5_accuracy",
            "mrr",
        ):
            assert 0.0 <= results[metric] <= 1.0
        assert results["top1_accuracy"] <= results["top3_accuracy"]
        assert results["top3_accuracy"] <= results["top5_accuracy"]
        assert results["mean_rank"] >= 1.0

        for segment in ("planted_itemset", "general"):
            metrics = results[segment]
            assert metrics["count"] > 0
            for metric in (
                "top1_accuracy",
                "top3_accuracy",
                "top5_accuracy",
                "mrr",
            ):
                assert 0.0 <= metrics[metric] <= 1.0
            assert metrics["top1_accuracy"] <= metrics["top3_accuracy"]
            assert metrics["top3_accuracy"] <= metrics["top5_accuracy"]
            assert metrics["mean_rank"] >= 1.0
        assert results["planted_itemset"]["top3_accuracy"] > 0.5


# Milestone 5: Present a comparison of the models' performance.
def test_milestone_5(tmp_path):
    from app.core import milestone_1, milestone_4, milestone_5
    from app.storage import Store

    existing_store = Store(str(tmp_path / "existing.json"))
    milestone_1(num_baskets=400, seed=31, store=existing_store)
    milestone_4(store=existing_store)
    report = milestone_5(store=existing_store)

    assert set(report) >= {
        "summary_table",
        "spot_checks",
        "takeaways",
        "formatted_report",
    }
    assert report["summary_table"]["rows"]
    assert "NGramBasketPredictor" in report["summary_table"]["columns"]
    assert "SmallBasketLM" in report["summary_table"]["columns"]
    assert {
        row["Segment"] for row in report["summary_table"]["rows"]
    } == {"Overall", "Planted-Itemset", "General"}
    assert report["summary_table"]["formatted"]
    assert "Hit@1" in report["bar_chart"]
    assert "#" in report["bar_chart"]
    assert len(report["spot_checks"]) >= 3
    assert report["takeaways"]["overall_leader"]
    assert report["takeaways"]["recommendations"]
    assert report["takeaways"]["planted_itemset_leader"]
    assert report["takeaways"]["general_transition_leader"]
    assert report["formatted_report"]
    assert "NGramBasketPredictor" in report["formatted_report"]
    assert "SmallBasketLM" in report["formatted_report"]
    assert "Top-1 Accuracy" in report["formatted_report"]
    overall_ngram_top1 = existing_store.get("evaluation")["ngram"]["top1_accuracy"]
    assert f"{overall_ngram_top1:.3f}" in report["formatted_report"]
    assert existing_store.get("comparison_report") == report

    for spot_check in report["spot_checks"]:
        for model_name in ("NGramBasketPredictor", "SmallBasketLM"):
            assert len(spot_check["recommendations"][model_name]) == 3

    fresh_store = Store(str(tmp_path / "fresh.json"))
    fresh_report = milestone_5(store=fresh_store)
    assert fresh_store.get("dataset") is not None
    assert fresh_store.get("evaluation") is not None
    assert fresh_store.get("comparison_report") == fresh_report
    assert "MRR" in fresh_report["formatted_report"]

