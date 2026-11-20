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
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Pretrain a small language model (like in the drill) on the basket sequences.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Present a comparison of the models' performance.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

