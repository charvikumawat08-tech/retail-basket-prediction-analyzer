"""Your code. One stub per milestone; build them in order and keep each one small."""

import numpy as np

from app.storage import Store


# --- Milestone 1: Generate a synthetic dataset of shopping baskets with planted itemsets.
# Hint: Use the data generation techniques from 'Teach a Language Model What Sells Together' to create your dataset.
def milestone_1(
    num_baskets: int = 2000,
    train_ratio: float = 0.8,
    bundle_prob: float = 0.85,
    seed: int = 42,
    store: Store | None = None,
) -> dict:
    if num_baskets < 2:
        raise ValueError("num_baskets must be at least 2")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= bundle_prob <= 1:
        raise ValueError("bundle_prob must be between 0 and 1")

    retail_items = [
        "pasta", "marinara_sauce", "parmesan", "coffee_beans", "paper_filters",
        "oat_milk", "tortilla_chips", "salsa", "guacamole", "chicken_breast",
        "curry_paste", "coconut_milk", "salmon", "lemon", "asparagus",
        "yogurt", "granola", "blueberries", "olive_oil", "garlic", "basil",
        "rice", "black_beans", "cheddar", "eggs", "spinach", "bananas",
        "bread", "peanut_butter", "apples",
    ]
    planted_itemsets = [
        ["pasta", "marinara_sauce", "parmesan"],
        ["coffee_beans", "paper_filters", "oat_milk"],
        ["tortilla_chips", "salsa", "guacamole"],
        ["chicken_breast", "curry_paste", "coconut_milk"],
        ["salmon", "lemon", "asparagus"],
        ["yogurt", "granola", "blueberries"],
    ]
    vocab = ["<PAD>", "<BOS>", "<EOS>", *retail_items]
    ranks = np.arange(1, len(retail_items) + 1, dtype=float)
    item_weights = 1 / np.power(ranks, 1.2)
    rng = np.random.default_rng(seed)

    baskets = []
    for _ in range(num_baskets):
        bundle = None
        if rng.random() < bundle_prob:
            bundle = planted_itemsets[int(rng.integers(len(planted_itemsets)))]

        available_items = [
            item for item in retail_items if bundle is None or item not in bundle
        ]
        weights = np.array(
            [item_weights[retail_items.index(item)] for item in available_items],
            dtype=float,
        )
        weights /= weights.sum()
        background_count = int(rng.integers(2, 6))
        background = rng.choice(
            available_items, size=background_count, replace=False, p=weights
        ).tolist()
        background = rng.permutation(background).tolist()

        if bundle:
            insertion = int(rng.integers(len(background) + 1))
            items = background[:insertion] + bundle + background[insertion:]
        else:
            items = background
        baskets.append(["<BOS>", *items, "<EOS>"])

    baskets = [baskets[index] for index in rng.permutation(num_baskets)]
    train_size = min(max(int(num_baskets * train_ratio), 1), num_baskets - 1)
    dataset = {
        "vocab": vocab,
        "planted_itemsets": planted_itemsets,
        "train_baskets": baskets[:train_size],
        "test_baskets": baskets[train_size:],
        "seed": seed,
    }

    if store is None:
        store = Store()
    store.put("dataset", dataset)
    return dataset


# --- Milestone 2: Implement the n-gram model to predict next items in a basket.
# Hint: Apply the n-gram model building technique from 'Build a Next-Word Predictor from Scratch' to your basket sequences.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Pretrain a small language model (like in the drill) on the basket sequences.
# Hint: Adapt the language model training process from 'Teach a Language Model What Sells Together' for your basket data.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).
# Hint: Implement evaluation metrics similar to those discussed in 'Build a Next-Word Predictor from Scratch' for next-item prediction.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Present a comparison of the models' performance.
# Hint: Summarize your findings and present the comparison clearly, perhaps using visualizations.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
