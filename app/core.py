"""Your code. One stub per milestone; build them in order and keep each one small."""

from collections import Counter

import numpy as np

from app.storage import Store


class NGramBasketPredictor:
    _IGNORED_TARGETS = {"<PAD>", "<BOS>"}
    _REQUIRED_SPECIAL_TOKENS = {"<PAD>", "<BOS>", "<EOS>"}

    def __init__(
        self, vocab: list[str] | None = None, smoothing: float = 0.5
    ) -> None:
        if not np.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("smoothing must be greater than 0")
        self.vocab = list(vocab or [])
        self.smoothing = smoothing
        self.unigram_counts: Counter[str] = Counter()
        self.bigram_counts: dict[tuple[str, ...], Counter[str]] = {}
        self.trigram_counts: dict[tuple[str, ...], Counter[str]] = {}
        if self.vocab:
            self._validate_vocab()

    def _validate_vocab(self) -> None:
        if len(set(self.vocab)) != len(self.vocab):
            raise ValueError("vocab must not contain duplicate tokens")
        if not self._REQUIRED_SPECIAL_TOKENS.issubset(self.vocab):
            raise ValueError("vocab must include <PAD>, <BOS>, and <EOS>")

    def fit(
        self,
        baskets: list[list[str]],
        vocab: list[str] | None = None,
    ) -> "NGramBasketPredictor":
        if vocab is not None:
            self.vocab = list(vocab)
        elif not self.vocab:
            observed = {
                token
                for basket in baskets
                for token in basket
                if token not in self._REQUIRED_SPECIAL_TOKENS
            }
            self.vocab = [
                "<PAD>", "<BOS>", "<EOS>", *sorted(observed)
            ]
        self._validate_vocab()
        vocab_items = set(self.vocab)
        self.unigram_counts = Counter()
        self.bigram_counts = {}
        self.trigram_counts = {}

        for basket in baskets:
            for index, target in enumerate(basket):
                if target not in vocab_items:
                    raise ValueError(f"basket token {target!r} is not in vocab")
                if target in self._IGNORED_TARGETS:
                    continue

                self.unigram_counts[target] += 1
                if index >= 1:
                    bigram_context = (basket[index - 1],)
                    self.bigram_counts.setdefault(
                        bigram_context, Counter()
                    )[target] += 1
                if index >= 2:
                    trigram_context = (basket[index - 2], basket[index - 1])
                    self.trigram_counts.setdefault(
                        trigram_context, Counter()
                    )[target] += 1

        return self

    @property
    def target_items(self) -> list[str]:
        return [token for token in self.vocab if token not in self._IGNORED_TARGETS]

    def _smoothed_distribution(self, counts: Counter[str]) -> dict[str, float]:
        target_items = self.target_items
        denominator = sum(counts.values()) + self.smoothing * len(target_items)
        return {
            item: (counts.get(item, 0) + self.smoothing) / denominator
            for item in target_items
        }

    def predict_proba(self, prefix: list[str]) -> dict[str, float]:
        if len(prefix) >= 2:
            context = (prefix[-2], prefix[-1])
            if context in self.trigram_counts:
                return self._smoothed_distribution(self.trigram_counts[context])

        if prefix:
            context = (prefix[-1],)
            if context in self.bigram_counts:
                return self._smoothed_distribution(self.bigram_counts[context])

        return self._smoothed_distribution(self.unigram_counts)

    def to_dict(self) -> dict:
        def serialize_context_counts(
            context_counts: dict[tuple[str, ...], Counter[str]],
        ) -> list[dict]:
            return [
                {
                    "context": list(context),
                    "target": target,
                    "count": count,
                }
                for context, counts in sorted(context_counts.items())
                for target, count in sorted(counts.items())
            ]

        return {
            "vocab": self.vocab,
            "smoothing": self.smoothing,
            "unigram_counts": dict(self.unigram_counts),
            "bigram_counts": serialize_context_counts(self.bigram_counts),
            "trigram_counts": serialize_context_counts(self.trigram_counts),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NGramBasketPredictor":
        predictor = cls(data["vocab"], data["smoothing"])
        predictor.unigram_counts = Counter(data["unigram_counts"])

        for name, context_size, context_counts in (
            ("bigram_counts", 1, predictor.bigram_counts),
            ("trigram_counts", 2, predictor.trigram_counts),
        ):
            for entry in data[name]:
                context = tuple(entry["context"])
                if len(context) != context_size:
                    raise ValueError(f"{name} contains an invalid context")
                context_counts.setdefault(context, Counter())[entry["target"]] = (
                    entry["count"]
                )
        return predictor


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
def milestone_2(smoothing: float = 0.5, store: Store | None = None) -> dict:
    if store is None:
        store = Store()

    dataset = store.get("dataset")
    if dataset is None:
        dataset = milestone_1(store=store)
    if not isinstance(dataset, dict):
        raise ValueError("stored dataset must be a dictionary")

    vocab = dataset.get("vocab")
    train_baskets = dataset.get("train_baskets")
    planted_itemsets = dataset.get("planted_itemsets")
    if not isinstance(vocab, list) or not isinstance(train_baskets, list):
        raise ValueError("stored dataset must include vocab and train_baskets lists")
    if not isinstance(planted_itemsets, list) or not planted_itemsets:
        raise ValueError("stored dataset must include planted itemsets")

    predictor = NGramBasketPredictor(vocab, smoothing).fit(train_baskets)
    store.put("ngram_model", predictor.to_dict())

    sample_prefix = ["<BOS>", *planted_itemsets[0][:-1]]
    sample_probabilities = predictor.predict_proba(sample_prefix)
    sample_item = max(sample_probabilities, key=lambda item: sample_probabilities[item])
    return {
        "vocab_size": len(predictor.vocab),
        "bigram_context_count": len(predictor.bigram_counts),
        "trigram_context_count": len(predictor.trigram_counts),
        "sample_prediction": {
            "prefix": sample_prefix,
            "item": sample_item,
            "probability": sample_probabilities[sample_item],
        },
    }


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
