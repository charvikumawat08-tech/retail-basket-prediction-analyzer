"""Your code. One stub per milestone; build them in order and keep each one small."""

from collections import Counter

import numpy as np
import pandas as pd

from app.storage import Store


class SmallBasketLM:
    _IGNORED_TARGETS = {"<PAD>", "<BOS>"}
    _REQUIRED_SPECIAL_TOKENS = {"<PAD>", "<BOS>", "<EOS>"}
    _ADAM_BETA1 = 0.9
    _ADAM_BETA2 = 0.999
    _ADAM_EPSILON = 1e-8
    _BATCH_SIZE = 128

    def __init__(
        self,
        vocab: list[str],
        embed_dim: int = 24,
        hidden_dim: int = 48,
        context_size: int = 3,
        seed: int = 42,
    ) -> None:
        if embed_dim < 1 or hidden_dim < 1 or context_size < 1:
            raise ValueError("model dimensions must be positive")
        if len(set(vocab)) != len(vocab):
            raise ValueError("vocab must not contain duplicate tokens")
        if not self._REQUIRED_SPECIAL_TOKENS.issubset(vocab):
            raise ValueError("vocab must include <PAD>, <BOS>, and <EOS>")

        self.vocab = list(vocab)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_size = context_size
        self.seed = seed
        self.target_items = [
            token for token in self.vocab if token not in self._IGNORED_TARGETS
        ]
        self._token_to_id = {token: index for index, token in enumerate(self.vocab)}
        self._target_to_id = {
            token: index for index, token in enumerate(self.target_items)
        }

        rng = np.random.default_rng(seed)
        input_dim = context_size * embed_dim
        output_dim = len(self.target_items)
        self.E = rng.normal(0, 0.1, size=(len(self.vocab), embed_dim))
        self.W1 = rng.normal(
            0, np.sqrt(2 / (input_dim + hidden_dim)), size=(input_dim, hidden_dim)
        )
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(
            0, np.sqrt(2 / (hidden_dim + output_dim)), size=(hidden_dim, output_dim)
        )
        self.b2 = np.zeros(output_dim)
        self.loss_history: list[float] = []

    def _contexts_from_prefix(self, prefix: list[str]) -> np.ndarray:
        tokens = prefix[-self.context_size:]
        for token in tokens:
            if token not in self._token_to_id:
                raise ValueError(f"prefix token {token!r} is not in vocab")
        tokens = ["<PAD>"] * (self.context_size - len(tokens)) + tokens
        return np.array(
            [[self._token_to_id[token] for token in tokens]], dtype=np.int64
        )

    def _forward(
        self, contexts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        embedded = self.E[contexts].reshape(len(contexts), -1)
        hidden = np.tanh(embedded @ self.W1 + self.b1)
        logits = hidden @ self.W2 + self.b2
        logits -= logits.max(axis=1, keepdims=True)
        exponentials = np.exp(logits)
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        return embedded, hidden, probabilities

    def _cross_entropy(self, contexts: np.ndarray, targets: np.ndarray) -> float:
        _, _, probabilities = self._forward(contexts)
        target_probabilities = probabilities[np.arange(len(targets)), targets]
        return float(-np.log(np.maximum(target_probabilities, 1e-12)).mean())

    def fit(
        self,
        baskets: list[list[str]],
        epochs: int = 25,
        lr: float = 0.03,
    ) -> "SmallBasketLM":
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be greater than 0")

        context_rows = []
        target_rows = []
        for basket in baskets:
            for token in basket:
                if token not in self._token_to_id:
                    raise ValueError(f"basket token {token!r} is not in vocab")
            for index in range(1, len(basket)):
                target = basket[index]
                if target in self._IGNORED_TARGETS:
                    continue
                context = basket[max(0, index - self.context_size):index]
                context = ["<PAD>"] * (self.context_size - len(context)) + context
                context_rows.append(
                    [self._token_to_id[token] for token in context]
                )
                target_rows.append(self._target_to_id[target])

        if not target_rows:
            raise ValueError("training baskets contain no next-item targets")

        contexts = np.asarray(context_rows, dtype=np.int64)
        targets = np.asarray(target_rows, dtype=np.int64)
        self.loss_history = [self._cross_entropy(contexts, targets)]

        parameters = {
            "E": self.E,
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }
        first_moments = {name: np.zeros_like(value) for name, value in parameters.items()}
        second_moments = {name: np.zeros_like(value) for name, value in parameters.items()}
        rng = np.random.default_rng(self.seed)
        step = 0

        for _ in range(epochs):
            epoch_order = rng.permutation(len(targets))
            for batch_start in range(0, len(epoch_order), self._BATCH_SIZE):
                batch_indices = epoch_order[
                    batch_start:batch_start + self._BATCH_SIZE
                ]
                batch_contexts = contexts[batch_indices]
                batch_targets = targets[batch_indices]
                embedded, hidden, probabilities = self._forward(batch_contexts)
                gradient_logits = probabilities
                gradient_logits[
                    np.arange(len(batch_targets)), batch_targets
                ] -= 1
                gradient_logits /= len(batch_targets)

                gradient_W2 = hidden.T @ gradient_logits
                gradient_b2 = gradient_logits.sum(axis=0)
                gradient_hidden = gradient_logits @ self.W2.T
                gradient_hidden *= 1 - hidden**2
                gradient_W1 = embedded.T @ gradient_hidden
                gradient_b1 = gradient_hidden.sum(axis=0)
                gradient_embedded = (gradient_hidden @ self.W1.T).reshape(
                    len(batch_targets), self.context_size, self.embed_dim
                )
                gradient_E = np.zeros_like(self.E)
                np.add.at(
                    gradient_E,
                    batch_contexts.reshape(-1),
                    gradient_embedded.reshape(-1, self.embed_dim),
                )

                gradients = {
                    "E": gradient_E,
                    "W1": gradient_W1,
                    "b1": gradient_b1,
                    "W2": gradient_W2,
                    "b2": gradient_b2,
                }
                step += 1
                for name, gradient in gradients.items():
                    first_moments[name] *= self._ADAM_BETA1
                    first_moments[name] += (1 - self._ADAM_BETA1) * gradient
                    second_moments[name] *= self._ADAM_BETA2
                    second_moments[name] += (1 - self._ADAM_BETA2) * gradient**2
                    corrected_first = first_moments[name] / (
                        1 - self._ADAM_BETA1**step
                    )
                    corrected_second = second_moments[name] / (
                        1 - self._ADAM_BETA2**step
                    )
                    parameters[name] -= lr * corrected_first / (
                        np.sqrt(corrected_second) + self._ADAM_EPSILON
                    )

            self.loss_history.append(self._cross_entropy(contexts, targets))

        return self

    def predict_proba(self, prefix: list[str]) -> dict[str, float]:
        contexts = self._contexts_from_prefix(prefix)
        _, _, probabilities = self._forward(contexts)
        return {
            item: float(probabilities[0, index])
            for index, item in enumerate(self.target_items)
        }

    def to_dict(self) -> dict:
        return {
            "vocab": self.vocab,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "context_size": self.context_size,
            "seed": self.seed,
            "E": self.E.tolist(),
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "loss_history": self.loss_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SmallBasketLM":
        model = cls(
            vocab=data["vocab"],
            embed_dim=data["embed_dim"],
            hidden_dim=data["hidden_dim"],
            context_size=data["context_size"],
            seed=data["seed"],
        )
        for name in ("E", "W1", "b1", "W2", "b2"):
            weights = np.asarray(data[name], dtype=float)
            if weights.shape != getattr(model, name).shape:
                raise ValueError(f"serialized {name} has an invalid shape")
            setattr(model, name, weights)
        model.loss_history = list(data.get("loss_history", []))
        return model


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
def milestone_3(
    embed_dim: int = 24,
    hidden_dim: int = 48,
    context_size: int = 3,
    epochs: int = 25,
    lr: float = 0.03,
    seed: int = 42,
    store: Store | None = None,
) -> dict:
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

    model = SmallBasketLM(
        vocab,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        context_size=context_size,
        seed=seed,
    ).fit(train_baskets, epochs=epochs, lr=lr)
    store.put("small_lm_model", model.to_dict())

    sample_prefix = ["<BOS>", *planted_itemsets[0][:-1]]
    sample_probabilities = model.predict_proba(sample_prefix)
    sample_item = max(sample_probabilities, key=lambda item: sample_probabilities[item])
    return {
        "initial_loss": model.loss_history[0],
        "final_loss": model.loss_history[-1],
        "loss_history": model.loss_history,
        "epochs": epochs,
        "vocab_size": len(model.vocab),
        "sample_prediction": {
            "prefix": sample_prefix,
            "item": sample_item,
            "probability": sample_probabilities[sample_item],
        },
    }


def evaluate_predictor(
    predictor: NGramBasketPredictor | SmallBasketLM,
    baskets: list[list[str]],
    planted_itemsets: list[list[str]],
) -> dict:
    planted_pairs = {
        pair
        for itemset in planted_itemsets
        for pair in zip(itemset, itemset[1:])
    }
    ranks = {"overall": [], "planted_itemset": [], "general": []}

    for basket in baskets:
        for target_index in range(1, len(basket)):
            target = basket[target_index]
            if target in {"<PAD>", "<BOS>"}:
                continue

            probabilities = predictor.predict_proba(basket[:target_index])
            if target not in probabilities:
                raise ValueError(f"target item {target!r} is not in model vocabulary")
            ranked_items = sorted(
                probabilities, key=lambda item: probabilities[item], reverse=True
            )
            target_rank = ranked_items.index(target) + 1
            ranks["overall"].append(target_rank)
            segment = (
                "planted_itemset"
                if (basket[target_index - 1], target) in planted_pairs
                else "general"
            )
            ranks[segment].append(target_rank)

    if not ranks["overall"]:
        raise ValueError("test baskets contain no next-item transitions")

    def summarize_ranks(segment_ranks: list[int]) -> dict:
        if not segment_ranks:
            return {
                "top1_accuracy": None,
                "top3_accuracy": None,
                "top5_accuracy": None,
                "mrr": None,
                "mean_rank": None,
                "count": 0,
            }
        count = len(segment_ranks)
        return {
            "top1_accuracy": sum(rank == 1 for rank in segment_ranks) / count,
            "top3_accuracy": sum(rank <= 3 for rank in segment_ranks) / count,
            "top5_accuracy": sum(rank <= 5 for rank in segment_ranks) / count,
            "mrr": sum(1 / rank for rank in segment_ranks) / count,
            "mean_rank": sum(segment_ranks) / count,
            "count": count,
        }

    return {
        **summarize_ranks(ranks["overall"]),
        "planted_itemset": summarize_ranks(ranks["planted_itemset"]),
        "general": summarize_ranks(ranks["general"]),
    }


# --- Milestone 4: Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).
# Hint: Implement evaluation metrics similar to those discussed in 'Build a Next-Word Predictor from Scratch' for next-item prediction.
def milestone_4(store: Store | None = None) -> dict:
    if store is None:
        store = Store()

    dataset = store.get("dataset")
    if dataset is None:
        dataset = milestone_1(store=store)
    if not isinstance(dataset, dict):
        raise ValueError("stored dataset must be a dictionary")

    test_baskets = dataset.get("test_baskets")
    planted_itemsets = dataset.get("planted_itemsets")
    if not isinstance(test_baskets, list) or not isinstance(planted_itemsets, list):
        raise ValueError("stored dataset must include test_baskets and planted_itemsets")

    if store.get("ngram_model") is None:
        milestone_2(store=store)
    if store.get("small_lm_model") is None:
        milestone_3(store=store)

    ngram_data = store.get("ngram_model")
    small_lm_data = store.get("small_lm_model")
    if not isinstance(ngram_data, dict) or not isinstance(small_lm_data, dict):
        raise ValueError("stored model data must be dictionaries")

    evaluation = {
        "ngram": evaluate_predictor(
            NGramBasketPredictor.from_dict(ngram_data),
            test_baskets,
            planted_itemsets,
        ),
        "small_lm": evaluate_predictor(
            SmallBasketLM.from_dict(small_lm_data),
            test_baskets,
            planted_itemsets,
        ),
    }
    store.put("evaluation", evaluation)
    return evaluation


# --- Milestone 5: Present a comparison of the models' performance.
# Hint: Summarize your findings and present the comparison clearly, perhaps using visualizations.
def milestone_5(store: Store | None = None) -> dict:
    if store is None:
        store = Store()

    required_keys = ("evaluation", "dataset", "ngram_model", "small_lm_model")
    if any(store.get(key) is None for key in required_keys):
        milestone_4(store=store)

    evaluation = store.get("evaluation")
    dataset = store.get("dataset")
    ngram_data = store.get("ngram_model")
    small_lm_data = store.get("small_lm_model")
    if not isinstance(evaluation, dict) or not isinstance(dataset, dict):
        raise ValueError("stored evaluation and dataset must be dictionaries")
    if not isinstance(ngram_data, dict) or not isinstance(small_lm_data, dict):
        raise ValueError("stored model data must be dictionaries")

    planted_itemsets = dataset.get("planted_itemsets")
    if not isinstance(planted_itemsets, list) or not planted_itemsets:
        raise ValueError("stored dataset must include planted itemsets")

    models = {
        "NGramBasketPredictor": NGramBasketPredictor.from_dict(ngram_data),
        "SmallBasketLM": SmallBasketLM.from_dict(small_lm_data),
    }
    evaluation_keys = {
        "NGramBasketPredictor": "ngram",
        "SmallBasketLM": "small_lm",
    }
    metric_labels = (
        ("Top-1 Accuracy", "top1_accuracy"),
        ("Top-3 Accuracy", "top3_accuracy"),
        ("Top-5 Accuracy", "top5_accuracy"),
        ("MRR", "mrr"),
        ("Mean Rank", "mean_rank"),
    )
    segments = (
        ("Overall", None),
        ("Planted-Itemset", "planted_itemset"),
        ("General", "general"),
    )

    table_rows = []
    for segment_label, segment_key in segments:
        for metric_label, metric_key in metric_labels:
            row = {"Segment": segment_label, "Metric": metric_label}
            for model_label, evaluation_key in evaluation_keys.items():
                metrics = evaluation[evaluation_key]
                if segment_key is not None:
                    metrics = metrics[segment_key]
                value = metrics[metric_key]
                row[model_label] = None if value is None else float(value)
            table_rows.append(row)

    dataframe = pd.DataFrame(table_rows)
    formatted_table = dataframe.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    )
    summary_table = {
        "columns": list(dataframe.columns),
        "rows": dataframe.to_dict(orient="records"),
        "formatted": formatted_table,
    }

    chart_lines = ["Hit-rate and MRR comparison (bar scale: 30 characters)"]
    chart_metrics = (
        ("Hit@1", "top1_accuracy"),
        ("Hit@3", "top3_accuracy"),
        ("Hit@5", "top5_accuracy"),
        ("MRR", "mrr"),
    )
    for metric_label, metric_key in chart_metrics:
        chart_lines.append(f"{metric_label}:")
        for model_label, evaluation_key in evaluation_keys.items():
            value = float(evaluation[evaluation_key][metric_key])
            bar = "#" * round(value * 30)
            chart_lines.append(f"  {model_label:<22} | {bar:<30} {value:.3f}")
    bar_chart = "\n".join(chart_lines)

    spot_checks = []
    for itemset in planted_itemsets[:3]:
        if len(itemset) < 2:
            continue
        prefix = ["<BOS>", *itemset[:-1]]
        recommendations = {}
        for model_label, model in models.items():
            probabilities = model.predict_proba(prefix)
            top_items = sorted(
                probabilities.items(), key=lambda item: item[1], reverse=True
            )[:3]
            recommendations[model_label] = [
                {"item": item, "probability": float(probability)}
                for item, probability in top_items
            ]
        spot_checks.append(
            {
                "prefix": prefix,
                "expected_next_item": itemset[-1],
                "recommendations": recommendations,
            }
        )
    if not spot_checks:
        raise ValueError("stored planted itemsets must contain at least two items")

    def metric_leader(metric_key: str, segment_key: str | None = None) -> str:
        scores = {}
        for model_label, evaluation_key in evaluation_keys.items():
            metrics = evaluation[evaluation_key]
            if segment_key is not None:
                metrics = metrics[segment_key]
            scores[model_label] = metrics[metric_key]
        left, right = scores.values()
        if left is None or right is None:
            return "Insufficient data"
        if abs(left - right) < 1e-12:
            return "Tie"
        if metric_key == "mean_rank":
            return min(scores, key=scores.get)
        return max(scores, key=scores.get)

    overall_leader = metric_leader("mrr")
    planted_leader = metric_leader("top3_accuracy", "planted_itemset")
    general_leader = metric_leader("top3_accuracy", "general")
    recommendations = [
        f"Use {overall_leader} as the default ranker based on overall MRR."
        if overall_leader not in {"Tie", "Insufficient data"}
        else "Compare overall MRR and Hit@k before choosing a default ranker.",
        (
            "Consider segment-aware ranking when planted-bundle and general "
            "transition leaders differ."
            if planted_leader != general_leader
            else "A single ranker can serve both segments when segment leaders agree."
        ),
        "Validate these synthetic-data results against real basket traffic and monitor metrics by segment.",
    ]
    takeaways = {
        "overall_leader": overall_leader,
        "overall_leader_metric": "mrr",
        "planted_itemset_leader": planted_leader,
        "planted_itemset_leader_metric": "top3_accuracy",
        "general_transition_leader": general_leader,
        "general_transition_leader_metric": "top3_accuracy",
        "recommendations": recommendations,
    }

    spot_check_lines = []
    for check in spot_checks:
        spot_check_lines.append(
            f"Prefix {check['prefix']} (expected: {check['expected_next_item']}):"
        )
        for model_label, items in check["recommendations"].items():
            rendered = ", ".join(
                f"{item['item']} ({item['probability']:.3f})" for item in items
            )
            spot_check_lines.append(f"  {model_label}: {rendered}")
    takeaway_lines = [
        f"Overall MRR leader: {overall_leader}.",
        f"Planted-itemset Hit@3 leader: {planted_leader}.",
        f"General-transition Hit@3 leader: {general_leader}.",
        *[f"Recommendation: {item}" for item in recommendations],
    ]
    formatted_report = "\n".join(
        [
            "Retail Basket Model Comparison",
            "==============================",
            "",
            "Side-by-side metrics",
            formatted_table,
            "",
            "Terminal bar visualization",
            bar_chart,
            "",
            "Qualitative spot-checks",
            *spot_check_lines,
            "",
            "Analyst takeaways",
            *takeaway_lines,
        ]
    )
    report = {
        "summary_table": summary_table,
        "bar_chart": bar_chart,
        "spot_checks": spot_checks,
        "takeaways": takeaways,
        "formatted_report": formatted_report,
    }
    store.put("comparison_report", report)
    return report
