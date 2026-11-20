# Retail Basket Prediction Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/me2cppzu9t/badge.svg)](https://ask.thili.ai/b/me2cppzu9t)

This project provides a tool for retail analysts to compare the effectiveness of different machine learning models in predicting the next item a customer will purchase. By training both an n-gram model and a small language model on simulated shopping basket data, the tool evaluates and contrasts their next-item prediction accuracy, offering insights into their respective strengths for retail applications.

## What you are building

**Goal.** You will build a system for retail analysts to compare the next-item prediction accuracy of an n-gram model and a small language model, both trained on simulated shopping basket data.

**Done when.** You have trained both an n-gram model and a small language model on simulated retail basket data, evaluated their next-item prediction performance using relevant metrics, and presented a comparison of their accuracy.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Generate a synthetic dataset of shopping baskets with planted itemsets.**
      Hint: Use the data generation techniques from 'Teach a Language Model What Sells Together' to create your dataset.
- [x] **2. Implement the n-gram model to predict next items in a basket.**
      Hint: Apply the n-gram model building technique from 'Build a Next-Word Predictor from Scratch' to your basket sequences.
- [x] **3. Pretrain a small language model (like in the drill) on the basket sequences.**
      Hint: Adapt the language model training process from 'Teach a Language Model What Sells Together' for your basket data.
- [x] **4. Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).**
      Hint: Implement evaluation metrics similar to those discussed in 'Build a Next-Word Predictor from Scratch' for next-item prediction.
- [ ] **5. Present a comparison of the models' performance.**  
      Hint: Summarize your findings and present the comparison clearly, perhaps using visualizations.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Explore additional evaluation metrics beyond accuracy and rank, such as recall@k or precision@k, to provide a more nuanced comparison.
- Analyze the types of errors each model makes (e.g., common misses, over-predictions) to identify their strengths and weaknesses on specific itemsets.
- Investigate the impact of different dataset generation parameters (e.g., number of baskets, itemset sizes) on model performance.
