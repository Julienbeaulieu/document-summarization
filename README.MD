# Text Summarization App

Goal of this V1: Deploy app that will summarize input text. Model is fine tuned on a news dataset so that it is specialized to summarize news articles.  

## Data

- Use the news summarization dataset to fine tune a T5 model from HugginFace
- Evaluate performance using Rouge metric

## Experiment Management with Weights and Biases + YACS

W & B:
- Keep track of all experiments in one place
- Easily compare runs
- Run multiple experiments
- Run sweeps: automated trials of hyper-parameters

YACS: 
- Configuration of hyperparameters and model hyperparameters

## Linting Script

Running `tasks/lint.sh` fully lints our codebase with a few different checkers:

- `pipenv check` scans our Python package dependency graph for known security vulnerabilities
- `pylint` does static analysis of Python files and reports both style and bug problems
- `pycodestyle` checks for simple code style guideline violations (somewhat overlapping with pylint)
- `mypy` performs static type checking of Python files
- `bandit` performs static analysis to find common security vulnerabilities in Python code
- `shellcheck` finds bugs and potential bugs in shell scrips

## Testing & Continuous Integration with CircleCI

Running `evaluation_tests/evaluate_news_summary.py` evaluates the trained predictors on respective test set, and make sure they are above threshold rouge scores.

Pushing a commit will build the jobs in `.circleci/config.yml`.

## Run model with Flask api

Run `api/app.py` to run a summarization prediction on input text. 