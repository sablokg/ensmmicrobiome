#!/usr/bin/env python
"""
Stacking Classifier CLI

Train a StackingClassifier (RandomForest + XGBoost + DecisionTree base models,
LogisticRegression meta-model) on a CSV dataset and print/save predictions.

Usage:
    python stacking_cli.py --data-path data.csv --target target
    python stacking_cli.py --data-path data.csv --target target --output preds.csv --model-out model.joblib
"""

import sys
import click
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


@click.command()
@click.option(
    "--data-path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the input CSV file.",
)
@click.option(
    "--target",
    "target_col",
    type=str,
    default="target",
    show_default=True,
    help="Name of the target column in the CSV.",
)
@click.option(
    "--test-size",
    type=float,
    default=0.2,
    show_default=True,
    help="Fraction of data to hold out for testing.",
)
@click.option(
    "--random-state",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for reproducibility.",
)
@click.option(
    "--rf-n-estimators",
    type=int,
    default=100,
    show_default=True,
    help="Number of trees for the RandomForest base model.",
)
@click.option(
    "--xgb-n-estimators",
    type=int,
    default=100,
    show_default=True,
    help="Number of boosting rounds for XGBoost base model.",
)
@click.option(
    "--xgb-max-depth",
    type=int,
    default=3,
    show_default=True,
    help="Max tree depth for XGBoost base model.",
)
@click.option(
    "--xgb-learning-rate",
    type=float,
    default=0.1,
    show_default=True,
    help="Learning rate for XGBoost base model.",
)
@click.option(
    "--dt-max-depth",
    type=int,
    default=5,
    show_default=True,
    help="Max depth for the DecisionTree base model.",
)
@click.option(
    "--stack-method",
    type=click.Choice(["predict_proba", "decision_function", "predict", "auto"]),
    default="predict_proba",
    show_default=True,
    help="Method used to generate base-model outputs for the meta-model.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Optional path to save predictions as a CSV.",
)
@click.option(
    "--model-out",
    "model_out_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Optional path to save the trained model (joblib).",
)
@click.option(
    "--report/--no-report",
    default=True,
    show_default=True,
    help="Print accuracy and classification report on the test set.",
)
def main(
    data_path,
    target_col,
    test_size,
    random_state,
    rf_n_estimators,
    xgb_n_estimators,
    xgb_max_depth,
    xgb_learning_rate,
    dt_max_depth,
    stack_method,
    output_path,
    model_out_path,
    report,
):
    """Train a stacking classifier and generate predictions."""

    click.echo(f"Loading data from {data_path} ...")
    df = pd.read_csv(data_path)

    if target_col not in df.columns:
        raise click.ClickException(
            f"Target column '{target_col}' not found in data. "
            f"Available columns: {list(df.columns)}"
        )

    # Features and target
    X = df.drop(columns=target_col)
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Base models
    estimators = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=rf_n_estimators, random_state=random_state
            ),
        ),
        (
            "xgb",
            XGBClassifier(
                n_estimators=xgb_n_estimators,
                max_depth=xgb_max_depth,
                learning_rate=xgb_learning_rate,
                random_state=random_state,
            ),
        ),
        (
            "dt",
            DecisionTreeClassifier(max_depth=dt_max_depth, random_state=random_state),
        ),
    ]

    # Stacking
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        stack_method=stack_method,
    )

    click.echo("Training stacking classifier ...")
    model.fit(X_train, y_train)

    click.echo("Generating predictions ...")
    predictions = model.predict(X_test)

    if report:
        acc = accuracy_score(y_test, predictions)
        click.echo(f"\nAccuracy: {acc:.4f}\n")
        click.echo(classification_report(y_test, predictions))
    else:
        click.echo(predictions)

    if output_path:
        pd.DataFrame(
            {"y_true": y_test.reset_index(drop=True), "y_pred": predictions}
        ).to_csv(output_path, index=False)
        click.echo(f"Predictions saved to {output_path}")

    if model_out_path:
        joblib.dump(model, model_out_path)
        click.echo(f"Model saved to {model_out_path}")


if __name__ == "__main__":
    main()