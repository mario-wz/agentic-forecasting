"""Terminal client for the supplement-sales forecasting agent.

Run from the repository root, for example::

    python implementations/supplement_sales/forecast_cli.py \
        --product "Whey Protein" --horizon 4

This client does not start a web server, bind a port, or require browser access.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from aieng.forecasting.methods.agentic import AgentPredictor, ContinuousAgentForecastOutput
from dotenv import load_dotenv
from streamlit_agent_app import (
    ADVANCED_MODEL,
    DATA_PATH,
    LITE_MODEL,
    ForecastPromptWrapper,
    build_agent_config,
    build_forecast_context,
    load_sales,
    product_history,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse command-line forecast settings."""
    parser = argparse.ArgumentParser(description="Generate a supplement-sales agent forecast.")
    parser.add_argument("--product", default="Whey Protein", help="Product to forecast.")
    parser.add_argument("--cutoff", default="2024-12-30", help="Last date visible to the agent (YYYY-MM-DD).")
    parser.add_argument("--horizon", type=int, default=4, help="Number of future weeks to forecast.")
    parser.add_argument("--history-weeks", type=int, default=52, help="Historical weeks supplied to the agent.")
    parser.add_argument("--model", choices=[LITE_MODEL, ADVANCED_MODEL], default=LITE_MODEL)
    parser.add_argument("--search", action="store_true", help="Enable external search context.")
    return parser.parse_args()


def main() -> None:
    """Load data, call the agent, and print a readable forecast."""
    args = parse_args()
    if args.horizon < 1:
        raise ValueError("--horizon must be at least 1.")
    if args.history_weeks < 1:
        raise ValueError("--history-weeks must be at least 1.")

    load_dotenv(ROOT / ".env", override=False)
    sales = load_sales(str(DATA_PATH))
    cutoff = pd.Timestamp(args.cutoff)
    task, context, visible = build_forecast_context(sales, args.product, cutoff, args.horizon)
    prompt_builder = ForecastPromptWrapper(sales, args.product, args.history_weeks)
    predictor = AgentPredictor(
        agent_config=build_agent_config(args.model, args.search),
        prompt_builder=prompt_builder,
        output_schema=ContinuousAgentForecastOutput,
    )

    print(f"Product: {args.product}")
    print(f"Cutoff: {cutoff.date()} | Horizon: {args.horizon} week(s)")
    print(f"Model: {args.model} | Visible observations: {len(visible)}")
    print("Calling forecasting agent...\n")

    predictions = predictor.predict(task, context)
    history = product_history(sales, args.product)
    target = sales.attrs["target_col"]
    for prediction in predictions:
        forecast = prediction.payload
        forecast_date = pd.Timestamp(prediction.forecast_date)
        actual = history.loc[forecast_date, target] if forecast_date in history.index else None
        actual_text = f" | actual={actual:.0f}" if pd.notna(actual) else ""
        print(
            f"{forecast_date.date()} | point={forecast.point_forecast:.1f} | "
            f"P05={forecast.quantiles[0.05]:.1f} | "
            f"P50={forecast.quantiles[0.50]:.1f} | "
            f"P95={forecast.quantiles[0.95]:.1f}{actual_text}"
        )

    rationale = predictions[0].metadata.get("rationale") if predictions else None
    if rationale:
        print("\nAgent rationale:\n")
        print(rationale)


if __name__ == "__main__":
    main()
