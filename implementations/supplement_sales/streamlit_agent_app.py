"""Streamlit UI for the supplement-sales forecasting agent.

Run from the repository root with::

    uv run streamlit run implementations/supplement_sales/streamlit_agent_app.py

The app deliberately keeps the notebook untouched. It reconstructs the same
cutoff-scoped DataService, prompt builder, and AgentPredictor used by
``02_agentic_forecasting.ipynb``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.features import StaticFrameAdapter, canonical_three_col
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import AgentPredictor, ContinuousAgentForecastOutput
from aieng.forecasting.methods.agentic.agent_factory import (
    AgentConfig,
    CodeExecutionConfig,
    ContextRetrievalConfig,
)
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from dotenv import load_dotenv
from pydantic import BaseModel


APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parents[1]
DATA_PATH = APP_DIR / "Supplement_Sales_Weekly_Expanded.csv"
DATE_COL = "Date"
PRODUCT_COL = "Product Name"
TARGET_CANDIDATES = ("Units Sold", "Unit Sold")
NUMERIC_CONTEXT = ("Price", "Revenue", "Discount", "Units Returned")


@st.cache_data(show_spinner=False)
def load_sales(path: str) -> pd.DataFrame:
    """Load and validate the supplement-sales table."""
    sales = pd.read_csv(path, parse_dates=[DATE_COL])
    target = next((column for column in TARGET_CANDIDATES if column in sales.columns), None)
    if target is None:
        raise ValueError(f"The dataset must contain one of {TARGET_CANDIDATES}.")
    required = {DATE_COL, PRODUCT_COL, *NUMERIC_CONTEXT, "Category", target}
    missing = sorted(required - set(sales.columns))
    if missing:
        raise ValueError(f"The dataset is missing required columns: {missing}")
    sales.attrs["target_col"] = target
    return sales.sort_values([PRODUCT_COL, DATE_COL]).reset_index(drop=True)


def product_history(sales: pd.DataFrame, product_name: str) -> pd.DataFrame:
    """Return one product on the complete weekly calendar."""
    target = sales.attrs["target_col"]
    dates = pd.DatetimeIndex(sales[DATE_COL].drop_duplicates().sort_values())
    product = sales.loc[sales[PRODUCT_COL] == product_name].copy()
    product = product.set_index(DATE_COL).sort_index().reindex(dates)
    if product[target].isna().any():
        raise ValueError(f"{product_name} has missing weekly target values.")
    return product


class SupplementSalesPromptBuilder(BaseModel):
    """Serialize the selected product's cutoff-safe context for the agent."""

    sales: Any
    product_name: str
    history_weeks: int = 52
    model_config = {"extra": "forbid"}

    def __call__(self, *, task: ForecastingTask, context: Any) -> str:
        target = context.get_series(task.target_series_id).tail(self.history_weeks)
        product_rows = product_history(self.sales, self.product_name).loc[
            lambda frame: frame.index <= pd.Timestamp(context.as_of)
        ].tail(self.history_weeks)
        target_col = self.sales.attrs["target_col"]
        history = product_rows.reset_index()[[DATE_COL, target_col, *NUMERIC_CONTEXT]]
        latest = product_rows.iloc[-1]
        payload: dict[str, Any] = {
            "task": task.task_id,
            "product_name": self.product_name,
            "category": str(latest["Category"]),
            "as_of": str(context.as_of)[:10],
            "horizons_weeks": list(task.horizons),
            "standard_quantiles": list(STANDARD_QUANTILES),
            "target_summary": {
                "last_units_sold": float(target["value"].iloc[-1]),
                "last_date": str(pd.Timestamp(target["timestamp"].iloc[-1]).date()),
                "mean_units_sold": float(target["value"].mean()),
                "min_units_sold": float(target["value"].min()),
                "max_units_sold": float(target["value"].max()),
            },
            "recent_history_csv": history.to_csv(index=False, float_format="%.4f").strip(),
            "context_note": (
                "Only observations on or before as_of are available. Future price, discount, revenue, "
                "returns, promotions, and other transaction context are unknown. Do not treat historical "
                "values as known future inputs. State important assumptions explicitly."
            ),
        }
        return json.dumps(payload, indent=2)


class ForecastPromptWrapper:
    """Add the structured-output contract to the product context."""

    def __init__(self, sales: pd.DataFrame, product_name: str, history_weeks: int) -> None:
        self.inner = SupplementSalesPromptBuilder(
            sales=sales,
            product_name=product_name,
            history_weeks=history_weeks,
        )
        self.schema = ContinuousAgentForecastOutput.prompt_schema_json()

    def __call__(self, *, task: ForecastingTask, context: Any) -> str:
        payload = json.loads(self.inner(task=task, context=context))
        payload["instructions"] = (
            "Produce exactly one forecast for every horizon in horizons_weeks. Use only information "
            "available on or before as_of. Use exactly the levels in standard_quantiles, make quantiles "
            "non-decreasing, make point_forecast equal to the 0.50 quantile, keep demand forecasts "
            "non-negative, and widen uncertainty appropriately with horizon. Put assumptions and the "
            "main evidence in the rationale fields. Return the result by calling set_model_response "
            "with a json_response string matching output_schema exactly."
        )
        payload["output_schema"] = self.schema
        return json.dumps(payload, indent=2)


def build_agent_config(model: str, enable_search: bool) -> AgentConfig:
    """Build the same agent configuration used by the notebook."""
    return AgentConfig(
        name="supplement_sales_streamlit_agent",
        model=model,
        instruction=(
            "You are a careful supplement-retail demand analyst. Analyze the supplied weekly sales "
            "history for the configured product. Use recent level, trend, volatility, seasonality if "
            "supported by the history, and observed relationships among price, discounts, revenue, "
            "returns, and category context. Do not invent observations, causal effects, promotions, "
            "or future covariates. Forecast non-negative weekly units sold, state assumptions, avoid "
            "false precision, and increase uncertainty for farther horizons."
        ),
        context_retrieval=ContextRetrievalConfig(enabled=enable_search),
        code_execution=CodeExecutionConfig(enabled=False),
    )


def build_forecast_context(
    sales: pd.DataFrame,
    product_name: str,
    cutoff: pd.Timestamp,
    horizon: int,
) -> tuple[ForecastingTask, Any, pd.DataFrame]:
    """Register the product and expose only data available at the cutoff."""
    target = sales.attrs["target_col"]
    history = product_history(sales, product_name)
    series_frame = history[[target]].reset_index().rename(columns={DATE_COL: "timestamp", target: "value"})
    series_frame["released_at"] = series_frame["timestamp"]
    series_frame = canonical_three_col(series_frame)
    series_id = f"supplement_sales_{product_name.lower().replace(' ', '_')}"

    service = DataService()
    service.register(
        series_id,
        StaticFrameAdapter(series_frame),
        SeriesMetadata(
            series_id=series_id,
            description=f"Weekly units sold for {product_name}",
            source=str(DATA_PATH),
            units="units sold",
            frequency="W-MON",
        ),
    )
    task = ForecastingTask(
        task_id=f"{series_id}_streamlit_forecast",
        target_series_id=series_id,
        horizons=list(range(1, horizon + 1)),
        frequency="W-MON",
        description=f"Forecast weekly units sold for {product_name} over the next {horizon} weeks.",
    )
    context = service.context(as_of=cutoff)
    visible = context.get_series(series_id)
    return task, context, visible


def score_rows(predictions: list[Any], actual: pd.Series) -> pd.DataFrame:
    """Combine forecast payloads with realized values where available."""
    rows = []
    for prediction in predictions:
        forecast_date = pd.Timestamp(prediction.forecast_date)
        payload = prediction.payload
        row: dict[str, Any] = {
            "Date": forecast_date,
            "Point forecast": float(payload.point_forecast),
            "P05": float(payload.quantiles[0.05]),
            "P50": float(payload.quantiles[0.50]),
            "P95": float(payload.quantiles[0.95]),
        }
        if forecast_date in actual.index:
            row["Actual"] = float(actual.loc[forecast_date])
        rows.append(row)
    return pd.DataFrame(rows).set_index("Date").sort_index()


def render_forecast_chart(frame: pd.DataFrame, history: pd.DataFrame, target: str) -> None:
    """Render recent history and forecast intervals."""
    figure = go.Figure()
    recent = history.tail(52)
    figure.add_trace(go.Scatter(x=recent.index, y=recent[target], name="Historical sales", mode="lines+markers"))
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["Point forecast"],
            name="Agent forecast",
            mode="lines+markers",
            line={"dash": "dash"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["P95"],
            name="90% interval",
            mode="lines",
            line={"width": 0},
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["P05"],
            name="90% interval",
            mode="lines",
            fill="tonexty",
            line={"width": 0},
            fillcolor="rgba(37, 99, 235, 0.18)",
        )
    )
    figure.update_layout(xaxis_title="Week", yaxis_title="Units sold", hovermode="x unified")
    st.plotly_chart(figure, use_container_width=True)


def main() -> None:  # noqa: PLR0915
    """Run the Streamlit interface."""
    st.set_page_config(page_title="Supplement Sales Agent", page_icon="📈", layout="wide")
    st.title("Supplement Sales Forecast Agent")
    st.caption("Ask the cutoff-aware agent for a probabilistic forecast for one product.")

    load_dotenv(ROOT / ".env", override=False)
    try:
        sales = load_sales(str(DATA_PATH))
    except Exception as error:
        st.error(f"Could not load the sales database: {error}")
        st.stop()

    products = sales[PRODUCT_COL].dropna().unique().tolist()
    dates = pd.DatetimeIndex(sales[DATE_COL].drop_duplicates().sort_values())
    with st.sidebar:
        st.header("Forecast setup")
        product_name = st.selectbox("Product", products, index=products.index("Whey Protein") if "Whey Protein" in products else 0)
        cutoff = st.date_input("Information cutoff", value=pd.Timestamp("2024-12-30").date(), min_value=dates.min().date(), max_value=dates[-2].date())
        cutoff = pd.Timestamp(cutoff)
        future_dates = dates[dates > cutoff]
        max_horizon = min(12, len(future_dates))
        horizon = st.slider("Forecast horizon (weeks)", min_value=1, max_value=max_horizon, value=min(4, max_horizon))
        history_weeks = st.slider("History supplied to agent", min_value=12, max_value=104, value=52, step=4)
        model = st.selectbox("Agent model", [LITE_MODEL, ADVANCED_MODEL], index=0)
        enable_search = st.toggle("Enable web search", value=False, help="Use only when external demand drivers are part of the question.")
        run_forecast = st.button("Generate forecast", type="primary", use_container_width=True)

    history = product_history(sales, product_name)
    target = sales.attrs["target_col"]
    st.subheader(f"{product_name}")
    overview = st.columns(4)
    overview[0].metric("Historical weeks", len(history.loc[:cutoff]))
    overview[1].metric("Last observed sales", f"{history.loc[:cutoff, target].iloc[-1]:,.0f}")
    overview[2].metric("Cutoff", cutoff.strftime("%Y-%m-%d"))
    overview[3].metric("Forecast horizon", f"{horizon} week(s)")

    st.info("The agent sees only product observations on or before the selected cutoff. Future commercial context is treated as unknown.")
    with st.expander("Recent cutoff-scoped data", expanded=False):
        st.dataframe(history.loc[:cutoff].tail(12), use_container_width=True)

    if not run_forecast:
        st.write("Choose the forecast settings and select **Generate forecast** to call the agent.")
        return

    try:
        task, context, visible = build_forecast_context(sales, product_name, cutoff, horizon)
        prompt_builder = ForecastPromptWrapper(sales, product_name, history_weeks)
        predictor = AgentPredictor(
            agent_config=build_agent_config(model, enable_search),
            prompt_builder=prompt_builder,
            output_schema=ContinuousAgentForecastOutput,
        )
        with st.spinner("The forecasting agent is analyzing the historical data..."):
            predictions = predictor.predict(task, context)
    except Exception as error:
        st.error(f"Forecast failed: {error}")
        st.exception(error)
        return

    actual = history.loc[future_dates[:horizon], target].astype(float)
    forecast_frame = score_rows(predictions, actual)
    st.success("Forecast generated.")
    st.dataframe(forecast_frame.style.format("{:.2f}"), use_container_width=True)
    render_forecast_chart(forecast_frame, history, target)

    rationale = predictions[0].metadata.get("rationale") if predictions else None
    if rationale:
        st.subheader("Agent rationale")
        st.write(rationale)
    with st.expander("Technical details"):
        st.write(f"Visible rows supplied through cutoff: {len(visible)}")
        st.write(f"Model: {model}")
        st.write(f"Web search enabled: {enable_search}")
        st.json(json.loads(prompt_builder(task=task, context=context)))

if __name__ == "__main__":
    main()
