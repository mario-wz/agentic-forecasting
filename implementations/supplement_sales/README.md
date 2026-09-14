# Supplement Sales Forecasting

This use case forecasts weekly units sold for 16 supplement products in
`Supplement_Sales_Weekly_Expanded.csv`. It is organized as a progression from
offline point-forecast baselines to cutoff-safe analyst agents and a separate
search-enabled agent.

## Start Here

Run from the repository root or from this directory after `uv sync`:

1. Run [`01_baseline_models.ipynb`](01_baseline_models.ipynb) to establish the
	conventional reference results.
2. Run [`02_agentic_forecasting.ipynb`](02_agentic_forecasting.ipynb) for one
	product and one historical holdout. Review the prompt before enabling live
	agent calls.
3. Run [`03_agent_analyst_backtest.ipynb`](03_agent_analyst_backtest.ipynb) for
	rolling-origin evaluation without web search. This is the recommended
	development benchmark for the agent strategy.
4. Run [`04_agent_search_forecasting.ipynb`](04_agent_search_forecasting.ipynb)
	only when external demand drivers are part of the question, such as
	regulation, competitor promotions, supply disruption, or broad consumer
	demand.

`RUN_AGENT` defaults should remain off while reviewing prompts. Set it to
`True` only when proxy credentials are available and you intend to make live
model calls.

## Notebook Map

- [`01_baseline_models.ipynb`](01_baseline_models.ipynb) compares Last Value,
  statsmodels ARIMA, Darts AutoARIMA, Prophet, and LightGBM across products on
  one chronological holdout.
- [`02_agentic_forecasting.ipynb`](02_agentic_forecasting.ipynb) builds a
  structured probabilistic forecast for one configured product and compares it
  with the recorded baseline results.
- [`03_agent_analyst_backtest.ipynb`](03_agent_analyst_backtest.ipynb) evaluates
  a no-search analyst agent and offline baselines across rolling origins, with
  interval coverage and horizon diagnostics.
- [`04_agent_search_forecasting.ipynb`](04_agent_search_forecasting.ipynb) runs
  a cutoff-aware search-agent holdout, compares its metrics with the recorded
  baselines, and plots the MAE ranking.

The baseline notebook compares:

- a last-value naive floor,
- explicit statsmodels ARIMA,
- the repository's Darts AutoARIMA implementation,
- Prophet, and
- LightGBM with lagged target, calendar, and sales-context features.

The notebooks resolve the CSV from either the current directory or
`implementations/supplement_sales/`, so both working-directory choices are
supported. They use the project environment created by `uv sync`.

## Configuration

The first configuration cell controls the experiment:

- `TRAIN_END_DATE` sets the chronological train/test cutoff. It must be one of the weekly dates in the CSV.
- `PRODUCTS_TO_RUN` controls the training scope: `None` runs every product; a list such as `["Whey Protein"]` runs only that product.
- `PRODUCT_TO_PLOT` selects the product shown in the individual forecast chart.
- `ARIMA_ORDER`, `LAGS`, `LIGHTGBM_CONTEXT_LAGS`, and the LightGBM context-column tuples control the baselines.

The source column is named `Units Sold`; the loader also accepts `Unit Sold`. Each product is modeled independently, not pooled with the other products. ARIMA, AutoARIMA, Prophet, and Last Value are univariate product-level models. LightGBM is multivariate: it uses target lags, calendar features, lagged `Price`, `Revenue`, `Discount`, and `Units Returned`, plus static `Category` features. Because future context values are not provided, LightGBM carries the last observed numeric context forward during recursive forecasting; `Location` and `Platform` remain excluded.

The notebook reports MAE, RMSE, MAPE, and sMAPE by product and as a mean across products. It renders an aggregate MAE chart, a product-by-model MAE heatmap, and an interactive actual-versus-forecast chart for the selected product. LightGBM uses a date-relative trend feature during both training and recursive prediction.

This remains a single chronological holdout and point-forecast comparison. The repository's formal continuous-forecast metric is CRPS through the `ForecastingTask`/`backtest()` harness; use a rolling-origin experiment with that harness before making a production model choice.

## Shared Configuration

The agent notebooks use these controls:

- `PRODUCT_NAME` selects the independently modeled product.
- `TRAIN_END_DATE` is the information cutoff and must be one of the weekly
	dates in the CSV.
- `HORIZON_WEEKS` controls the forecast length.
- `AGENT_MODEL` uses the project `LITE_MODEL` by default; the advanced model is
	available for deliberate quality runs.
- `RUN_AGENT` controls live calls and should be reviewed before changing.

Every agent prompt is built from a cutoff-scoped `DataService` context. Future
sales observations are excluded structurally, and future price, discount,
revenue, and returns are treated as unknown rather than supplied as covariates.

## Agentic Forecasting

The agent notebooks register only the selected product with a cutoff-safe
`DataService`, preview the exact structured prompt, and request a
`ContinuousAgentForecastOutput` when `RUN_AGENT = True`. The output contains
one point forecast and the standard quantile grid for each horizon.

After the forecast cell, it scores the selected product on the same four-week
chronological holdout used by the baseline notebook with MAE, RMSE, MAPE, and
sMAPE. It also computes a quantile-based CRPS approximation for the agent,
shows the recorded baseline results for the same product and cutoff, and plots
the baseline MAE ranking plus the agent interval against actual sales. The
baseline results are displayed for comparison rather than refit in this
notebook; refresh them in `01_baseline_models.ipynb` when changing the product
or cutoff.

These remain single-holdout diagnostics. They do not establish calibration or
a stable model ranking; use a rolling-origin `DataService` backtest before
making a production model choice.

The rolling-origin analyst notebook is the recommended development benchmark
before adding web search. It keeps search disabled, enriches the prompt with
derived demand statistics, compares Last Value and seasonal-naive baselines,
and plots error, interval coverage, and interval width by horizon. When
`RUN_AGENT = True`, it displays each analyst forecast's rationale and Langfuse
trace URL and writes them to
`supplement_sales_analyst_audit.json`. When it is `False`, the notebook reports
that no analyst calls were made.

The search-enabled notebook is a separate A/B arm. It keeps the same
cutoff-scoped sales context and structured forecast contract, enables the
bounded `search_web` context-retrieval tool, and uses a distinct agent name so
search results cannot overwrite no-search artifacts. The tool searches through
the Vector proxy's `googleSearch` extension, appends grounded source URLs, and
verifies retrospective results against the forecast cutoff. Review the prompt
before enabling live calls. After a run, the notebook displays each horizon's
rationale and trace URL and writes them to
`supplement_sales_search_audit.json`.

## Outputs and Interpretation

The notebooks create small, inspectable artifacts beside the notebooks:

- `supplement_sales_prompt_preview.txt` contains the structured prompt for the
	single-product agent run.
- `supplement_sales_analyst_prompt_preview.txt` contains the rolling-origin
	analyst prompt preview.
- `supplement_sales_search_prompt_preview.txt` contains the search-agent prompt
	preview.
- `supplement_sales_analyst_audit.json` records analyst origin, horizon,
	rationale, and Langfuse trace metadata.
- `supplement_sales_search_audit.json` records search-agent horizon, rationale,
	and Langfuse trace metadata.

Open the Langfuse trace URLs from the audit files to inspect the actual agent
turn. For the search arm, inspect `search_web.google_search` for the grounded
summary and source URLs, and `search_web.leakage_verifier` for cutoff checks and
retries. A rationale or score alone does not prove that search was used; a
`search_web` span in the trace does.

The baseline and search-agent holdout comparisons are diagnostic, not a final
model-selection claim. The recorded baseline values are refreshed by running
Notebook 1, while the agent values depend on the selected product, cutoff,
model, and live response. Use the rolling-origin harness before making a
production choice.
