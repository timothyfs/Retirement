
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Retirement Planning Cockpit", page_icon="📈", layout="wide")

# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------
CURRENCIES: Dict[str, Dict[str, str]] = {
    "EUR": {"symbol": "€", "name": "Euro"},
    "GBP": {"symbol": "£", "name": "British Pound"},
    "USD": {"symbol": "$", "name": "US Dollar"},
    "CHF": {"symbol": "CHF ", "name": "Swiss Franc"},
}


def fmt_money(value: float, currency: str) -> str:
    symbol = CURRENCIES[currency]["symbol"]
    return f"{symbol}{value:,.0f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}"


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


# ------------------------------------------------------------
# Default data
# ------------------------------------------------------------
DEFAULT_SETTINGS = {
    "currency": "EUR",
    "scenario_name": "Base",
    "base_spending_pre75": 100_000.0,
    "base_spending_post75": 80_000.0,
    "inflation": 0.025,
    "stress_uplift": 0.00,
    "mc_runs": 3000,
    "random_seed": 42,
    "target_monthly_income": 10_000.0,
    "target_success_rate": 0.85,
    "optimizer_max_extra_work_years": 10,
    "optimizer_property_sale_step_years": 5,
}

DEFAULT_HOUSEHOLD = pd.DataFrame(
    [
        {
            "enabled": True,
            "name": "You",
            "current_age": 53,
            "retirement_age": 55,
            "life_expectancy": 95,
            "pension_age": 67,
            "pension_annual": 14_000.0,
        },
        {
            "enabled": True,
            "name": "Wife",
            "current_age": 51,
            "retirement_age": 55,
            "life_expectancy": 95,
            "pension_age": 67,
            "pension_annual": 13_000.0,
        },
    ]
)

DEFAULT_ASSETS = pd.DataFrame(
    [
        {
            "enabled": True,
            "name": "Investment portfolio",
            "category": "liquid",
            "value": 1_070_000.0,
            "annual_return": 0.060,
            "volatility": 0.140,
            "monthly_contribution": 5_000.0,
            "sale_age": 0,
            "sale_proceeds": 0.0,
            "income_annual": 0.0,
            "income_start_age": 0,
            "income_end_age": 0,
            "inflation_linked_income": False,
        },
        {
            "enabled": True,
            "name": "Main property",
            "category": "property",
            "value": 1_020_000.0,
            "annual_return": 0.025,
            "volatility": 0.070,
            "monthly_contribution": 0.0,
            "sale_age": 75,
            "sale_proceeds": 500_000.0,
            "income_annual": 0.0,
            "income_start_age": 0,
            "income_end_age": 0,
            "inflation_linked_income": False,
        },
        {
            "enabled": True,
            "name": "Rental property",
            "category": "property",
            "value": 0.0,
            "annual_return": 0.025,
            "volatility": 0.070,
            "monthly_contribution": 0.0,
            "sale_age": 0,
            "sale_proceeds": 0.0,
            "income_annual": 24_000.0,
            "income_start_age": 55,
            "income_end_age": 95,
            "inflation_linked_income": False,
        },
    ]
)

DEFAULT_LIABILITIES = pd.DataFrame(
    [
        {
            "enabled": True,
            "name": "Main mortgage",
            "linked_asset": "Main property",
            "balance": 50_000.0,
            "interest_rate": 0.035,
            "monthly_payment": 1_200.0,
            "target_completion_age": 57,
            "include_in_net_worth": True,
        }
    ]
)

DEFAULT_INCOMES = pd.DataFrame(
    [
        {
            "enabled": True,
            "name": "Consulting",
            "annual_amount": 10_000.0,
            "start_age": 55,
            "end_age": 60,
            "inflation_linked": False,
        }
    ]
)

DEFAULT_EXPENSES = pd.DataFrame(
    [
        {
            "enabled": False,
            "name": "Car purchase",
            "mode": "one_off",
            "amount": 50_000.0,
            "start_age": 56,
            "end_age": 56,
            "inflation_linked": False,
        },
        {
            "enabled": False,
            "name": "Children wedding",
            "mode": "one_off",
            "amount": 30_000.0,
            "start_age": 60,
            "end_age": 60,
            "inflation_linked": False,
        },
        {
            "enabled": False,
            "name": "Big holiday",
            "mode": "annual",
            "amount": 10_000.0,
            "start_age": 55,
            "end_age": 65,
            "inflation_linked": True,
        },
        {
            "enabled": False,
            "name": "Car lease",
            "mode": "monthly",
            "amount": 800.0,
            "start_age": 55,
            "end_age": 60,
            "inflation_linked": False,
        },
    ]
)


def init_state() -> None:
    defaults = {
        "settings": DEFAULT_SETTINGS,
        "household_df": DEFAULT_HOUSEHOLD,
        "assets_df": DEFAULT_ASSETS,
        "liabilities_df": DEFAULT_LIABILITIES,
        "incomes_df": DEFAULT_INCOMES,
        "expenses_df": DEFAULT_EXPENSES,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            if isinstance(value, pd.DataFrame):
                st.session_state[key] = value.copy()
            elif isinstance(value, dict):
                st.session_state[key] = dict(value)
            else:
                st.session_state[key] = value


# ------------------------------------------------------------
# Model helpers
# ------------------------------------------------------------
@dataclass
class ModelInputs:
    settings: dict
    household: pd.DataFrame
    assets: pd.DataFrame
    liabilities: pd.DataFrame
    incomes: pd.DataFrame
    expenses: pd.DataFrame

    @property
    def currency(self) -> str:
        return self.settings["currency"]

    @property
    def base_spending_pre75(self) -> float:
        return float(self.settings["base_spending_pre75"])

    @property
    def base_spending_post75(self) -> float:
        return float(self.settings["base_spending_post75"])

    @property
    def inflation(self) -> float:
        return float(self.settings["inflation"])

    @property
    def stress_uplift(self) -> float:
        return float(self.settings["stress_uplift"])

    @property
    def mc_runs(self) -> int:
        return int(self.settings["mc_runs"])

    @property
    def random_seed(self) -> int:
        return int(self.settings["random_seed"])

    @property
    def target_monthly_income(self) -> float:
        return float(self.settings["target_monthly_income"])

    @property
    def target_success_rate(self) -> float:
        return float(self.settings["target_success_rate"])

    @property
    def optimizer_max_extra_work_years(self) -> int:
        return int(self.settings["optimizer_max_extra_work_years"])

    @property
    def optimizer_property_sale_step_years(self) -> int:
        return int(self.settings["optimizer_property_sale_step_years"])

    @property
    def primary_current_age(self) -> int:
        df = self.household[self.household["enabled"] == True]
        if df.empty:
            return 55
        return int(df["current_age"].max())

    @property
    def retirement_age(self) -> int:
        df = self.household[self.household["enabled"] == True]
        if df.empty:
            return 55
        return int(df["retirement_age"].max())

    @property
    def life_expectancy(self) -> int:
        df = self.household[self.household["enabled"] == True]
        if df.empty:
            return 95
        return int(df["life_expectancy"].max())

    @property
    def years_to_retirement(self) -> int:
        return max(0, self.retirement_age - self.primary_current_age)


def normalize_tables(raw_inputs: ModelInputs) -> ModelInputs:
    household = raw_inputs.household.copy().fillna(
        {
            "enabled": True,
            "name": "",
            "current_age": 0,
            "retirement_age": 0,
            "life_expectancy": 95,
            "pension_age": 0,
            "pension_annual": 0.0,
        }
    )
    assets = raw_inputs.assets.copy().fillna(
        {
            "enabled": True,
            "name": "",
            "category": "liquid",
            "value": 0.0,
            "annual_return": 0.0,
            "volatility": 0.0,
            "monthly_contribution": 0.0,
            "sale_age": 0,
            "sale_proceeds": 0.0,
            "income_annual": 0.0,
            "income_start_age": 0,
            "income_end_age": 0,
            "inflation_linked_income": False,
        }
    )
    liabilities = raw_inputs.liabilities.copy().fillna(
        {
            "enabled": True,
            "name": "",
            "linked_asset": "",
            "balance": 0.0,
            "interest_rate": 0.0,
            "monthly_payment": 0.0,
            "target_completion_age": 0,
            "include_in_net_worth": True,
        }
    )
    incomes = raw_inputs.incomes.copy().fillna(
        {
            "enabled": True,
            "name": "",
            "annual_amount": 0.0,
            "start_age": 0,
            "end_age": 0,
            "inflation_linked": False,
        }
    )
    expenses = raw_inputs.expenses.copy().fillna(
        {
            "enabled": True,
            "name": "",
            "mode": "one_off",
            "amount": 0.0,
            "start_age": 0,
            "end_age": 0,
            "inflation_linked": False,
        }
    )

    household = household[household["enabled"] == True].copy()
    assets = assets[assets["enabled"] == True].copy()
    liabilities = liabilities[liabilities["enabled"] == True].copy()
    incomes = incomes[incomes["enabled"] == True].copy()
    expenses = expenses[expenses["enabled"] == True].copy()

    return ModelInputs(
        settings=raw_inputs.settings,
        household=household,
        assets=assets,
        liabilities=liabilities,
        incomes=incomes,
        expenses=expenses,
    )


def income_amount_for_age(base_amount: float, start_age: int, age: int, inflation: float, inflation_linked: bool) -> float:
    if age < start_age:
        return 0.0
    if not inflation_linked:
        return float(base_amount)
    return float(base_amount) * ((1.0 + inflation) ** max(0, age - start_age))


def annual_base_spending(age: int, inputs: ModelInputs) -> float:
    base = inputs.base_spending_pre75 if age < 75 else inputs.base_spending_post75
    base *= (1.0 + inputs.stress_uplift)
    return base * ((1.0 + inputs.inflation) ** max(0, age - inputs.retirement_age))


def annual_extra_expenses(age: int, inputs: ModelInputs) -> float:
    total = 0.0
    for _, row in inputs.expenses.iterrows():
        mode = str(row["mode"])
        start_age = int(row["start_age"])
        end_age = int(row["end_age"])
        amount = float(row["amount"])
        linked = bool(row["inflation_linked"])
        if age < start_age or age > end_age:
            continue
        if mode == "one_off" and age != start_age:
            continue

        if mode == "monthly":
            base = amount * 12.0
        else:
            base = amount

        if linked:
            base *= (1.0 + inputs.inflation) ** max(0, age - start_age)
        total += base
    return total


def annual_pensions(age: int, inputs: ModelInputs) -> float:
    total = 0.0
    for _, row in inputs.household.iterrows():
        pension_age = int(row["pension_age"])
        pension_annual = float(row["pension_annual"])
        total += income_amount_for_age(
            base_amount=pension_annual,
            start_age=pension_age,
            age=age,
            inflation=inputs.inflation,
            inflation_linked=True,
        )
    return total


def annual_other_income(age: int, inputs: ModelInputs) -> float:
    total = 0.0
    for _, row in inputs.incomes.iterrows():
        start_age = int(row["start_age"])
        end_age = int(row["end_age"])
        if age < start_age or age > end_age:
            continue
        total += income_amount_for_age(
            base_amount=float(row["annual_amount"]),
            start_age=start_age,
            age=age,
            inflation=inputs.inflation,
            inflation_linked=bool(row["inflation_linked"]),
        )
    for _, row in inputs.assets.iterrows():
        start_age = int(row["income_start_age"])
        end_age = int(row["income_end_age"]) if int(row["income_end_age"]) > 0 else inputs.life_expectancy
        if age < start_age or age > end_age:
            continue
        total += income_amount_for_age(
            base_amount=float(row["income_annual"]),
            start_age=start_age,
            age=age,
            inflation=inputs.inflation,
            inflation_linked=bool(row["inflation_linked_income"]),
        )
    return total


def amortize_one_year(balance: float, annual_rate: float, monthly_payment: float) -> Tuple[float, float]:
    total_paid = 0.0
    bal = float(balance)
    for _ in range(12):
        if bal <= 0:
            break
        interest = bal * (annual_rate / 12.0)
        bal += interest
        payment = min(monthly_payment, bal) if monthly_payment > 0 else 0.0
        bal -= payment
        total_paid += payment
    return max(0.0, bal), total_paid


def build_yearly_projection(inputs: ModelInputs, mc_random_returns: np.ndarray | None = None) -> pd.DataFrame:
    ages = list(range(inputs.primary_current_age, inputs.life_expectancy + 1))
    years_to_ret = inputs.years_to_retirement

    assets = inputs.assets.copy()
    assets["current_value"] = assets["value"].astype(float)
    liabilities = inputs.liabilities.copy()
    liabilities["current_balance"] = liabilities["balance"].astype(float)
    asset_sale_done = {name: False for name in assets["name"].tolist()}

    projection_rows: List[dict] = []

    liquid_categories = {"liquid", "investment", "cash", "other"}

    for year_idx, age in enumerate(ages):
        is_retired = age >= inputs.retirement_age

        contributions = 0.0
        liquid_growth = 0.0
        property_growth = 0.0
        asset_income = 0.0
        asset_sale_inflow = 0.0

        # Contributions and growth
        for idx, row in assets.iterrows():
            category = str(row["category"]).lower()
            current_value = float(assets.at[idx, "current_value"])

            if not is_retired and category in liquid_categories:
                contrib = float(row["monthly_contribution"]) * 12.0
                current_value += contrib
                contributions += contrib

            annual_return = float(row["annual_return"])
            if mc_random_returns is not None and category in liquid_categories and year_idx >= years_to_ret:
                annual_return = mc_random_returns[year_idx - years_to_ret]

            growth = current_value * annual_return
            current_value += growth

            if category in liquid_categories:
                liquid_growth += growth
            else:
                property_growth += growth

            # Recurring asset income
            income_start = int(row["income_start_age"])
            income_end = int(row["income_end_age"]) if int(row["income_end_age"]) > 0 else inputs.life_expectancy
            if age >= income_start and age <= income_end:
                asset_income += income_amount_for_age(
                    base_amount=float(row["income_annual"]),
                    start_age=income_start,
                    age=age,
                    inflation=inputs.inflation,
                    inflation_linked=bool(row["inflation_linked_income"]),
                )

            # Optional sale
            sale_age = int(row["sale_age"])
            sale_proceeds = float(row["sale_proceeds"])
            name = str(row["name"])
            if sale_age > 0 and age >= sale_age and not asset_sale_done.get(name, False):
                if sale_proceeds > 0:
                    asset_sale_inflow += sale_proceeds
                current_value = max(0.0, current_value - sale_proceeds) if category == "property" else current_value
                asset_sale_done[name] = True

            assets.at[idx, "current_value"] = max(0.0, current_value)

        # Liability amortization
        mortgage_payments = 0.0
        total_liabilities = 0.0
        for idx, row in liabilities.iterrows():
            balance = float(liabilities.at[idx, "current_balance"])
            annual_rate = float(row["interest_rate"])
            monthly_payment = float(row["monthly_payment"])
            completion_age = int(row["target_completion_age"])
            if completion_age > 0 and age > completion_age:
                monthly_payment = 0.0 if balance <= 0 else monthly_payment
            new_balance, paid = amortize_one_year(balance, annual_rate, monthly_payment)
            liabilities.at[idx, "current_balance"] = new_balance
            mortgage_payments += paid
            if bool(row["include_in_net_worth"]):
                total_liabilities += new_balance

        pensions = annual_pensions(age, inputs)
        other_income = annual_other_income(age, inputs)
        total_income = pensions + other_income + asset_income

        base_spending = annual_base_spending(age, inputs) if is_retired else 0.0
        life_events = annual_extra_expenses(age, inputs)
        total_spending = base_spending + mortgage_payments + life_events

        # Fund spending from liquid pool and sale proceeds
        liquid_mask = assets["category"].str.lower().isin(liquid_categories)
        liquid_pool = float(assets.loc[liquid_mask, "current_value"].sum()) + asset_sale_inflow
        net_portfolio_draw = max(0.0, total_spending - total_income) if is_retired else 0.0
        liquid_after_draw = max(0.0, liquid_pool - net_portfolio_draw)

        # Push drawdown back proportionally to liquid assets
        prior_liquid_sum = float(assets.loc[liquid_mask, "current_value"].sum())
        if prior_liquid_sum > 0 and liquid_after_draw >= 0:
            ratio = liquid_after_draw / prior_liquid_sum
            assets.loc[liquid_mask, "current_value"] = assets.loc[liquid_mask, "current_value"] * ratio
        elif liquid_mask.any() and net_portfolio_draw > 0:
            assets.loc[liquid_mask, "current_value"] = 0.0

        liquid_assets_end = float(assets.loc[liquid_mask, "current_value"].sum())
        non_liquid_assets_end = float(assets.loc[~liquid_mask, "current_value"].sum())
        net_worth_end = liquid_assets_end + non_liquid_assets_end - total_liabilities

        projection_rows.append(
            {
                "age": age,
                "phase": "Retirement" if is_retired else "Pre-retirement",
                "liquid_assets_end": liquid_assets_end,
                "non_liquid_assets_end": non_liquid_assets_end,
                "total_assets_end": liquid_assets_end + non_liquid_assets_end,
                "liabilities_end": total_liabilities,
                "net_worth_end": net_worth_end,
                "contributions": contributions,
                "liquid_growth": liquid_growth,
                "non_liquid_growth": property_growth,
                "income_total": total_income,
                "pensions": pensions,
                "other_income": other_income + asset_income,
                "base_spending": base_spending,
                "mortgage_payments": mortgage_payments,
                "life_events": life_events,
                "total_spending": total_spending,
                "net_portfolio_draw": net_portfolio_draw,
                "asset_sale_inflow": asset_sale_inflow,
            }
        )

    return pd.DataFrame(projection_rows)


def monte_carlo(inputs: ModelInputs) -> pd.DataFrame:
    rng = np.random.default_rng(inputs.random_seed)
    retirement_years = max(1, inputs.life_expectancy - inputs.retirement_age + 1)

    liquid_assets = inputs.assets[inputs.assets["category"].str.lower().isin(["liquid", "investment", "cash", "other"])]
    if liquid_assets.empty:
        avg_return = 0.0
        avg_vol = 0.0
    else:
        weights = liquid_assets["value"].clip(lower=0).astype(float)
        total = float(weights.sum())
        if total <= 0:
            avg_return = float(liquid_assets["annual_return"].mean())
            avg_vol = float(liquid_assets["volatility"].mean())
        else:
            weights = weights / total
            avg_return = float((weights * liquid_assets["annual_return"].astype(float)).sum())
            avg_vol = float((weights * liquid_assets["volatility"].astype(float)).sum())

    paths = []
    for _ in range(inputs.mc_runs):
        returns = rng.normal(avg_return, avg_vol, retirement_years)
        proj = build_yearly_projection(inputs, mc_random_returns=returns)
        retirement_proj = proj[proj["age"] >= inputs.retirement_age].copy()
        paths.append(retirement_proj["liquid_assets_end"].to_numpy())

    ages = list(range(inputs.retirement_age, inputs.life_expectancy + 1))
    mc_df = pd.DataFrame(paths, columns=ages)
    return mc_df


def summarize_mc(mc_df: pd.DataFrame) -> Dict[str, float]:
    final_values = mc_df.iloc[:, -1]
    return {
        "success_rate": float((final_values > 0).mean()),
        "median_final_liquid": float(final_values.median()),
        "p10_final_liquid": float(final_values.quantile(0.10)),
        "p90_final_liquid": float(final_values.quantile(0.90)),
    }


def safety_label(success_rate: float) -> str:
    if success_rate >= 0.90:
        return "Healthy"
    if success_rate >= 0.70:
        return "Watchlist"
    return "At Risk"


def optimize_to_target(inputs: ModelInputs) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    notes = []

    # Baseline
    baseline_proj = build_yearly_projection(inputs)
    baseline_mc = monte_carlo(inputs)
    baseline_summary = summarize_mc(baseline_mc)
    baseline_row = baseline_proj[baseline_proj["age"] == inputs.retirement_age].iloc[0]
    baseline_target = inputs.target_monthly_income * 12.0

    rows.append(
        {
            "strategy": "Current plan",
            "retirement_age": inputs.retirement_age,
            "target_income": baseline_target,
            "success_rate": baseline_summary["success_rate"],
            "liquid_at_retirement": baseline_row["liquid_assets_end"],
        }
    )

    # Extra work years search
    found = False
    for extra_years in range(1, inputs.optimizer_max_extra_work_years + 1):
        trial = ModelInputs(
            settings=dict(inputs.settings),
            household=inputs.household.copy(),
            assets=inputs.assets.copy(),
            liabilities=inputs.liabilities.copy(),
            incomes=inputs.incomes.copy(),
            expenses=inputs.expenses.copy(),
        )
        trial.household.loc[:, "retirement_age"] = trial.household["retirement_age"].astype(int) + extra_years
        trial_proj = build_yearly_projection(trial)
        trial_mc = monte_carlo(trial)
        trial_summary = summarize_mc(trial_mc)
        trial_row = trial_proj[trial_proj["age"] == trial.retirement_age].iloc[0]

        rows.append(
            {
                "strategy": f"Work {extra_years} more year(s)",
                "retirement_age": trial.retirement_age,
                "target_income": baseline_target,
                "success_rate": trial_summary["success_rate"],
                "liquid_at_retirement": trial_row["liquid_assets_end"],
            }
        )
        if not found and trial_summary["success_rate"] >= inputs.target_success_rate:
            notes.append(
                f"Working {extra_years} more year(s) reaches about {fmt_pct(trial_summary['success_rate'])} success, "
                f"which clears your target of {fmt_pct(inputs.target_success_rate)}."
            )
            found = True
            break

    # Property sale search
    properties = inputs.assets[inputs.assets["category"].str.lower() == "property"].copy()
    for _, prop in properties.iterrows():
        if float(prop["value"]) <= 0 and float(prop["sale_proceeds"]) <= 0:
            continue
        prop_name = str(prop["name"])
        for sale_age in range(inputs.retirement_age, inputs.life_expectancy + 1, max(1, inputs.optimizer_property_sale_step_years)):
            trial = ModelInputs(
                settings=dict(inputs.settings),
                household=inputs.household.copy(),
                assets=inputs.assets.copy(),
                liabilities=inputs.liabilities.copy(),
                incomes=inputs.incomes.copy(),
                expenses=inputs.expenses.copy(),
            )
            mask = trial.assets["name"] == prop_name
            trial.assets.loc[mask, "sale_age"] = sale_age
            if float(trial.assets.loc[mask, "sale_proceeds"].iloc[0]) <= 0:
                trial.assets.loc[mask, "sale_proceeds"] = float(trial.assets.loc[mask, "value"].iloc[0])

            trial_proj = build_yearly_projection(trial)
            trial_mc = monte_carlo(trial)
            trial_summary = summarize_mc(trial_mc)
            trial_row = trial_proj[trial_proj["age"] == trial.retirement_age].iloc[0]

            rows.append(
                {
                    "strategy": f"Sell {prop_name} at age {sale_age}",
                    "retirement_age": trial.retirement_age,
                    "target_income": baseline_target,
                    "success_rate": trial_summary["success_rate"],
                    "liquid_at_retirement": trial_row["liquid_assets_end"],
                }
            )

    results = pd.DataFrame(rows).sort_values(["success_rate", "liquid_at_retirement"], ascending=[False, False]).reset_index(drop=True)

    if results.iloc[0]["strategy"] != "Current plan":
        top = results.iloc[0]
        notes.append(
            f"Best simple lever in the current search space: **{top['strategy']}**. "
            f"That gets you to about {fmt_pct(float(top['success_rate']))} success."
        )

    notes.append(
        "This reverse planner is intentionally simple. It searches one lever at a time so it stays transparent. "
        "The next step would be a true multi-variable optimizer that combines retirement age, spending, sale timing, and extra income."
    )
    return results, notes


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
init_state()

st.title("Retirement Planning Cockpit")
st.caption(
    "Scenario-led retirement planning with deterministic projections, Monte Carlo analysis, liabilities, life events, and reverse planning."
)

with st.sidebar:
    st.header("Display")
    settings = st.session_state["settings"]
    settings["currency"] = st.selectbox(
        "Currency",
        options=list(CURRENCIES.keys()),
        index=list(CURRENCIES.keys()).index(settings["currency"]),
        format_func=lambda code: f"{code} — {CURRENCIES[code]['name']}",
    )
    settings["scenario_name"] = st.text_input("Scenario name", value=settings["scenario_name"])
    st.session_state["settings"] = settings
    st.info(
        "Outputs use comma separators. Number editors may still show raw numeric entry while typing on some devices."
    )

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Cockpit", "Household", "Assets & Debt", "Income & Events", "Monte Carlo", "Reverse Planner"]
)

with tab1:
    st.subheader("Core assumptions")
    c1, c2, c3, c4 = st.columns(4)
    settings = st.session_state["settings"]
    settings["base_spending_pre75"] = c1.number_input("Base spending before 75", min_value=0.0, value=float(settings["base_spending_pre75"]), step=1_000.0)
    settings["base_spending_post75"] = c2.number_input("Base spending after 75", min_value=0.0, value=float(settings["base_spending_post75"]), step=1_000.0)
    settings["inflation"] = c3.slider("Inflation", 0.0, 0.10, float(settings["inflation"]), step=0.001, format="%.1f%%")
    settings["stress_uplift"] = c4.slider("Stress uplift to spending", -0.20, 0.30, float(settings["stress_uplift"]), step=0.01, format="%.0f%%")
    st.session_state["settings"] = settings

    raw_inputs = ModelInputs(
        settings=st.session_state["settings"],
        household=st.session_state["household_df"],
        assets=st.session_state["assets_df"],
        liabilities=st.session_state["liabilities_df"],
        incomes=st.session_state["incomes_df"],
        expenses=st.session_state["expenses_df"],
    )
    inputs = normalize_tables(raw_inputs)

    if inputs.household.empty:
        st.warning("Enable at least one household member.")
    else:
        projection = build_yearly_projection(inputs)
        mc_df = monte_carlo(inputs)
        mc_summary = summarize_mc(mc_df)
        retirement_row = projection[projection["age"] == inputs.retirement_age].iloc[0]
        success = mc_summary["success_rate"]
        safety = safety_label(success)
        annual_target_income = inputs.target_monthly_income * 12.0
        withdrawal_rate = safe_div(retirement_row["net_portfolio_draw"], retirement_row["liquid_assets_end"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current net worth", fmt_money(float(projection.iloc[0]["net_worth_end"]), inputs.currency))
        m2.metric("Liquid at retirement", fmt_money(float(retirement_row["liquid_assets_end"]), inputs.currency))
        m3.metric("Success rate", fmt_pct(success))
        m4.metric("Retirement withdrawal rate", fmt_pct(withdrawal_rate))

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Net portfolio draw at retirement", fmt_money(float(retirement_row["net_portfolio_draw"]), inputs.currency))
        m6.metric("Median final liquid", fmt_money(float(mc_summary["median_final_liquid"]), inputs.currency))
        m7.metric("10th percentile final liquid", fmt_money(float(mc_summary["p10_final_liquid"]), inputs.currency))
        m8.metric("Target monthly income", fmt_money(float(inputs.target_monthly_income), inputs.currency))

        st.subheader("Retirement safety")
        if safety == "Healthy":
            st.success(f"{safety} — current plan looks resilient on these assumptions.")
        elif safety == "Watchlist":
            st.warning(f"{safety} — the plan may work, but sequence risk and spending pressure matter.")
        else:
            st.error(f"{safety} — retirement age, spending, debt schedule, or asset strategy likely need work.")

        cashflow_df = pd.DataFrame(
            {
                "Item": [
                    "Base spending",
                    "Mortgage / debt payments",
                    "Life events",
                    "Pensions",
                    "Other income",
                    "Net portfolio draw",
                ],
                "Value": [
                    float(retirement_row["base_spending"]),
                    float(retirement_row["mortgage_payments"]),
                    float(retirement_row["life_events"]),
                    float(retirement_row["pensions"]),
                    float(retirement_row["other_income"]),
                    float(retirement_row["net_portfolio_draw"]),
                ],
            }
        )
        cashflow_df["Display"] = cashflow_df["Value"].map(lambda x: fmt_money(x, inputs.currency))
        st.subheader("Retirement year cash flow")
        st.dataframe(cashflow_df[["Item", "Display"]], use_container_width=True, hide_index=True)

        chart_df = projection[["age", "liquid_assets_end", "net_worth_end"]].set_index("age")
        st.subheader("Projection")
        st.line_chart(chart_df)

with tab2:
    st.subheader("Household members")
    st.caption("Disable the spouse row if you want a single-person plan.")
    household_df = st.data_editor(
        st.session_state["household_df"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "enabled": st.column_config.CheckboxColumn("Enabled"),
            "name": st.column_config.TextColumn("Name"),
            "current_age": st.column_config.NumberColumn("Current age", step=1),
            "retirement_age": st.column_config.NumberColumn("Retirement age", step=1),
            "life_expectancy": st.column_config.NumberColumn("Life expectancy", step=1),
            "pension_age": st.column_config.NumberColumn("Pension age", step=1),
            "pension_annual": st.column_config.NumberColumn("Pension annual", step=1000, format="%.0f"),
        },
        key="household_editor",
    )
    st.session_state["household_df"] = household_df

with tab3:
    st.subheader("Assets")
    assets_df = st.data_editor(
        st.session_state["assets_df"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "enabled": st.column_config.CheckboxColumn("Enabled"),
            "name": st.column_config.TextColumn("Name"),
            "category": st.column_config.SelectboxColumn("Category", options=["liquid", "property", "investment", "cash", "other"]),
            "value": st.column_config.NumberColumn("Value", step=1000, format="%.0f"),
            "annual_return": st.column_config.NumberColumn("Annual return", step=0.001, format="%.3f"),
            "volatility": st.column_config.NumberColumn("Volatility", step=0.001, format="%.3f"),
            "monthly_contribution": st.column_config.NumberColumn("Monthly contribution", step=100, format="%.0f"),
            "sale_age": st.column_config.NumberColumn("Sale age (0 off)", step=1),
            "sale_proceeds": st.column_config.NumberColumn("Sale proceeds", step=1000, format="%.0f"),
            "income_annual": st.column_config.NumberColumn("Annual income", step=1000, format="%.0f"),
            "income_start_age": st.column_config.NumberColumn("Income start age", step=1),
            "income_end_age": st.column_config.NumberColumn("Income end age", step=1),
            "inflation_linked_income": st.column_config.CheckboxColumn("Inflation linked income"),
        },
        key="assets_editor",
    )
    st.session_state["assets_df"] = assets_df

    st.subheader("Liabilities / mortgages")
    st.caption("Set a monthly payment and target completion age for each debt.")
    liabilities_df = st.data_editor(
        st.session_state["liabilities_df"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "enabled": st.column_config.CheckboxColumn("Enabled"),
            "name": st.column_config.TextColumn("Name"),
            "linked_asset": st.column_config.TextColumn("Linked asset"),
            "balance": st.column_config.NumberColumn("Balance", step=1000, format="%.0f"),
            "interest_rate": st.column_config.NumberColumn("Interest rate", step=0.001, format="%.3f"),
            "monthly_payment": st.column_config.NumberColumn("Monthly payment", step=100, format="%.0f"),
            "target_completion_age": st.column_config.NumberColumn("Target completion age", step=1),
            "include_in_net_worth": st.column_config.CheckboxColumn("Include in net worth"),
        },
        key="liabilities_editor",
    )
    st.session_state["liabilities_df"] = liabilities_df

with tab4:
    st.subheader("Other income sources")
    incomes_df = st.data_editor(
        st.session_state["incomes_df"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "enabled": st.column_config.CheckboxColumn("Enabled"),
            "name": st.column_config.TextColumn("Name"),
            "annual_amount": st.column_config.NumberColumn("Annual amount", step=1000, format="%.0f"),
            "start_age": st.column_config.NumberColumn("Start age", step=1),
            "end_age": st.column_config.NumberColumn("End age", step=1),
            "inflation_linked": st.column_config.CheckboxColumn("Inflation linked"),
        },
        key="incomes_editor",
    )
    st.session_state["incomes_df"] = incomes_df

    st.subheader("Big expense items / life events")
    st.caption("Use one-off for weddings, education, house works, or car purchases. Use monthly for leases. Use annual for recurring holidays.")
    expenses_df = st.data_editor(
        st.session_state["expenses_df"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "enabled": st.column_config.CheckboxColumn("Enabled"),
            "name": st.column_config.TextColumn("Name"),
            "mode": st.column_config.SelectboxColumn("Mode", options=["one_off", "monthly", "annual"]),
            "amount": st.column_config.NumberColumn("Amount", step=1000, format="%.0f"),
            "start_age": st.column_config.NumberColumn("Start age", step=1),
            "end_age": st.column_config.NumberColumn("End age", step=1),
            "inflation_linked": st.column_config.CheckboxColumn("Inflation linked"),
        },
        key="expenses_editor",
    )
    st.session_state["expenses_df"] = expenses_df

with tab5:
    st.subheader("Monte Carlo settings")
    settings = st.session_state["settings"]
    c1, c2 = st.columns(2)
    settings["mc_runs"] = c1.number_input("Monte Carlo runs", min_value=500, max_value=10000, value=int(settings["mc_runs"]), step=500)
    settings["random_seed"] = c2.number_input("Random seed", min_value=0, max_value=999999, value=int(settings["random_seed"]), step=1)
    st.session_state["settings"] = settings

    raw_inputs = ModelInputs(
        settings=st.session_state["settings"],
        household=st.session_state["household_df"],
        assets=st.session_state["assets_df"],
        liabilities=st.session_state["liabilities_df"],
        incomes=st.session_state["incomes_df"],
        expenses=st.session_state["expenses_df"],
    )
    inputs = normalize_tables(raw_inputs)

    if not inputs.household.empty:
        mc_df = monte_carlo(inputs)
        mc_chart = pd.DataFrame(
            {
                "Age": mc_df.columns.astype(int),
                "Median": mc_df.median(axis=0).values,
                "10th percentile": mc_df.quantile(0.10, axis=0).values,
                "90th percentile": mc_df.quantile(0.90, axis=0).values,
            }
        ).set_index("Age")
        st.line_chart(mc_chart)

        with st.expander("Simulation notes", expanded=False):
            st.markdown(
                """
                - Monte Carlo currently shocks liquid asset returns only.
                - Property assets are modelled deterministically unless sold.
                - Debts, pensions, rental income, and life events flow through every scenario.
                """
            )

with tab6:
    st.subheader("Reverse planner")
    settings = st.session_state["settings"]
    rp1, rp2, rp3 = st.columns(3)
    settings["target_monthly_income"] = rp1.number_input("Target monthly income", min_value=0.0, value=float(settings["target_monthly_income"]), step=500.0)
    settings["target_success_rate"] = rp2.slider("Target success rate", 0.50, 0.99, float(settings["target_success_rate"]), step=0.01, format="%.0f%%")
    settings["optimizer_max_extra_work_years"] = rp3.number_input("Max extra work years to test", min_value=1, max_value=20, value=int(settings["optimizer_max_extra_work_years"]), step=1)
    settings["optimizer_property_sale_step_years"] = st.number_input(
        "Property sale timing step (years)", min_value=1, max_value=10, value=int(settings["optimizer_property_sale_step_years"]), step=1
    )
    st.session_state["settings"] = settings

    raw_inputs = ModelInputs(
        settings=st.session_state["settings"],
        household=st.session_state["household_df"],
        assets=st.session_state["assets_df"],
        liabilities=st.session_state["liabilities_df"],
        incomes=st.session_state["incomes_df"],
        expenses=st.session_state["expenses_df"],
    )
    inputs = normalize_tables(raw_inputs)

    if not inputs.household.empty:
        results, notes = optimize_to_target(inputs)
        display = results.copy()
        display["success_rate"] = display["success_rate"].map(lambda x: fmt_pct(float(x)))
        display["target_income"] = display["target_income"].map(lambda x: fmt_money(float(x), inputs.currency))
        display["liquid_at_retirement"] = display["liquid_at_retirement"].map(lambda x: fmt_money(float(x), inputs.currency))
        st.dataframe(display.head(12), use_container_width=True, hide_index=True)

        st.subheader("What the model sees")
        for note in notes:
            st.markdown(f"- {note}")

        st.info(
            "This reverse planner is a first optimisation layer. A stronger version would test combined actions such as "
            "working longer plus reducing a future car lease plus selling a property at a chosen age."
        )

st.divider()
st.subheader("High value improvements to consider next")
st.markdown(
    """
- Taxes by country and account type, especially if you want GBP, EUR, USD, and CHF handled properly rather than just symbol switching.
- FX modelling, so a GBP pension can fund a EUR lifestyle with exchange-rate stress.
- Glide path investing, where asset allocation changes as retirement gets closer.
- Dynamic spending guardrails, so the plan trims withdrawals slightly after bad market years instead of pretending spending never moves.
- Healthcare and long-term care cost ramps later in life.
- Emergency cash floor, so the app separates truly investable assets from reserve cash.
- Legacy goal or inheritance target, so the model optimises for both lifestyle and what you want left behind.
"""
)
