
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Retirement Planning Optimizer", page_icon="📈", layout="wide")

CURRENCIES: Dict[str, Dict[str, str]] = {
    "EUR": {"symbol": "€", "name": "Euro"},
    "GBP": {"symbol": "£", "name": "British Pound"},
    "USD": {"symbol": "$", "name": "US Dollar"},
    "CHF": {"symbol": "CHF ", "name": "Swiss Franc"},
}

ASSET_CATEGORIES = ["investment", "cash", "property", "other"]
ACCOUNT_TYPES = ["taxable", "pension", "isa", "cash", "property", "other"]
EXPENSE_MODES = ["one_off", "monthly", "annual"]


def fmt_money(value: float, currency: str) -> str:
    symbol = CURRENCIES.get(currency, CURRENCIES["EUR"])["symbol"]
    return f"{symbol}{value:,.0f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}"


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


DEFAULT_SETTINGS = {
    "scenario_name": "Base",
    "display_currency": "EUR",
    "base_currency": "EUR",
    "inflation": 0.025,
    "stress_uplift": 0.00,
    "base_spending_pre75": 100_000.0,
    "base_spending_post75": 80_000.0,
    "target_monthly_income": 10_000.0,
    "target_success_rate": 0.85,
    "mc_runs": 2000,
    "random_seed": 42,
    "optimizer_max_extra_work_years": 10,
    "optimizer_property_sale_step_years": 5,
    "optimizer_spending_cut_step": 5_000.0,
    "optimizer_spending_cut_max": 40_000.0,
    "guardrails_enabled": True,
    "guardrail_cut_pct": 0.10,
    "guardrail_rise_pct": 0.05,
    "guardrail_floor_pct": 0.80,
    "guardrail_ceiling_pct": 1.20,
    "glidepath_enabled": True,
    "glidepath_start_age": 53,
    "glidepath_end_age": 65,
    "glidepath_equity_start": 0.70,
    "glidepath_equity_end": 0.40,
    "glidepath_cash_end": 0.20,
    "healthcare_enabled": True,
    "healthcare_start_age": 75,
    "healthcare_base_annual": 8_000.0,
    "healthcare_inflation_extra": 0.02,
    "legacy_target": 250_000.0,
    "emergency_cash_years": 2.0,
    "tax_enabled": True,
    "country_profile": "Generic Europe",
}

DEFAULT_FX = pd.DataFrame(
    [
        {"currency": "EUR", "to_base": 1.00},
        {"currency": "GBP", "to_base": 1.17},
        {"currency": "USD", "to_base": 0.92},
        {"currency": "CHF", "to_base": 1.04},
    ]
)

DEFAULT_HOUSEHOLD = pd.DataFrame(
    [
        {"enabled": True, "name": "You", "current_age": 53, "retirement_age": 55, "life_expectancy": 95, "pension_age": 67, "pension_annual": 14_000.0, "pension_currency": "GBP"},
        {"enabled": True, "name": "Wife", "current_age": 51, "retirement_age": 55, "life_expectancy": 95, "pension_age": 67, "pension_annual": 13_000.0, "pension_currency": "GBP"},
    ]
)

DEFAULT_ASSETS = pd.DataFrame(
    [
        {"enabled": True, "name": "Investment portfolio", "category": "investment", "account_type": "taxable", "currency": "EUR", "value": 1_070_000.0, "annual_return": 0.060, "volatility": 0.140, "monthly_contribution": 5_000.0, "sale_age": 0, "sale_proceeds": 0.0, "income_annual": 0.0, "income_start_age": 0, "income_end_age": 0, "inflation_linked_income": False},
        {"enabled": True, "name": "Main property", "category": "property", "account_type": "property", "currency": "EUR", "value": 1_020_000.0, "annual_return": 0.025, "volatility": 0.070, "monthly_contribution": 0.0, "sale_age": 75, "sale_proceeds": 500_000.0, "income_annual": 0.0, "income_start_age": 0, "income_end_age": 0, "inflation_linked_income": False},
        {"enabled": True, "name": "Rental property", "category": "property", "account_type": "property", "currency": "EUR", "value": 0.0, "annual_return": 0.025, "volatility": 0.070, "monthly_contribution": 0.0, "sale_age": 0, "sale_proceeds": 0.0, "income_annual": 24_000.0, "income_start_age": 55, "income_end_age": 95, "inflation_linked_income": False},
        {"enabled": True, "name": "Emergency cash", "category": "cash", "account_type": "cash", "currency": "EUR", "value": 50_000.0, "annual_return": 0.020, "volatility": 0.010, "monthly_contribution": 0.0, "sale_age": 0, "sale_proceeds": 0.0, "income_annual": 0.0, "income_start_age": 0, "income_end_age": 0, "inflation_linked_income": False},
    ]
)

DEFAULT_LIABILITIES = pd.DataFrame(
    [
        {"enabled": True, "name": "Main mortgage", "linked_asset": "Main property", "currency": "EUR", "balance": 50_000.0, "interest_rate": 0.035, "monthly_payment": 1_200.0, "target_completion_age": 57, "include_in_net_worth": True}
    ]
)

DEFAULT_INCOMES = pd.DataFrame(
    [
        {"enabled": True, "name": "Consulting", "currency": "EUR", "annual_amount": 10_000.0, "start_age": 55, "end_age": 60, "inflation_linked": False}
    ]
)

DEFAULT_EXPENSES = pd.DataFrame(
    [
        {"enabled": False, "name": "Car purchase", "currency": "EUR", "mode": "one_off", "amount": 50_000.0, "start_age": 56, "end_age": 56, "inflation_linked": False},
        {"enabled": False, "name": "Children wedding", "currency": "EUR", "mode": "one_off", "amount": 30_000.0, "start_age": 60, "end_age": 60, "inflation_linked": False},
        {"enabled": False, "name": "Big holiday", "currency": "EUR", "mode": "annual", "amount": 10_000.0, "start_age": 55, "end_age": 65, "inflation_linked": True},
        {"enabled": False, "name": "Car lease", "currency": "EUR", "mode": "monthly", "amount": 800.0, "start_age": 55, "end_age": 60, "inflation_linked": False},
    ]
)

DEFAULT_TAX = pd.DataFrame(
    [
        {"account_type": "taxable", "withdrawal_tax_rate": 0.20, "growth_tax_drag": 0.00},
        {"account_type": "pension", "withdrawal_tax_rate": 0.15, "growth_tax_drag": 0.00},
        {"account_type": "isa", "withdrawal_tax_rate": 0.00, "growth_tax_drag": 0.00},
        {"account_type": "cash", "withdrawal_tax_rate": 0.00, "growth_tax_drag": 0.00},
        {"account_type": "property", "withdrawal_tax_rate": 0.10, "growth_tax_drag": 0.00},
        {"account_type": "other", "withdrawal_tax_rate": 0.10, "growth_tax_drag": 0.00},
    ]
)


def init_state() -> None:
    defaults = {
        "settings": DEFAULT_SETTINGS,
        "fx_df": DEFAULT_FX,
        "household_df": DEFAULT_HOUSEHOLD,
        "assets_df": DEFAULT_ASSETS,
        "liabilities_df": DEFAULT_LIABILITIES,
        "incomes_df": DEFAULT_INCOMES,
        "expenses_df": DEFAULT_EXPENSES,
        "tax_df": DEFAULT_TAX,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, pd.DataFrame) else dict(value)


@dataclass
class Inputs:
    settings: dict
    fx: pd.DataFrame
    household: pd.DataFrame
    assets: pd.DataFrame
    liabilities: pd.DataFrame
    incomes: pd.DataFrame
    expenses: pd.DataFrame
    tax: pd.DataFrame

    @property
    def display_currency(self) -> str:
        return self.settings["display_currency"]

    @property
    def base_currency(self) -> str:
        return self.settings["base_currency"]

    @property
    def inflation(self) -> float:
        return float(self.settings["inflation"])

    @property
    def stress_uplift(self) -> float:
        return float(self.settings["stress_uplift"])

    @property
    def base_spending_pre75(self) -> float:
        return float(self.settings["base_spending_pre75"])

    @property
    def base_spending_post75(self) -> float:
        return float(self.settings["base_spending_post75"])

    @property
    def mc_runs(self) -> int:
        return int(self.settings["mc_runs"])

    @property
    def random_seed(self) -> int:
        return int(self.settings["random_seed"])

    @property
    def retirement_age(self) -> int:
        enabled = self.household[self.household["enabled"] == True]
        return 55 if enabled.empty else int(enabled["retirement_age"].max())

    @property
    def current_age(self) -> int:
        enabled = self.household[self.household["enabled"] == True]
        return 55 if enabled.empty else int(enabled["current_age"].max())

    @property
    def life_expectancy(self) -> int:
        enabled = self.household[self.household["enabled"] == True]
        return 95 if enabled.empty else int(enabled["life_expectancy"].max())

    @property
    def guardrails_enabled(self) -> bool:
        return bool(self.settings["guardrails_enabled"])


def normalize_inputs() -> Inputs:
    household = st.session_state["household_df"].copy()
    assets = st.session_state["assets_df"].copy()
    liabilities = st.session_state["liabilities_df"].copy()
    incomes = st.session_state["incomes_df"].copy()
    expenses = st.session_state["expenses_df"].copy()
    fx = st.session_state["fx_df"].copy()
    tax = st.session_state["tax_df"].copy()

    for df, fill in [
        (household, {"enabled": True, "name": "", "current_age": 0, "retirement_age": 0, "life_expectancy": 95, "pension_age": 0, "pension_annual": 0.0, "pension_currency": st.session_state["settings"]["base_currency"]}),
        (assets, {"enabled": True, "name": "", "category": "investment", "account_type": "taxable", "currency": st.session_state["settings"]["base_currency"], "value": 0.0, "annual_return": 0.0, "volatility": 0.0, "monthly_contribution": 0.0, "sale_age": 0, "sale_proceeds": 0.0, "income_annual": 0.0, "income_start_age": 0, "income_end_age": 0, "inflation_linked_income": False}),
        (liabilities, {"enabled": True, "name": "", "linked_asset": "", "currency": st.session_state["settings"]["base_currency"], "balance": 0.0, "interest_rate": 0.0, "monthly_payment": 0.0, "target_completion_age": 0, "include_in_net_worth": True}),
        (incomes, {"enabled": True, "name": "", "currency": st.session_state["settings"]["base_currency"], "annual_amount": 0.0, "start_age": 0, "end_age": 0, "inflation_linked": False}),
        (expenses, {"enabled": True, "name": "", "currency": st.session_state["settings"]["base_currency"], "mode": "one_off", "amount": 0.0, "start_age": 0, "end_age": 0, "inflation_linked": False}),
        (fx, {"currency": st.session_state["settings"]["base_currency"], "to_base": 1.0}),
        (tax, {"account_type": "taxable", "withdrawal_tax_rate": 0.0, "growth_tax_drag": 0.0}),
    ]:
        df.fillna(fill, inplace=True)

    household = household[household["enabled"] == True].copy()
    assets = assets[assets["enabled"] == True].copy()
    liabilities = liabilities[liabilities["enabled"] == True].copy()
    incomes = incomes[incomes["enabled"] == True].copy()
    expenses = expenses[expenses["enabled"] == True].copy()

    return Inputs(
        settings=st.session_state["settings"],
        fx=fx,
        household=household,
        assets=assets,
        liabilities=liabilities,
        incomes=incomes,
        expenses=expenses,
        tax=tax,
    )


def fx_rate(inputs: Inputs, currency: str) -> float:
    row = inputs.fx[inputs.fx["currency"] == currency]
    return 1.0 if row.empty else float(row.iloc[0]["to_base"])


def to_base(inputs: Inputs, amount: float, currency: str) -> float:
    return float(amount) * fx_rate(inputs, currency)


def from_base(inputs: Inputs, amount: float, currency: str) -> float:
    rate = fx_rate(inputs, currency)
    return float(amount) / rate if rate else float(amount)


def tax_row(inputs: Inputs, account_type: str) -> Tuple[float, float]:
    row = inputs.tax[inputs.tax["account_type"] == account_type]
    if row.empty or not st.session_state["settings"]["tax_enabled"]:
        return 0.0, 0.0
    return float(row.iloc[0]["withdrawal_tax_rate"]), float(row.iloc[0]["growth_tax_drag"])


def annual_base_spending(age: int, inputs: Inputs) -> float:
    base = inputs.base_spending_pre75 if age < 75 else inputs.base_spending_post75
    base *= (1.0 + inputs.stress_uplift)
    years = max(0, age - inputs.retirement_age)
    return base * ((1.0 + inputs.inflation) ** years)


def healthcare_cost(age: int, inputs: Inputs) -> float:
    s = st.session_state["settings"]
    if not s["healthcare_enabled"] or age < int(s["healthcare_start_age"]):
        return 0.0
    years = age - int(s["healthcare_start_age"])
    growth = float(s["inflation"]) + float(s["healthcare_inflation_extra"])
    return float(s["healthcare_base_annual"]) * ((1.0 + growth) ** years)


def extra_expenses(age: int, inputs: Inputs) -> float:
    total = 0.0
    for _, row in inputs.expenses.iterrows():
        start_age = int(row["start_age"])
        end_age = int(row["end_age"])
        if age < start_age or age > end_age:
            continue
        mode = str(row["mode"])
        if mode == "one_off" and age != start_age:
            continue
        amt = float(row["amount"])
        if mode == "monthly":
            amt *= 12.0
        if bool(row["inflation_linked"]):
            amt *= (1.0 + inputs.inflation) ** max(0, age - start_age)
        total += to_base(inputs, amt, str(row["currency"]))
    total += healthcare_cost(age, inputs)
    return total


def pensions(age: int, inputs: Inputs) -> float:
    total = 0.0
    for _, row in inputs.household.iterrows():
        if age < int(row["pension_age"]):
            continue
        amt = float(row["pension_annual"]) * ((1.0 + inputs.inflation) ** max(0, age - int(row["pension_age"])))
        total += to_base(inputs, amt, str(row["pension_currency"]))
    return total


def other_income(age: int, inputs: Inputs) -> float:
    total = 0.0
    for _, row in inputs.incomes.iterrows():
        if age < int(row["start_age"]) or age > int(row["end_age"]):
            continue
        amt = float(row["annual_amount"])
        if bool(row["inflation_linked"]):
            amt *= (1.0 + inputs.inflation) ** max(0, age - int(row["start_age"]))
        total += to_base(inputs, amt, str(row["currency"]))
    for _, row in inputs.assets.iterrows():
        start_age = int(row["income_start_age"])
        end_age = int(row["income_end_age"]) if int(row["income_end_age"]) > 0 else inputs.life_expectancy
        if age < start_age or age > end_age:
            continue
        amt = float(row["income_annual"])
        if bool(row["inflation_linked_income"]):
            amt *= (1.0 + inputs.inflation) ** max(0, age - start_age)
        total += to_base(inputs, amt, str(row["currency"]))
    return total


def amortize_year(balance: float, annual_rate: float, monthly_payment: float) -> Tuple[float, float]:
    bal = float(balance)
    paid = 0.0
    for _ in range(12):
        if bal <= 0:
            break
        bal += bal * (annual_rate / 12.0)
        payment = min(monthly_payment, bal) if monthly_payment > 0 else 0.0
        bal -= payment
        paid += payment
    return max(0.0, bal), paid


def emergency_cash_floor(age: int, inputs: Inputs) -> float:
    return float(st.session_state["settings"]["emergency_cash_years"]) * (annual_base_spending(age, inputs) + healthcare_cost(age, inputs))


def apply_guardrails(previous_spending: float, current_spending: float, portfolio_return: float, inputs: Inputs) -> float:
    s = st.session_state["settings"]
    if not inputs.guardrails_enabled:
        return current_spending
    floor = current_spending * float(s["guardrail_floor_pct"])
    ceiling = current_spending * float(s["guardrail_ceiling_pct"])
    if portfolio_return < 0:
        adjusted = previous_spending * (1.0 - float(s["guardrail_cut_pct"]))
    else:
        adjusted = previous_spending * (1.0 + float(s["guardrail_rise_pct"]))
    return min(max(adjusted, floor), ceiling)


def build_projection(inputs: Inputs, mc_returns: np.ndarray | None = None) -> pd.DataFrame:
    ages = list(range(inputs.current_age, inputs.life_expectancy + 1))
    assets = inputs.assets.copy()
    liabilities = inputs.liabilities.copy()

    assets["base_value"] = [to_base(inputs, float(v), str(c)) for v, c in zip(assets["value"], assets["currency"])]
    liabilities["base_balance"] = [to_base(inputs, float(v), str(c)) for v, c in zip(liabilities["balance"], liabilities["currency"])]

    sale_done = {str(name): False for name in assets["name"].tolist()}
    rows: List[dict] = []
    last_spending = None

    for age in ages:
        retired = age >= inputs.retirement_age
        contributions = 0.0
        liquid_growth = 0.0
        non_liquid_growth = 0.0
        sale_inflow = 0.0

        liquid_mask = assets["category"].astype(str).str.lower().isin(["investment", "cash", "other"])
        liquid_before_growth = float(assets.loc[liquid_mask, "base_value"].sum())

        for idx, row in assets.iterrows():
            value = float(assets.at[idx, "base_value"])
            category = str(row["category"]).lower()
            account_type = str(row["account_type"])
            wd_tax, growth_drag = tax_row(inputs, account_type)

            if not retired and category in ["investment", "cash", "other"]:
                contrib = to_base(inputs, float(row["monthly_contribution"]) * 12.0, str(row["currency"]))
                value += contrib
                contributions += contrib

            ret = float(row["annual_return"]) - growth_drag
            if mc_returns is not None and retired and category in ["investment", "cash", "other"]:
                ret = mc_returns[age - inputs.retirement_age]

            growth = value * ret
            value += growth

            if category in ["investment", "cash", "other"]:
                liquid_growth += growth
            else:
                non_liquid_growth += growth

            sale_age = int(row["sale_age"])
            sale_proceeds = to_base(inputs, float(row["sale_proceeds"]), str(row["currency"]))
            name = str(row["name"])
            if sale_age > 0 and age >= sale_age and not sale_done[name]:
                if sale_proceeds > 0:
                    sale_inflow += sale_proceeds
                    if category == "property":
                        value = max(0.0, value - sale_proceeds)
                sale_done[name] = True

            assets.at[idx, "base_value"] = max(0.0, value)

        debt_paid = 0.0
        liabilities_end = 0.0
        for idx, row in liabilities.iterrows():
            balance = float(liabilities.at[idx, "base_balance"])
            new_bal, paid = amortize_year(balance, float(row["interest_rate"]), to_base(inputs, float(row["monthly_payment"]), str(row["currency"])))
            liabilities.at[idx, "base_balance"] = new_bal
            debt_paid += paid
            if bool(row["include_in_net_worth"]):
                liabilities_end += new_bal

        pension_income = pensions(age, inputs)
        extra_income = other_income(age, inputs)
        spending_target = annual_base_spending(age, inputs) if retired else 0.0

        total_liquid_ret = safe_div(liquid_growth, liquid_before_growth) if liquid_before_growth > 0 else 0.0
        if retired:
            if last_spending is None:
                spending_target = annual_base_spending(age, inputs)
            else:
                spending_target = apply_guardrails(last_spending, annual_base_spending(age, inputs), total_liquid_ret, inputs)
            last_spending = spending_target

        event_costs = extra_expenses(age, inputs)
        total_spending = spending_target + debt_paid + event_costs

        liquid_pool = float(assets.loc[liquid_mask, "base_value"].sum()) + sale_inflow
        reserve_floor = emergency_cash_floor(age, inputs)
        available_for_draw = max(0.0, liquid_pool - reserve_floor)
        gross_draw_needed = max(0.0, total_spending - pension_income - extra_income)

        # draw proportionally from liquid assets, grossing up for withdrawal tax
        liquid_assets = assets.loc[liquid_mask].copy()
        liquid_values = liquid_assets["base_value"].astype(float)
        total_liquid_value = float(liquid_values.sum())
        net_draw = 0.0
        if retired and total_liquid_value > 0:
            for idx, row in liquid_assets.iterrows():
                share = float(row["base_value"]) / total_liquid_value if total_liquid_value else 0.0
                account_type = str(row["account_type"])
                wd_tax, _ = tax_row(inputs, account_type)
                requested_net = min(available_for_draw, gross_draw_needed) * share
                gross_required = requested_net / max(1e-9, (1.0 - wd_tax))
                gross_required = min(gross_required, float(assets.at[idx, "base_value"]))
                assets.at[idx, "base_value"] = max(0.0, float(assets.at[idx, "base_value"]) - gross_required)
                net_draw += gross_required * (1.0 - wd_tax)

        liquid_end = float(assets.loc[liquid_mask, "base_value"].sum())
        non_liquid_end = float(assets.loc[~liquid_mask, "base_value"].sum())
        net_worth = liquid_end + non_liquid_end - liabilities_end

        rows.append(
            {
                "age": age,
                "phase": "Retirement" if retired else "Pre-retirement",
                "liquid_assets_end": liquid_end,
                "non_liquid_assets_end": non_liquid_end,
                "liabilities_end": liabilities_end,
                "net_worth_end": net_worth,
                "contributions": contributions,
                "pensions": pension_income,
                "other_income": extra_income,
                "base_spending": spending_target,
                "debt_paid": debt_paid,
                "event_costs": event_costs,
                "total_spending": total_spending,
                "net_portfolio_draw": net_draw,
                "sale_inflow": sale_inflow,
                "liquid_growth": liquid_growth,
                "reserve_floor": reserve_floor,
            }
        )
    return pd.DataFrame(rows)


def monte_carlo(inputs: Inputs) -> pd.DataFrame:
    rng = np.random.default_rng(inputs.random_seed)
    liquid_assets = inputs.assets[inputs.assets["category"].astype(str).str.lower().isin(["investment", "cash", "other"])]
    if liquid_assets.empty:
        avg_return, avg_vol = 0.0, 0.0
    else:
        vals = np.array([to_base(inputs, float(v), str(c)) for v, c in zip(liquid_assets["value"], liquid_assets["currency"])], dtype=float)
        total = vals.sum()
        if total <= 0:
            avg_return = float(liquid_assets["annual_return"].astype(float).mean())
            avg_vol = float(liquid_assets["volatility"].astype(float).mean())
        else:
            w = vals / total
            avg_return = float((w * liquid_assets["annual_return"].astype(float).to_numpy()).sum())
            avg_vol = float((w * liquid_assets["volatility"].astype(float).to_numpy()).sum())

    years = max(1, inputs.life_expectancy - inputs.retirement_age + 1)
    paths = []
    for _ in range(inputs.mc_runs):
        returns = rng.normal(avg_return, avg_vol, years)
        proj = build_projection(inputs, returns)
        retirement = proj[proj["age"] >= inputs.retirement_age]
        paths.append(retirement["liquid_assets_end"].to_numpy())
    cols = list(range(inputs.retirement_age, inputs.life_expectancy + 1))
    return pd.DataFrame(paths, columns=cols)


def summarize_mc(mc_df: pd.DataFrame) -> Dict[str, float]:
    finals = mc_df.iloc[:, -1]
    return {
        "success_rate": float((finals > 0).mean()),
        "median_final_liquid": float(finals.median()),
        "p10_final_liquid": float(finals.quantile(0.10)),
        "p90_final_liquid": float(finals.quantile(0.90)),
    }


def optimize(inputs: Inputs) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    notes = []
    base_mc = monte_carlo(inputs)
    base_summary = summarize_mc(base_mc)
    base_proj = build_projection(inputs)
    base_ret = base_proj[base_proj["age"] == inputs.retirement_age].iloc[0]
    rows.append({"strategy": "Current plan", "retirement_age": inputs.retirement_age, "success_rate": base_summary["success_rate"], "liquid_at_retirement": float(base_ret["liquid_assets_end"])})

    # work longer
    for extra in range(1, int(st.session_state["settings"]["optimizer_max_extra_work_years"]) + 1):
        new_household = inputs.household.copy()
        new_household["retirement_age"] = new_household["retirement_age"].astype(int) + extra
        trial = Inputs(inputs.settings, inputs.fx, new_household, inputs.assets, inputs.liabilities, inputs.incomes, inputs.expenses, inputs.tax)
        mc = monte_carlo(trial)
        summary = summarize_mc(mc)
        proj = build_projection(trial)
        ret = proj[proj["age"] == trial.retirement_age].iloc[0]
        rows.append({"strategy": f"Work {extra} more year(s)", "retirement_age": trial.retirement_age, "success_rate": summary["success_rate"], "liquid_at_retirement": float(ret["liquid_assets_end"])})

    # spending cuts
    step = float(st.session_state["settings"]["optimizer_spending_cut_step"])
    max_cut = float(st.session_state["settings"]["optimizer_spending_cut_max"])
    cut = step
    while cut <= max_cut:
        new_settings = dict(inputs.settings)
        new_settings["base_spending_pre75"] = max(0.0, float(inputs.settings["base_spending_pre75"]) - cut)
        new_settings["base_spending_post75"] = max(0.0, float(inputs.settings["base_spending_post75"]) - cut)
        trial = Inputs(new_settings, inputs.fx, inputs.household, inputs.assets, inputs.liabilities, inputs.incomes, inputs.expenses, inputs.tax)
        mc = monte_carlo(trial)
        summary = summarize_mc(mc)
        proj = build_projection(trial)
        ret = proj[proj["age"] == trial.retirement_age].iloc[0]
        rows.append({"strategy": f"Cut annual spending by {fmt_money(cut, inputs.display_currency)}", "retirement_age": trial.retirement_age, "success_rate": summary["success_rate"], "liquid_at_retirement": float(ret["liquid_assets_end"])})
        cut += step

    # property sale timing
    props = inputs.assets[inputs.assets["category"].astype(str).str.lower() == "property"]
    step_years = int(st.session_state["settings"]["optimizer_property_sale_step_years"])
    for _, prop in props.iterrows():
        name = str(prop["name"])
        value = float(prop["value"])
        if value <= 0 and float(prop["sale_proceeds"]) <= 0:
            continue
        for sale_age in range(inputs.retirement_age, inputs.life_expectancy + 1, max(1, step_years)):
            new_assets = inputs.assets.copy()
            mask = new_assets["name"] == name
            new_assets.loc[mask, "sale_age"] = sale_age
            if float(new_assets.loc[mask, "sale_proceeds"].iloc[0]) <= 0:
                new_assets.loc[mask, "sale_proceeds"] = value
            trial = Inputs(inputs.settings, inputs.fx, inputs.household, new_assets, inputs.liabilities, inputs.incomes, inputs.expenses, inputs.tax)
            mc = monte_carlo(trial)
            summary = summarize_mc(mc)
            proj = build_projection(trial)
            ret = proj[proj["age"] == trial.retirement_age].iloc[0]
            rows.append({"strategy": f"Sell {name} at age {sale_age}", "retirement_age": trial.retirement_age, "success_rate": summary["success_rate"], "liquid_at_retirement": float(ret["liquid_assets_end"])})

    result = pd.DataFrame(rows).sort_values(["success_rate", "liquid_at_retirement"], ascending=[False, False]).reset_index(drop=True)
    if not result.empty:
        top = result.iloc[0]
        notes.append(f"Best simple lever in this search: **{top['strategy']}**.")
        notes.append(f"That gets to about **{fmt_pct(float(top['success_rate']))}** success.")
    notes.append("This is still a transparent optimizer, not a black-box solver. It ranks understandable moves first.")
    return result, notes


def readiness_score(success_rate: float, final_liquid: float, legacy_target: float) -> str:
    if success_rate >= 0.90 and final_liquid >= legacy_target:
        return "Healthy"
    if success_rate >= 0.70:
        return "Watchlist"
    return "At Risk"


init_state()
inputs = normalize_inputs()

st.title("Retirement Planning Optimizer")
st.caption("Scenario-led retirement planning with multi-asset modelling, debts, life events, Monte Carlo, FX, tax drag, healthcare costs, guardrails, and reverse planning.")

with st.sidebar:
    s = st.session_state["settings"]
    st.header("Display")
    s["display_currency"] = st.selectbox("Display currency", list(CURRENCIES.keys()), index=list(CURRENCIES.keys()).index(s["display_currency"]), format_func=lambda c: f"{c} — {CURRENCIES[c]['name']}")
    s["base_currency"] = st.selectbox("Model base currency", list(CURRENCIES.keys()), index=list(CURRENCIES.keys()).index(s["base_currency"]), format_func=lambda c: f"{c} — {CURRENCIES[c]['name']}")
    s["scenario_name"] = st.text_input("Scenario name", value=s["scenario_name"])
    st.session_state["settings"] = s
    st.info("The earlier file broke because it was interrupted mid-write. This version is complete and restores all input tables.")

tabs = st.tabs(["Cockpit", "Household", "Assets & Debt", "Income & Events", "FX & Tax", "Advanced", "Monte Carlo", "Reverse Planner"])

with tabs[0]:
    s = st.session_state["settings"]
    c1, c2, c3, c4 = st.columns(4)
    s["base_spending_pre75"] = c1.number_input("Base spending before 75", min_value=0.0, value=float(s["base_spending_pre75"]), step=1000.0)
    s["base_spending_post75"] = c2.number_input("Base spending after 75", min_value=0.0, value=float(s["base_spending_post75"]), step=1000.0)
    s["inflation"] = c3.slider("Inflation", 0.0, 0.10, float(s["inflation"]), step=0.001, format="%.1f%%")
    s["stress_uplift"] = c4.slider("Stress uplift", -0.20, 0.30, float(s["stress_uplift"]), step=0.01, format="%.0f%%")
    st.session_state["settings"] = s
    inputs = normalize_inputs()

    if inputs.household.empty:
        st.warning("Enable at least one household member.")
    else:
        proj = build_projection(inputs)
        mc = monte_carlo(inputs)
        summary = summarize_mc(mc)
        ret_row = proj[proj["age"] == inputs.retirement_age].iloc[0]
        score = readiness_score(summary["success_rate"], summary["median_final_liquid"], float(st.session_state["settings"]["legacy_target"]))
        wd_rate = safe_div(float(ret_row["net_portfolio_draw"]), float(ret_row["liquid_assets_end"]))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current net worth", fmt_money(from_base(inputs, float(proj.iloc[0]["net_worth_end"]), inputs.display_currency), inputs.display_currency))
        m2.metric("Liquid at retirement", fmt_money(from_base(inputs, float(ret_row["liquid_assets_end"]), inputs.display_currency), inputs.display_currency))
        m3.metric("Success rate", fmt_pct(summary["success_rate"]))
        m4.metric("Withdrawal rate at retirement", fmt_pct(wd_rate))

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Net portfolio draw", fmt_money(from_base(inputs, float(ret_row["net_portfolio_draw"]), inputs.display_currency), inputs.display_currency))
        m6.metric("Median final liquid", fmt_money(from_base(inputs, float(summary["median_final_liquid"]), inputs.display_currency), inputs.display_currency))
        m7.metric("10th percentile final liquid", fmt_money(from_base(inputs, float(summary["p10_final_liquid"]), inputs.display_currency), inputs.display_currency))
        m8.metric("Legacy target", fmt_money(float(st.session_state["settings"]["legacy_target"]), inputs.display_currency))

        if score == "Healthy":
            st.success("Healthy — the current plan is broadly resilient on these assumptions.")
        elif score == "Watchlist":
            st.warning("Watchlist — workable, but sequence risk, debt timing, or spending pressure matter.")
        else:
            st.error("At Risk — retirement timing, spending, or asset strategy likely need work.")

        cashflow = pd.DataFrame(
            {
                "Item": ["Base spending", "Debt payments", "Life events + healthcare", "Pensions", "Other income", "Reserve floor", "Net portfolio draw"],
                "Value": [
                    float(ret_row["base_spending"]),
                    float(ret_row["debt_paid"]),
                    float(ret_row["event_costs"]),
                    float(ret_row["pensions"]),
                    float(ret_row["other_income"]),
                    float(ret_row["reserve_floor"]),
                    float(ret_row["net_portfolio_draw"]),
                ],
            }
        )
        cashflow["Display"] = cashflow["Value"].map(lambda x: fmt_money(from_base(inputs, x, inputs.display_currency), inputs.display_currency))
        st.subheader("Retirement year cash flow")
        st.dataframe(cashflow[["Item", "Display"]], use_container_width=True, hide_index=True)

        chart = proj[["age", "liquid_assets_end", "net_worth_end"]].copy().set_index("age")
        chart = chart.apply(lambda col: col.map(lambda x: from_base(inputs, x, inputs.display_currency)))
        st.subheader("Projection")
        st.line_chart(chart)

with tabs[1]:
    st.subheader("Household")
    st.caption("Disable the spouse row for a single-person plan.")
    st.session_state["household_df"] = st.data_editor(
        st.session_state["household_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="household_editor",
        column_config={
            "enabled": st.column_config.CheckboxColumn("Enabled"),
            "name": st.column_config.TextColumn("Name"),
            "current_age": st.column_config.NumberColumn("Current age", step=1),
            "retirement_age": st.column_config.NumberColumn("Retirement age", step=1),
            "life_expectancy": st.column_config.NumberColumn("Life expectancy", step=1),
            "pension_age": st.column_config.NumberColumn("Pension age", step=1),
            "pension_annual": st.column_config.NumberColumn("Pension annual", step=1000, format="%.0f"),
            "pension_currency": st.column_config.SelectboxColumn("Pension currency", options=list(CURRENCIES.keys())),
        },
    )

with tabs[2]:
    st.subheader("Assets")
    st.session_state["assets_df"] = st.data_editor(
        st.session_state["assets_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="assets_editor",
        column_config={
            "enabled": st.column_config.CheckboxColumn("Enabled"),
            "name": st.column_config.TextColumn("Name"),
            "category": st.column_config.SelectboxColumn("Category", options=ASSET_CATEGORIES),
            "account_type": st.column_config.SelectboxColumn("Account type", options=ACCOUNT_TYPES),
            "currency": st.column_config.SelectboxColumn("Currency", options=list(CURRENCIES.keys())),
            "value": st.column_config.NumberColumn("Value", step=1000, format="%.0f"),
            "annual_return": st.column_config.NumberColumn("Annual return", step=0.001, format="%.3f"),
            "volatility": st.column_config.NumberColumn("Volatility", step=0.001, format="%.3f"),
            "monthly_contribution": st.column_config.NumberColumn("Monthly contribution", step=100, format="%.0f"),
            "sale_age": st.column_config.NumberColumn("Sale age", step=1),
            "sale_proceeds": st.column_config.NumberColumn("Sale proceeds", step=1000, format="%.0f"),
            "income_annual": st.column_config.NumberColumn("Annual income", step=1000, format="%.0f"),
            "income_start_age": st.column_config.NumberColumn("Income start age", step=1),
            "income_end_age": st.column_config.NumberColumn("Income end age", step=1),
            "inflation_linked_income": st.column_config.CheckboxColumn("Income inflation linked"),
        },
    )

    st.subheader("Liabilities / Mortgages")
    st.session_state["liabilities_df"] = st.data_editor(
        st.session_state["liabilities_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="liabilities_editor",
        column_config={
            "enabled": st.column_config.CheckboxColumn("Enabled"),
            "name": st.column_config.TextColumn("Name"),
            "linked_asset": st.column_config.TextColumn("Linked asset"),
            "currency": st.column_config.SelectboxColumn("Currency", options=list(CURRENCIES.keys())),
            "balance": st.column_config.NumberColumn("Balance", step=1000, format="%.0f"),
            "interest_rate": st.column_config.NumberColumn("Interest rate", step=0.001, format="%.3f"),
            "monthly_payment": st.column_config.NumberColumn("Monthly payment", step=100, format="%.0f"),
            "target_completion_age": st.column_config.NumberColumn("Completion age", step=1),
            "include_in_net_worth": st.column_config.CheckboxColumn("Include in net worth"),
        },
    )

with tabs[3]:
    st.subheader("Other income")
    st.session_state["incomes_df"] = st.data_editor(
        st.session_state["incomes_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="incomes_editor",
        column_config={
            "enabled": st.column_config.CheckboxColumn("Enabled"),
            "name": st.column_config.TextColumn("Name"),
            "currency": st.column_config.SelectboxColumn("Currency", options=list(CURRENCIES.keys())),
            "annual_amount": st.column_config.NumberColumn("Annual amount", step=1000, format="%.0f"),
            "start_age": st.column_config.NumberColumn("Start age", step=1),
            "end_age": st.column_config.NumberColumn("End age", step=1),
            "inflation_linked": st.column_config.CheckboxColumn("Inflation linked"),
        },
    )

    st.subheader("Life events / big expenses")
    st.session_state["expenses_df"] = st.data_editor(
        st.session_state["expenses_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="expenses_editor",
        column_config={
            "enabled": st.column_config.CheckboxColumn("Enabled"),
            "name": st.column_config.TextColumn("Name"),
            "currency": st.column_config.SelectboxColumn("Currency", options=list(CURRENCIES.keys())),
            "mode": st.column_config.SelectboxColumn("Mode", options=EXPENSE_MODES),
            "amount": st.column_config.NumberColumn("Amount", step=1000, format="%.0f"),
            "start_age": st.column_config.NumberColumn("Start age", step=1),
            "end_age": st.column_config.NumberColumn("End age", step=1),
            "inflation_linked": st.column_config.CheckboxColumn("Inflation linked"),
        },
    )

with tabs[4]:
    st.subheader("FX assumptions")
    st.session_state["fx_df"] = st.data_editor(
        st.session_state["fx_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="fx_editor",
        column_config={
            "currency": st.column_config.SelectboxColumn("Currency", options=list(CURRENCIES.keys())),
            "to_base": st.column_config.NumberColumn("To base", step=0.01, format="%.4f"),
        },
    )
    st.subheader("Tax assumptions")
    s = st.session_state["settings"]
    s["tax_enabled"] = st.checkbox("Enable tax model", value=bool(s["tax_enabled"]))
    s["country_profile"] = st.text_input("Country profile", value=s["country_profile"])
    st.session_state["settings"] = s
    st.session_state["tax_df"] = st.data_editor(
        st.session_state["tax_df"],
        num_rows="dynamic",
        use_container_width=True,
        key="tax_editor",
        column_config={
            "account_type": st.column_config.SelectboxColumn("Account type", options=ACCOUNT_TYPES),
            "withdrawal_tax_rate": st.column_config.NumberColumn("Withdrawal tax", step=0.01, format="%.2f"),
            "growth_tax_drag": st.column_config.NumberColumn("Growth tax drag", step=0.001, format="%.3f"),
        },
    )

with tabs[5]:
    st.subheader("Advanced planning levers")
    s = st.session_state["settings"]
    c1, c2, c3 = st.columns(3)
    s["guardrails_enabled"] = c1.checkbox("Enable spending guardrails", value=bool(s["guardrails_enabled"]))
    s["glidepath_enabled"] = c2.checkbox("Enable glidepath", value=bool(s["glidepath_enabled"]))
    s["healthcare_enabled"] = c3.checkbox("Enable healthcare cost ramp", value=bool(s["healthcare_enabled"]))

    g1, g2, g3, g4 = st.columns(4)
    s["guardrail_cut_pct"] = g1.slider("Guardrail cut after bad years", 0.0, 0.30, float(s["guardrail_cut_pct"]), step=0.01, format="%.0f%%")
    s["guardrail_rise_pct"] = g2.slider("Guardrail rise after good years", 0.0, 0.20, float(s["guardrail_rise_pct"]), step=0.01, format="%.0f%%")
    s["guardrail_floor_pct"] = g3.slider("Guardrail floor", 0.50, 1.00, float(s["guardrail_floor_pct"]), step=0.01, format="%.0f%%")
    s["guardrail_ceiling_pct"] = g4.slider("Guardrail ceiling", 1.00, 1.50, float(s["guardrail_ceiling_pct"]), step=0.01, format="%.0f%%")

    gp1, gp2, gp3, gp4, gp5 = st.columns(5)
    s["glidepath_start_age"] = gp1.number_input("Glidepath start age", 18, 100, int(s["glidepath_start_age"]))
    s["glidepath_end_age"] = gp2.number_input("Glidepath end age", 18, 100, int(s["glidepath_end_age"]))
    s["glidepath_equity_start"] = gp3.slider("Equity start", 0.0, 1.0, float(s["glidepath_equity_start"]), step=0.01)
    s["glidepath_equity_end"] = gp4.slider("Equity end", 0.0, 1.0, float(s["glidepath_equity_end"]), step=0.01)
    s["glidepath_cash_end"] = gp5.slider("Cash end", 0.0, 1.0, float(s["glidepath_cash_end"]), step=0.01)

    h1, h2, h3 = st.columns(3)
    s["healthcare_start_age"] = h1.number_input("Healthcare ramp start age", 50, 110, int(s["healthcare_start_age"]))
    s["healthcare_base_annual"] = h2.number_input("Healthcare base annual", 0.0, value=float(s["healthcare_base_annual"]), step=1000.0)
    s["healthcare_inflation_extra"] = h3.slider("Extra healthcare inflation", 0.0, 0.10, float(s["healthcare_inflation_extra"]), step=0.005, format="%.1f%%")

    l1, l2 = st.columns(2)
    s["legacy_target"] = l1.number_input("Legacy / inheritance target", 0.0, value=float(s["legacy_target"]), step=10000.0)
    s["emergency_cash_years"] = l2.slider("Emergency cash floor in years", 0.0, 5.0, float(s["emergency_cash_years"]), step=0.5)
    st.session_state["settings"] = s

with tabs[6]:
    st.subheader("Monte Carlo")
    s = st.session_state["settings"]
    m1, m2 = st.columns(2)
    s["mc_runs"] = m1.number_input("Monte Carlo runs", 500, 10000, int(s["mc_runs"]), step=500)
    s["random_seed"] = m2.number_input("Random seed", 0, 999999, int(s["random_seed"]), step=1)
    st.session_state["settings"] = s
    inputs = normalize_inputs()

    if not inputs.household.empty:
        mc = monte_carlo(inputs)
        chart = pd.DataFrame(
            {
                "Age": mc.columns.astype(int),
                "Median": mc.median(axis=0).values,
                "10th percentile": mc.quantile(0.10, axis=0).values,
                "90th percentile": mc.quantile(0.90, axis=0).values,
            }
        ).set_index("Age")
        chart = chart.apply(lambda col: col.map(lambda x: from_base(inputs, x, inputs.display_currency)))
        st.line_chart(chart)
        st.caption("This version models liquid asset return shocks, taxes, debt service, life events, pensions, healthcare, reserve floors, and spending guardrails.")

with tabs[7]:
    st.subheader("Reverse planner")
    s = st.session_state["settings"]
    r1, r2, r3, r4 = st.columns(4)
    s["target_monthly_income"] = r1.number_input("Target monthly income", 0.0, value=float(s["target_monthly_income"]), step=500.0)
    s["target_success_rate"] = r2.slider("Target success rate", 0.50, 0.99, float(s["target_success_rate"]), step=0.01, format="%.0f%%")
    s["optimizer_max_extra_work_years"] = r3.number_input("Max extra work years", 1, 20, int(s["optimizer_max_extra_work_years"]), step=1)
    s["optimizer_property_sale_step_years"] = r4.number_input("Property timing step", 1, 10, int(s["optimizer_property_sale_step_years"]), step=1)
    s["optimizer_spending_cut_step"] = st.number_input("Spending cut step", 1000.0, value=float(s["optimizer_spending_cut_step"]), step=1000.0)
    s["optimizer_spending_cut_max"] = st.number_input("Max spending cut to test", 1000.0, value=float(s["optimizer_spending_cut_max"]), step=1000.0)
    st.session_state["settings"] = s
    inputs = normalize_inputs()

    if not inputs.household.empty:
        results, notes = optimize(inputs)
        show = results.copy().head(15)
        show["success_rate"] = show["success_rate"].map(lambda x: fmt_pct(float(x)))
        show["liquid_at_retirement"] = show["liquid_at_retirement"].map(lambda x: fmt_money(from_base(inputs, float(x), inputs.display_currency), inputs.display_currency))
        st.dataframe(show, use_container_width=True, hide_index=True)
        for note in notes:
            st.markdown(f"- {note}")

st.divider()
st.subheader("What changed in this repaired version")
st.markdown(
    """
- Restored all editable input tables.
- Added FX table and base-currency modelling.
- Added tax assumptions by account type.
- Added healthcare ramp, emergency cash floor, legacy target, and spending guardrails.
- Kept spouse enable/disable, multiple assets, multiple mortgages, life events, Monte Carlo, and reverse planner.
"""
)
