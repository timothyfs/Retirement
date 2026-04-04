
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Retirement Planner", page_icon="📈", layout="wide")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
CURRENCIES: Dict[str, Dict[str, str]] = {
    "EUR": {"symbol": "€", "name": "Euro"},
    "GBP": {"symbol": "£", "name": "British Pound"},
    "USD": {"symbol": "$", "name": "US Dollar"},
    "CHF": {"symbol": "CHF ", "name": "Swiss Franc"},
}

PAGES = [
    "Welcome",
    "Household",
    "Assets",
    "Debt",
    "Income",
    "Expenses",
    "FX & Tax",
    "Settings",
    "Results",
    "Optimizer",
]

def fmt_money(value: float, currency: str) -> str:
    symbol = CURRENCIES.get(currency, CURRENCIES["EUR"])["symbol"]
    return f"{symbol}{value:,.0f}"

def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}"

def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b

def to_records(df: pd.DataFrame) -> list[dict]:
    return df.to_dict("records")

def from_records(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)

# ------------------------------------------------------------
# Defaults
# ------------------------------------------------------------
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
    "mc_runs": 1500,
    "random_seed": 42,
    "healthcare_enabled": True,
    "healthcare_start_age": 75,
    "healthcare_base_annual": 8_000.0,
    "healthcare_inflation_extra": 0.02,
    "legacy_target": 250_000.0,
    "emergency_cash_years": 2.0,
    "tax_enabled": True,
}

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

DEFAULT_DEBT = pd.DataFrame(
    [
        {"enabled": True, "name": "Main mortgage", "linked_asset": "Main property", "currency": "EUR", "balance": 50_000.0, "interest_rate": 0.035, "monthly_payment": 1_200.0, "include_in_net_worth": True}
    ]
)

DEFAULT_INCOME = pd.DataFrame(
    [
        {"enabled": True, "name": "Consulting", "currency": "EUR", "annual_amount": 10_000.0, "start_age": 55, "end_age": 60, "inflation_linked": False}
    ]
)

DEFAULT_EXPENSES = pd.DataFrame(
    [
        {"enabled": False, "name": "Car purchase", "currency": "EUR", "mode": "one_off", "amount": 50_000.0, "start_age": 56, "end_age": 56, "inflation_linked": False},
        {"enabled": False, "name": "Children wedding", "currency": "EUR", "mode": "one_off", "amount": 30_000.0, "start_age": 60, "end_age": 60, "inflation_linked": False},
    ]
)

DEFAULT_FX = pd.DataFrame(
    [
        {"currency": "EUR", "to_base": 1.00},
        {"currency": "GBP", "to_base": 1.17},
        {"currency": "USD", "to_base": 0.92},
        {"currency": "CHF", "to_base": 1.04},
    ]
)

DEFAULT_TAX = pd.DataFrame(
    [
        {"account_type": "taxable", "withdrawal_tax_rate": 0.20, "growth_tax_drag": 0.00},
        {"account_type": "pension", "withdrawal_tax_rate": 0.15, "growth_tax_drag": 0.00},
        {"account_type": "cash", "withdrawal_tax_rate": 0.00, "growth_tax_drag": 0.00},
        {"account_type": "property", "withdrawal_tax_rate": 0.10, "growth_tax_drag": 0.00},
        {"account_type": "other", "withdrawal_tax_rate": 0.10, "growth_tax_drag": 0.00},
        {"account_type": "isa", "withdrawal_tax_rate": 0.00, "growth_tax_drag": 0.00},
    ]
)

# ------------------------------------------------------------
# State
# ------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "settings": dict(DEFAULT_SETTINGS),
        "household_records": to_records(DEFAULT_HOUSEHOLD),
        "asset_records": to_records(DEFAULT_ASSETS),
        "debt_records": to_records(DEFAULT_DEBT),
        "income_records": to_records(DEFAULT_INCOME),
        "expense_records": to_records(DEFAULT_EXPENSES),
        "fx_records": to_records(DEFAULT_FX),
        "tax_records": to_records(DEFAULT_TAX),
        "page": "Welcome",
        "apply_counter": 0,
        "auto_recalc": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
@dataclass
class Inputs:
    settings: dict
    household: pd.DataFrame
    assets: pd.DataFrame
    debt: pd.DataFrame
    income: pd.DataFrame
    expenses: pd.DataFrame
    fx: pd.DataFrame
    tax: pd.DataFrame

    @property
    def display_currency(self) -> str:
        return self.settings["display_currency"]

    @property
    def base_currency(self) -> str:
        return self.settings["base_currency"]

    @property
    def current_age(self) -> int:
        h = self.household[self.household["enabled"] == True]
        return 55 if h.empty else int(h["current_age"].max())

    @property
    def retirement_age(self) -> int:
        h = self.household[self.household["enabled"] == True]
        return 55 if h.empty else int(h["retirement_age"].max())

    @property
    def life_expectancy(self) -> int:
        h = self.household[self.household["enabled"] == True]
        return 95 if h.empty else int(h["life_expectancy"].max())

    @property
    def mc_runs(self) -> int:
        return int(self.settings["mc_runs"])

    @property
    def random_seed(self) -> int:
        return int(self.settings["random_seed"])


def get_inputs() -> Inputs:
    return Inputs(
        settings=dict(st.session_state["settings"]),
        household=from_records(st.session_state["household_records"]).fillna(False),
        assets=from_records(st.session_state["asset_records"]).fillna(False),
        debt=from_records(st.session_state["debt_records"]).fillna(False),
        income=from_records(st.session_state["income_records"]).fillna(False),
        expenses=from_records(st.session_state["expense_records"]).fillna(False),
        fx=from_records(st.session_state["fx_records"]).fillna(0),
        tax=from_records(st.session_state["tax_records"]).fillna(0),
    )


def fx_rate(inputs: Inputs, currency: str) -> float:
    row = inputs.fx[inputs.fx["currency"] == currency]
    return 1.0 if row.empty else float(row.iloc[0]["to_base"])


def to_base(inputs: Inputs, amount: float, currency: str) -> float:
    return float(amount) * fx_rate(inputs, currency)


def from_base(inputs: Inputs, amount: float, currency: str) -> float:
    rate = fx_rate(inputs, currency)
    return float(amount) / rate if rate else float(amount)


def tax_rates(inputs: Inputs, account_type: str) -> Tuple[float, float]:
    if not inputs.settings["tax_enabled"]:
        return 0.0, 0.0
    row = inputs.tax[inputs.tax["account_type"] == account_type]
    if row.empty:
        return 0.0, 0.0
    return float(row.iloc[0]["withdrawal_tax_rate"]), float(row.iloc[0]["growth_tax_drag"])


def annual_base_spending(age: int, inputs: Inputs) -> float:
    base = inputs.settings["base_spending_pre75"] if age < 75 else inputs.settings["base_spending_post75"]
    base *= (1.0 + inputs.settings["stress_uplift"])
    years = max(0, age - inputs.retirement_age)
    return base * ((1.0 + inputs.settings["inflation"]) ** years)


def annual_healthcare(age: int, inputs: Inputs) -> float:
    if not inputs.settings["healthcare_enabled"] or age < inputs.settings["healthcare_start_age"]:
        return 0.0
    years = age - inputs.settings["healthcare_start_age"]
    rate = inputs.settings["inflation"] + inputs.settings["healthcare_inflation_extra"]
    return inputs.settings["healthcare_base_annual"] * ((1.0 + rate) ** years)


def annual_expenses(age: int, inputs: Inputs) -> float:
    total = annual_healthcare(age, inputs)
    for _, row in inputs.expenses.iterrows():
        if not bool(row.get("enabled", True)):
            continue
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
            amt *= (1.0 + inputs.settings["inflation"]) ** max(0, age - start_age)
        total += to_base(inputs, amt, str(row["currency"]))
    return total


def annual_pensions(age: int, inputs: Inputs) -> float:
    total = 0.0
    for _, row in inputs.household.iterrows():
        if not bool(row.get("enabled", True)):
            continue
        if age < int(row["pension_age"]):
            continue
        amt = float(row["pension_annual"]) * ((1.0 + inputs.settings["inflation"]) ** max(0, age - int(row["pension_age"])))
        total += to_base(inputs, amt, str(row["pension_currency"]))
    return total


def annual_other_income(age: int, inputs: Inputs) -> float:
    total = 0.0
    for _, row in inputs.income.iterrows():
        if not bool(row.get("enabled", True)):
            continue
        if age < int(row["start_age"]) or age > int(row["end_age"]):
            continue
        amt = float(row["annual_amount"])
        if bool(row["inflation_linked"]):
            amt *= (1.0 + inputs.settings["inflation"]) ** max(0, age - int(row["start_age"]))
        total += to_base(inputs, amt, str(row["currency"]))
    for _, row in inputs.assets.iterrows():
        if not bool(row.get("enabled", True)):
            continue
        start_age = int(row["income_start_age"])
        end_age = int(row["income_end_age"]) if int(row["income_end_age"]) > 0 else inputs.life_expectancy
        if age < start_age or age > end_age:
            continue
        amt = float(row["income_annual"])
        if bool(row["inflation_linked_income"]):
            amt *= (1.0 + inputs.settings["inflation"]) ** max(0, age - start_age)
        total += to_base(inputs, amt, str(row["currency"]))
    return total


def amortize_one_year(balance: float, annual_rate: float, monthly_payment: float) -> Tuple[float, float]:
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


def build_projection(inputs: Inputs, retirement_returns: np.ndarray | None = None) -> pd.DataFrame:
    assets = inputs.assets.copy()
    debt = inputs.debt.copy()

    assets = assets[assets["enabled"] == True].copy() if "enabled" in assets.columns else assets.copy()
    debt = debt[debt["enabled"] == True].copy() if "enabled" in debt.columns else debt.copy()

    if assets.empty:
        assets = pd.DataFrame(columns=["category", "currency", "value"])

    assets["base_value"] = [to_base(inputs, float(v), str(c)) for v, c in zip(assets.get("value", []), assets.get("currency", []))]
    debt["base_balance"] = [to_base(inputs, float(v), str(c)) for v, c in zip(debt.get("balance", []), debt.get("currency", []))]

    rows = []
    sale_done = {str(name): False for name in assets.get("name", pd.Series(dtype=str)).tolist()}

    for age in range(inputs.current_age, inputs.life_expectancy + 1):
        retired = age >= inputs.retirement_age
        contributions = 0.0
        liquid_growth = 0.0
        sale_inflow = 0.0
        liquid_mask = assets["category"].astype(str).isin(["investment", "cash", "other"]) if not assets.empty else pd.Series(dtype=bool)

        for idx, row in assets.iterrows():
            value = float(assets.at[idx, "base_value"])
            category = str(row["category"])
            _, growth_tax_drag = tax_rates(inputs, str(row["account_type"]))

            if not retired and category in ["investment", "cash", "other"]:
                contrib = to_base(inputs, float(row["monthly_contribution"]) * 12.0, str(row["currency"]))
                value += contrib
                contributions += contrib

            annual_return = float(row["annual_return"]) - growth_tax_drag
            if retirement_returns is not None and retired and category in ["investment", "cash", "other"]:
                annual_return = retirement_returns[age - inputs.retirement_age]

            growth = value * annual_return
            value += growth
            if category in ["investment", "cash", "other"]:
                liquid_growth += growth

            sale_age = int(row["sale_age"])
            sale_proceeds = to_base(inputs, float(row["sale_proceeds"]), str(row["currency"]))
            name = str(row["name"])
            if sale_age > 0 and age >= sale_age and not sale_done.get(name, False):
                if sale_proceeds > 0:
                    sale_inflow += sale_proceeds
                    if category == "property":
                        value = max(0.0, value - sale_proceeds)
                sale_done[name] = True

            assets.at[idx, "base_value"] = max(0.0, value)

        debt_paid = 0.0
        liabilities_end = 0.0
        for idx, row in debt.iterrows():
            bal = float(debt.at[idx, "base_balance"])
            new_bal, paid = amortize_one_year(bal, float(row["interest_rate"]), to_base(inputs, float(row["monthly_payment"]), str(row["currency"])))
            debt.at[idx, "base_balance"] = new_bal
            debt_paid += paid
            if bool(row["include_in_net_worth"]):
                liabilities_end += new_bal

        pensions = annual_pensions(age, inputs)
        other_income = annual_other_income(age, inputs)
        base_spending = annual_base_spending(age, inputs) if retired else 0.0
        life_expenses = annual_expenses(age, inputs)
        total_spending = base_spending + debt_paid + life_expenses

        liquid_total = float(assets.loc[liquid_mask, "base_value"].sum()) if not assets.empty and liquid_mask.any() else 0.0
        liquid_total += sale_inflow
        reserve_floor = inputs.settings["emergency_cash_years"] * (base_spending + annual_healthcare(age, inputs))
        net_need = max(0.0, total_spending - pensions - other_income)
        available_draw = max(0.0, liquid_total - reserve_floor)
        gross_draw = min(available_draw, net_need)

        if not assets.empty and liquid_mask.any() and gross_draw > 0:
            liquid_before = float(assets.loc[liquid_mask, "base_value"].sum())
            if liquid_before > 0:
                ratio = max(0.0, (liquid_before - gross_draw) / liquid_before)
                assets.loc[liquid_mask, "base_value"] = assets.loc[liquid_mask, "base_value"] * ratio

        liquid_end = float(assets.loc[liquid_mask, "base_value"].sum()) if not assets.empty and liquid_mask.any() else 0.0
        non_liquid_end = float(assets.loc[~liquid_mask, "base_value"].sum()) if not assets.empty else 0.0
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
                "pensions": pensions,
                "other_income": other_income,
                "base_spending": base_spending,
                "debt_paid": debt_paid,
                "life_expenses": life_expenses,
                "total_spending": total_spending,
                "net_portfolio_draw": gross_draw,
                "reserve_floor": reserve_floor,
                "liquid_growth": liquid_growth,
            }
        )
    return pd.DataFrame(rows)


def monte_carlo(inputs: Inputs) -> pd.DataFrame:
    rng = np.random.default_rng(inputs.random_seed)
    assets = inputs.assets.copy()
    assets = assets[assets["enabled"] == True].copy() if "enabled" in assets.columns else assets.copy()
    liquid_assets = assets[assets["category"].astype(str).isin(["investment", "cash", "other"])] if not assets.empty else pd.DataFrame()

    if liquid_assets.empty:
        avg_return, avg_vol = 0.0, 0.0
    else:
        vals = np.array([to_base(inputs, float(v), str(c)) for v, c in zip(liquid_assets["value"], liquid_assets["currency"])], dtype=float)
        total = vals.sum()
        if total <= 0:
            avg_return = float(liquid_assets["annual_return"].astype(float).mean())
            avg_vol = float(liquid_assets["volatility"].astype(float).mean())
        else:
            weights = vals / total
            avg_return = float((weights * liquid_assets["annual_return"].astype(float).to_numpy()).sum())
            avg_vol = float((weights * liquid_assets["volatility"].astype(float).to_numpy()).sum())

    years = max(1, inputs.life_expectancy - inputs.retirement_age + 1)
    paths = []
    for _ in range(inputs.mc_runs):
        returns = rng.normal(avg_return, avg_vol, years)
        proj = build_projection(inputs, returns)
        retirement = proj[proj["age"] >= inputs.retirement_age]
        paths.append(retirement["liquid_assets_end"].to_numpy())
    return pd.DataFrame(paths, columns=list(range(inputs.retirement_age, inputs.life_expectancy + 1)))


@st.cache_data(show_spinner=False)
def run_results(snapshot: dict):
    inputs = Inputs(
        settings=snapshot["settings"],
        household=from_records(snapshot["household"]),
        assets=from_records(snapshot["assets"]),
        debt=from_records(snapshot["debt"]),
        income=from_records(snapshot["income"]),
        expenses=from_records(snapshot["expenses"]),
        fx=from_records(snapshot["fx"]),
        tax=from_records(snapshot["tax"]),
    )
    proj = build_projection(inputs)
    mc = monte_carlo(inputs)
    finals = mc.iloc[:, -1]
    summary = {
        "success_rate": float((finals > 0).mean()),
        "median_final_liquid": float(finals.median()),
        "p10_final_liquid": float(finals.quantile(0.10)),
    }
    return proj, mc, summary


def snapshot_state() -> dict:
    return {
        "settings": dict(st.session_state["settings"]),
        "household": list(st.session_state["household_records"]),
        "assets": list(st.session_state["asset_records"]),
        "debt": list(st.session_state["debt_records"]),
        "income": list(st.session_state["income_records"]),
        "expenses": list(st.session_state["expense_records"]),
        "fx": list(st.session_state["fx_records"]),
        "tax": list(st.session_state["tax_records"]),
    }


def optimize(inputs: Inputs) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    base_proj = build_projection(inputs)
    base_mc = monte_carlo(inputs)
    finals = base_mc.iloc[:, -1]
    base_success = float((finals > 0).mean())
    base_ret = base_proj[base_proj["age"] == inputs.retirement_age].iloc[0]
    rows.append({"strategy": "Current plan", "retirement_age": inputs.retirement_age, "success_rate": base_success, "liquid_at_retirement": float(base_ret["liquid_assets_end"])})

    for extra in range(1, int(inputs.settings["optimizer_max_extra_work_years"]) + 1):
        trial_household = inputs.household.copy()
        trial_household["retirement_age"] = trial_household["retirement_age"].astype(int) + extra
        trial = Inputs(inputs.settings, trial_household, inputs.assets, inputs.debt, inputs.income, inputs.expenses, inputs.fx, inputs.tax)
        trial_proj = build_projection(trial)
        trial_mc = monte_carlo(trial)
        trial_success = float((trial_mc.iloc[:, -1] > 0).mean())
        trial_ret = trial_proj[trial_proj["age"] == trial.retirement_age].iloc[0]
        rows.append({"strategy": f"Work {extra} more year(s)", "retirement_age": trial.retirement_age, "success_rate": trial_success, "liquid_at_retirement": float(trial_ret["liquid_assets_end"])})

    result = pd.DataFrame(rows).sort_values(["success_rate", "liquid_at_retirement"], ascending=[False, False]).reset_index(drop=True)
    notes = []
    if not result.empty:
        notes.append(f"Best simple lever right now: **{result.iloc[0]['strategy']}**.")
        notes.append(f"That gets to about **{fmt_pct(float(result.iloc[0]['success_rate']))}** success.")
    return result, notes

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    settings = st.session_state["settings"]
    st.header("Planner")
    settings["display_currency"] = st.selectbox("Display currency", list(CURRENCIES.keys()), index=list(CURRENCIES.keys()).index(settings["display_currency"]))
    settings["base_currency"] = st.selectbox("Model base currency", list(CURRENCIES.keys()), index=list(CURRENCIES.keys()).index(settings["base_currency"]))
    settings["scenario_name"] = st.text_input("Scenario name", value=settings["scenario_name"])
    st.session_state["settings"] = settings

    st.divider()
    st.header("Section")
    st.session_state["page"] = st.radio("Go to", PAGES, index=PAGES.index(st.session_state["page"]))

    st.divider()
    st.header("Calculation")
    st.session_state["auto_recalc"] = st.checkbox("Auto recalculate", value=st.session_state["auto_recalc"])
    if st.button("Apply inputs and recalculate", use_container_width=True, type="primary"):
        st.session_state["apply_counter"] += 1

page = st.session_state["page"]

# ------------------------------------------------------------
# Pages
# ------------------------------------------------------------
st.title("Retirement Planner")
st.caption("This version is intentionally simpler and more stable. Each section explains itself, and heavy calculations only run on Results or Optimizer after you apply inputs.")

if page == "Welcome":
    st.subheader("How to use this app")
    st.markdown(
        """
1. Fill in the sections on the left one by one.
2. Keep **Auto recalculate** off while entering data.
3. Click **Apply inputs and recalculate** when you want updated results.
4. Open **Results** to see the plan.
5. Open **Optimizer** to test simple improvement options.

This app has been simplified to reduce flaky input behavior and make the workflow easier to follow.
"""
    )

elif page == "Household":
    st.subheader("Household")
    st.caption("Who is in the plan. Disable the spouse row for a one-person plan.")
    st.info("Use pension fields for expected state pension or similar guaranteed retirement income.")
    st.session_state["household_records"] = st.data_editor(
        from_records(st.session_state["household_records"]),
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
            "pension_annual": st.column_config.NumberColumn("Annual pension", step=1000, format="%.0f"),
            "pension_currency": st.column_config.SelectboxColumn("Pension currency", options=list(CURRENCIES.keys())),
        },
    ).to_dict("records")

elif page == "Assets":
    st.subheader("Assets")
    st.caption("Add savings, portfolios, properties, and cash reserves.")
    st.info("Use 'monthly contribution' only for assets you are still adding money to before retirement.")
    st.session_state["asset_records"] = st.data_editor(
        from_records(st.session_state["asset_records"]),
        num_rows="dynamic",
        use_container_width=True,
        key="asset_editor",
        column_config={
            "enabled": st.column_config.CheckboxColumn("Enabled"),
            "name": st.column_config.TextColumn("Name"),
            "category": st.column_config.SelectboxColumn("Category", options=["investment", "cash", "property", "other"]),
            "account_type": st.column_config.SelectboxColumn("Account type", options=["taxable", "pension", "isa", "cash", "property", "other"]),
            "currency": st.column_config.SelectboxColumn("Currency", options=list(CURRENCIES.keys())),
            "value": st.column_config.NumberColumn("Current value", step=1000, format="%.0f"),
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
    ).to_dict("records")

elif page == "Debt":
    st.subheader("Debt")
    st.caption("Add mortgages and other loans.")
    st.info("Monthly payment is what you actually expect to pay. The model amortizes the balance each year.")
    st.session_state["debt_records"] = st.data_editor(
        from_records(st.session_state["debt_records"]),
        num_rows="dynamic",
        use_container_width=True,
        key="debt_editor",
        column_config={
            "enabled": st.column_config.CheckboxColumn("Enabled"),
            "name": st.column_config.TextColumn("Name"),
            "linked_asset": st.column_config.TextColumn("Linked asset"),
            "currency": st.column_config.SelectboxColumn("Currency", options=list(CURRENCIES.keys())),
            "balance": st.column_config.NumberColumn("Balance", step=1000, format="%.0f"),
            "interest_rate": st.column_config.NumberColumn("Interest rate", step=0.001, format="%.3f"),
            "monthly_payment": st.column_config.NumberColumn("Monthly payment", step=100, format="%.0f"),
            "include_in_net_worth": st.column_config.CheckboxColumn("Include in net worth"),
        },
    ).to_dict("records")

elif page == "Income":
    st.subheader("Other income")
    st.caption("Add consulting, rental income, part-time work, or anything else that comes in during specific ages.")
    st.session_state["income_records"] = st.data_editor(
        from_records(st.session_state["income_records"]),
        num_rows="dynamic",
        use_container_width=True,
        key="income_editor",
        column_config={
            "enabled": st.column_config.CheckboxColumn("Enabled"),
            "name": st.column_config.TextColumn("Name"),
            "currency": st.column_config.SelectboxColumn("Currency", options=list(CURRENCIES.keys())),
            "annual_amount": st.column_config.NumberColumn("Annual amount", step=1000, format="%.0f"),
            "start_age": st.column_config.NumberColumn("Start age", step=1),
            "end_age": st.column_config.NumberColumn("End age", step=1),
            "inflation_linked": st.column_config.CheckboxColumn("Inflation linked"),
        },
    ).to_dict("records")

elif page == "Expenses":
    st.subheader("Life events and large expenses")
    st.caption("Add weddings, education, holidays, house works, car purchases, or leases.")
    st.info("Use 'one_off' for a single event, 'monthly' for leases, and 'annual' for repeating yearly items.")
    st.session_state["expense_records"] = st.data_editor(
        from_records(st.session_state["expense_records"]),
        num_rows="dynamic",
        use_container_width=True,
        key="expense_editor",
        column_config={
            "enabled": st.column_config.CheckboxColumn("Enabled"),
            "name": st.column_config.TextColumn("Name"),
            "currency": st.column_config.SelectboxColumn("Currency", options=list(CURRENCIES.keys())),
            "mode": st.column_config.SelectboxColumn("Mode", options=["one_off", "monthly", "annual"]),
            "amount": st.column_config.NumberColumn("Amount", step=1000, format="%.0f"),
            "start_age": st.column_config.NumberColumn("Start age", step=1),
            "end_age": st.column_config.NumberColumn("End age", step=1),
            "inflation_linked": st.column_config.CheckboxColumn("Inflation linked"),
        },
    ).to_dict("records")

elif page == "FX & Tax":
    st.subheader("FX and tax")
    st.caption("Use this only if you need multiple currencies or account-level tax assumptions.")
    st.info("If you want simplicity, leave these close to the defaults.")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state["fx_records"] = st.data_editor(
            from_records(st.session_state["fx_records"]),
            num_rows="dynamic",
            use_container_width=True,
            key="fx_editor",
            column_config={
                "currency": st.column_config.SelectboxColumn("Currency", options=list(CURRENCIES.keys())),
                "to_base": st.column_config.NumberColumn("To base", step=0.01, format="%.4f"),
            },
        ).to_dict("records")
    with c2:
        settings = st.session_state["settings"]
        settings["tax_enabled"] = st.checkbox("Enable tax model", value=bool(settings["tax_enabled"]))
        st.session_state["settings"] = settings
        st.session_state["tax_records"] = st.data_editor(
            from_records(st.session_state["tax_records"]),
            num_rows="dynamic",
            use_container_width=True,
            key="tax_editor",
            column_config={
                "account_type": st.column_config.SelectboxColumn("Account type", options=["taxable", "pension", "isa", "cash", "property", "other"]),
                "withdrawal_tax_rate": st.column_config.NumberColumn("Withdrawal tax", step=0.01, format="%.2f"),
                "growth_tax_drag": st.column_config.NumberColumn("Growth tax drag", step=0.001, format="%.3f"),
            },
        ).to_dict("records")

elif page == "Settings":
    st.subheader("Planning settings")
    st.caption("These are the core assumptions that drive the whole model.")
    st.info("Keep this page short and understandable. You can refine later.")
    settings = st.session_state["settings"]
    c1, c2, c3, c4 = st.columns(4)
    settings["base_spending_pre75"] = c1.number_input("Base spending before 75", min_value=0.0, value=float(settings["base_spending_pre75"]), step=1000.0)
    settings["base_spending_post75"] = c2.number_input("Base spending after 75", min_value=0.0, value=float(settings["base_spending_post75"]), step=1000.0)
    settings["inflation"] = c3.slider("Inflation", 0.0, 0.10, float(settings["inflation"]), step=0.001, format="%.1f%%")
    settings["stress_uplift"] = c4.slider("Stress uplift", -0.20, 0.30, float(settings["stress_uplift"]), step=0.01, format="%.0f%%")

    c5, c6, c7 = st.columns(3)
    settings["mc_runs"] = c5.number_input("Monte Carlo runs", 500, 5000, int(settings["mc_runs"]), step=500)
    settings["random_seed"] = c6.number_input("Random seed", 0, 999999, int(settings["random_seed"]), step=1)
    settings["emergency_cash_years"] = c7.slider("Emergency cash floor in years", 0.0, 5.0, float(settings["emergency_cash_years"]), step=0.5)

    c8, c9, c10 = st.columns(3)
    settings["healthcare_enabled"] = c8.checkbox("Healthcare ramp", value=bool(settings["healthcare_enabled"]))
    settings["healthcare_start_age"] = c9.number_input("Healthcare start age", 50, 110, int(settings["healthcare_start_age"]))
    settings["healthcare_base_annual"] = c10.number_input("Healthcare annual", 0.0, value=float(settings["healthcare_base_annual"]), step=1000.0)

    c11, c12 = st.columns(2)
    settings["legacy_target"] = c11.number_input("Legacy target", 0.0, value=float(settings["legacy_target"]), step=10000.0)
    settings["target_success_rate"] = c12.slider("Target success rate", 0.50, 0.99, float(settings["target_success_rate"]), step=0.01, format="%.0f%%")

    settings["optimizer_max_extra_work_years"] = st.number_input("Max extra work years to test", 1, 20, int(settings["optimizer_max_extra_work_years"]), step=1)
    st.session_state["settings"] = settings

elif page == "Results":
    st.subheader("Results")
    st.caption("This is where the heavy calculations run.")
    if not (st.session_state["auto_recalc"] or st.session_state["apply_counter"] > 0):
        st.info("Click **Apply inputs and recalculate** in the sidebar first.")
    else:
        snapshot = snapshot_state()
        proj, mc, summary = run_results(snapshot)
        inputs = get_inputs()
        ret_row = proj[proj["age"] == inputs.retirement_age].iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current net worth", fmt_money(from_base(inputs, float(proj.iloc[0]["net_worth_end"]), inputs.display_currency), inputs.display_currency))
        m2.metric("Liquid at retirement", fmt_money(from_base(inputs, float(ret_row["liquid_assets_end"]), inputs.display_currency), inputs.display_currency))
        m3.metric("Success rate", fmt_pct(summary["success_rate"]))
        m4.metric("Withdrawal rate at retirement", fmt_pct(safe_div(float(ret_row["net_portfolio_draw"]), float(ret_row["liquid_assets_end"]))))

        m5, m6, m7 = st.columns(3)
        m5.metric("Net draw at retirement", fmt_money(from_base(inputs, float(ret_row["net_portfolio_draw"]), inputs.display_currency), inputs.display_currency))
        m6.metric("Median final liquid", fmt_money(from_base(inputs, float(summary["median_final_liquid"]), inputs.display_currency), inputs.display_currency))
        m7.metric("10th percentile final liquid", fmt_money(from_base(inputs, float(summary["p10_final_liquid"]), inputs.display_currency), inputs.display_currency))

        st.subheader("Projection")
        chart = proj[["age", "liquid_assets_end", "net_worth_end"]].copy().set_index("age")
        chart = chart.apply(lambda col: col.map(lambda x: from_base(inputs, x, inputs.display_currency)))
        st.line_chart(chart)

        st.subheader("Monte Carlo")
        mc_chart = pd.DataFrame(
            {
                "Age": mc.columns.astype(int),
                "Median": mc.median(axis=0).values,
                "10th percentile": mc.quantile(0.10, axis=0).values,
                "90th percentile": mc.quantile(0.90, axis=0).values,
            }
        ).set_index("Age")
        mc_chart = mc_chart.apply(lambda col: col.map(lambda x: from_base(inputs, x, inputs.display_currency)))
        st.line_chart(mc_chart)

elif page == "Optimizer":
    st.subheader("Optimizer")
    st.caption("This is a first-pass optimizer. It keeps the logic easy to understand.")
    if not (st.session_state["auto_recalc"] or st.session_state["apply_counter"] > 0):
        st.info("Click **Apply inputs and recalculate** in the sidebar first.")
    else:
        inputs = get_inputs()
        results, notes = optimize(inputs)
        show = results.copy()
        show["success_rate"] = show["success_rate"].map(lambda x: fmt_pct(float(x)))
        show["liquid_at_retirement"] = show["liquid_at_retirement"].map(lambda x: fmt_money(from_base(inputs, float(x), inputs.display_currency), inputs.display_currency))
        st.dataframe(show.head(12), use_container_width=True, hide_index=True)
        for note in notes:
            st.markdown(f"- {note}")

st.divider()
st.markdown("This rebuild focuses on stability and easier input, not maximum feature density.")
