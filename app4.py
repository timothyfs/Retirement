
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Retirement Planner", page_icon="📈", layout="wide")

CURRENCIES: Dict[str, Dict[str, str]] = {
    "EUR": {"symbol": "€", "name": "Euro"},
    "GBP": {"symbol": "£", "name": "British Pound"},
    "USD": {"symbol": "$", "name": "US Dollar"},
    "CHF": {"symbol": "CHF ", "name": "Swiss Franc"},
}

PAGES = ["Basic Inputs", "Advanced Assumptions", "Results", "Optimizer"]
ASSET_TYPES = ["Investment", "Cash", "Property", "Other"]
DEBT_TYPES = ["Mortgage", "Loan", "Other"]
EXPENSE_TYPES = ["One off", "Monthly", "Annual"]


def fmt_money(value: float, currency: str) -> str:
    symbol = CURRENCIES.get(currency, CURRENCIES["EUR"])["symbol"]
    return f"{symbol}{value:,.0f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}"


def as_float(v, default=0.0) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, str):
            s = v.strip().replace(",", "")
            if s == "":
                return default
            return float(s)
        return float(v)
    except Exception:
        return default


def as_int(v, default=0) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, str):
            s = v.strip().replace(",", "")
            if s == "":
                return default
            return int(float(s))
        return int(float(v))
    except Exception:
        return default


def as_bool(v, default=False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "yes", "1", "y"}:
            return True
        if s in {"false", "no", "0", "n", ""}:
            return False
    return default


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def df_from_records(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def clean_currency(currency: str, default: str = "EUR") -> str:
    return currency if currency in CURRENCIES else default


DEFAULT_SETTINGS = {
    "scenario_name": "Base",
    "display_currency": "EUR",
    "base_currency": "EUR",
    "inflation": 0.025,
    "spending_pre75": 100000.0,
    "spending_post75": 80000.0,
    "healthcare_enabled": True,
    "healthcare_start_age": 75,
    "healthcare_annual": 8000.0,
    "healthcare_extra_inflation": 0.02,
    "legacy_target": 250000.0,
    "emergency_cash_years": 2.0,
    "enable_monte_carlo": False,
    "mc_runs": 200,
    "random_seed": 42,
    "optimizer_max_extra_years": 10,
}

DEFAULT_HOUSEHOLD = [
    {"enabled": True, "name": "You", "current_age": 53, "retirement_age": 55, "life_expectancy": 95, "pension_age": 67, "pension_annual": 14000.0, "pension_currency": "GBP"},
    {"enabled": True, "name": "Spouse", "current_age": 51, "retirement_age": 55, "life_expectancy": 95, "pension_age": 67, "pension_annual": 13000.0, "pension_currency": "GBP"},
]

DEFAULT_ASSETS = [
    {"enabled": True, "name": "Investment portfolio", "asset_type": "Investment", "currency": "EUR", "value": 1070000.0, "annual_return": 0.06, "volatility": 0.14, "monthly_contribution": 5000.0, "sale_age": 0, "sale_proceeds": 0.0, "income_annual": 0.0, "income_start_age": 0, "income_end_age": 0, "inflation_linked_income": False},
    {"enabled": True, "name": "Main property", "asset_type": "Property", "currency": "EUR", "value": 1020000.0, "annual_return": 0.025, "volatility": 0.07, "monthly_contribution": 0.0, "sale_age": 75, "sale_proceeds": 500000.0, "income_annual": 0.0, "income_start_age": 0, "income_end_age": 0, "inflation_linked_income": False},
    {"enabled": True, "name": "Rental income source", "asset_type": "Property", "currency": "EUR", "value": 0.0, "annual_return": 0.0, "volatility": 0.0, "monthly_contribution": 0.0, "sale_age": 0, "sale_proceeds": 0.0, "income_annual": 24000.0, "income_start_age": 55, "income_end_age": 95, "inflation_linked_income": False},
    {"enabled": True, "name": "Emergency cash", "asset_type": "Cash", "currency": "EUR", "value": 50000.0, "annual_return": 0.02, "volatility": 0.01, "monthly_contribution": 0.0, "sale_age": 0, "sale_proceeds": 0.0, "income_annual": 0.0, "income_start_age": 0, "income_end_age": 0, "inflation_linked_income": False},
]

DEFAULT_DEBTS = [
    {"enabled": True, "name": "Main mortgage", "debt_type": "Mortgage", "currency": "EUR", "balance": 50000.0, "interest_rate": 0.035, "monthly_payment": 1200.0, "include_in_net_worth": True}
]

DEFAULT_EXTRA_INCOME = [
    {"enabled": True, "name": "Consulting", "currency": "EUR", "annual_amount": 10000.0, "start_age": 55, "end_age": 60, "inflation_linked": False}
]

DEFAULT_EXPENSES = [
    {"enabled": False, "name": "Car purchase", "currency": "EUR", "expense_type": "One off", "amount": 50000.0, "start_age": 56, "end_age": 56, "inflation_linked": False}
]


def init_state() -> None:
    defaults = {
        "settings": dict(DEFAULT_SETTINGS),
        "household_records": list(DEFAULT_HOUSEHOLD),
        "asset_records": list(DEFAULT_ASSETS),
        "debt_records": list(DEFAULT_DEBTS),
        "extra_income_records": list(DEFAULT_EXTRA_INCOME),
        "expense_records": list(DEFAULT_EXPENSES),
        "page": "Basic Inputs",
        "run_counter": 0,
        "auto_recalc": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


@dataclass
class Inputs:
    settings: dict
    household: pd.DataFrame
    assets: pd.DataFrame
    debts: pd.DataFrame
    extra_income: pd.DataFrame
    expenses: pd.DataFrame

    @property
    def display_currency(self) -> str:
        return clean_currency(self.settings["display_currency"])

    @property
    def current_age(self) -> int:
        df = self.household[self.household["enabled"] == True]
        return 55 if df.empty else as_int(df["current_age"].max(), 55)

    @property
    def retirement_age(self) -> int:
        df = self.household[self.household["enabled"] == True]
        return 55 if df.empty else as_int(df["retirement_age"].max(), 55)

    @property
    def life_expectancy(self) -> int:
        df = self.household[self.household["enabled"] == True]
        return 95 if df.empty else as_int(df["life_expectancy"].max(), 95)

    @property
    def mc_runs(self) -> int:
        return as_int(self.settings["mc_runs"], 200)

    @property
    def random_seed(self) -> int:
        return as_int(self.settings["random_seed"], 42)

    @property
    def monte_carlo_enabled(self) -> bool:
        return as_bool(self.settings.get("enable_monte_carlo", False), False)


def clean_household(records):
    df = df_from_records(records)
    if df.empty:
        return pd.DataFrame(columns=["enabled","name","current_age","retirement_age","life_expectancy","pension_age","pension_annual","pension_currency"])
    df["enabled"] = df.get("enabled", False).apply(as_bool)
    df["name"] = df.get("name", "").apply(str)
    for col in ["current_age", "retirement_age", "life_expectancy", "pension_age"]:
        df[col] = df.get(col, 0).apply(as_int)
    df["pension_annual"] = df.get("pension_annual", 0.0).apply(as_float)
    df["pension_currency"] = df.get("pension_currency", "EUR").apply(clean_currency)
    return df


def clean_assets(records):
    df = df_from_records(records)
    if df.empty:
        return pd.DataFrame(columns=["enabled","name","asset_type","currency","value","annual_return","volatility","monthly_contribution","sale_age","sale_proceeds","income_annual","income_start_age","income_end_age","inflation_linked_income"])
    df["enabled"] = df.get("enabled", False).apply(as_bool)
    df["name"] = df.get("name", "").apply(str)
    df["asset_type"] = df.get("asset_type", "Other").apply(lambda x: x if x in ASSET_TYPES else "Other")
    df["currency"] = df.get("currency", "EUR").apply(clean_currency)
    for col in ["value","annual_return","volatility","monthly_contribution","sale_proceeds","income_annual"]:
        df[col] = df.get(col, 0.0).apply(as_float)
    for col in ["sale_age","income_start_age","income_end_age"]:
        df[col] = df.get(col, 0).apply(as_int)
    df["inflation_linked_income"] = df.get("inflation_linked_income", False).apply(as_bool)
    return df


def clean_debts(records):
    df = df_from_records(records)
    if df.empty:
        return pd.DataFrame(columns=["enabled","name","debt_type","currency","balance","interest_rate","monthly_payment","include_in_net_worth"])
    df["enabled"] = df.get("enabled", False).apply(as_bool)
    df["name"] = df.get("name", "").apply(str)
    df["debt_type"] = df.get("debt_type", "Other").apply(lambda x: x if x in DEBT_TYPES else "Other")
    df["currency"] = df.get("currency", "EUR").apply(clean_currency)
    for col in ["balance","interest_rate","monthly_payment"]:
        df[col] = df.get(col, 0.0).apply(as_float)
    df["include_in_net_worth"] = df.get("include_in_net_worth", True).apply(as_bool)
    return df


def clean_extra_income(records):
    df = df_from_records(records)
    if df.empty:
        return pd.DataFrame(columns=["enabled","name","currency","annual_amount","start_age","end_age","inflation_linked"])
    df["enabled"] = df.get("enabled", False).apply(as_bool)
    df["name"] = df.get("name", "").apply(str)
    df["currency"] = df.get("currency", "EUR").apply(clean_currency)
    df["annual_amount"] = df.get("annual_amount", 0.0).apply(as_float)
    df["start_age"] = df.get("start_age", 0).apply(as_int)
    df["end_age"] = df.get("end_age", 0).apply(as_int)
    df["inflation_linked"] = df.get("inflation_linked", False).apply(as_bool)
    return df


def clean_expenses(records):
    df = df_from_records(records)
    if df.empty:
        return pd.DataFrame(columns=["enabled","name","currency","expense_type","amount","start_age","end_age","inflation_linked"])
    df["enabled"] = df.get("enabled", False).apply(as_bool)
    df["name"] = df.get("name", "").apply(str)
    df["currency"] = df.get("currency", "EUR").apply(clean_currency)
    df["expense_type"] = df.get("expense_type", "One off").apply(lambda x: x if x in EXPENSE_TYPES else "One off")
    df["amount"] = df.get("amount", 0.0).apply(as_float)
    df["start_age"] = df.get("start_age", 0).apply(as_int)
    df["end_age"] = df.get("end_age", 0).apply(as_int)
    df["inflation_linked"] = df.get("inflation_linked", False).apply(as_bool)
    return df


def get_inputs_from_state():
    warnings = []
    inputs = Inputs(
        settings=dict(st.session_state["settings"]),
        household=clean_household(st.session_state["household_records"]),
        assets=clean_assets(st.session_state["asset_records"]),
        debts=clean_debts(st.session_state["debt_records"]),
        extra_income=clean_extra_income(st.session_state["extra_income_records"]),
        expenses=clean_expenses(st.session_state["expense_records"]),
    )
    if inputs.household.empty or inputs.household["enabled"].sum() == 0:
        warnings.append("No enabled household member found.")
    if inputs.assets.empty or inputs.assets["enabled"].sum() == 0:
        warnings.append("No enabled asset found.")
    return inputs, warnings


def annual_base_spending(age, inputs):
    base = as_float(inputs.settings["spending_pre75"], 0.0) if age < 75 else as_float(inputs.settings["spending_post75"], 0.0)
    years = max(0, age - inputs.retirement_age)
    return base * ((1.0 + as_float(inputs.settings["inflation"], 0.025)) ** years)


def annual_healthcare(age, inputs):
    if not as_bool(inputs.settings["healthcare_enabled"], True):
        return 0.0
    start_age = as_int(inputs.settings["healthcare_start_age"], 75)
    if age < start_age:
        return 0.0
    years = age - start_age
    annual = as_float(inputs.settings["healthcare_annual"], 0.0)
    extra = as_float(inputs.settings["healthcare_extra_inflation"], 0.02)
    inflation = as_float(inputs.settings["inflation"], 0.025)
    return annual * ((1.0 + inflation + extra) ** years)


def annual_pension_income(age, inputs):
    total = 0.0
    inflation = as_float(inputs.settings["inflation"], 0.025)
    for _, row in inputs.household.iterrows():
        if not as_bool(row["enabled"], True):
            continue
        pension_age = as_int(row["pension_age"], 0)
        if age < pension_age:
            continue
        years = age - pension_age
        total += as_float(row["pension_annual"], 0.0) * ((1.0 + inflation) ** years)
    return total


def annual_other_income(age, inputs):
    total = 0.0
    inflation = as_float(inputs.settings["inflation"], 0.025)
    for _, row in inputs.extra_income.iterrows():
        if not as_bool(row["enabled"], True):
            continue
        start_age = as_int(row["start_age"], 0)
        end_age = as_int(row["end_age"], 0)
        if age < start_age or age > end_age:
            continue
        amt = as_float(row["annual_amount"], 0.0)
        if as_bool(row["inflation_linked"], False):
            amt *= ((1.0 + inflation) ** max(0, age - start_age))
        total += amt
    for _, row in inputs.assets.iterrows():
        if not as_bool(row["enabled"], True):
            continue
        start_age = as_int(row["income_start_age"], 0)
        end_age = as_int(row["income_end_age"], 0)
        end_age = end_age if end_age > 0 else inputs.life_expectancy
        if age < start_age or age > end_age:
            continue
        amt = as_float(row["income_annual"], 0.0)
        if as_bool(row["inflation_linked_income"], False):
            amt *= ((1.0 + inflation) ** max(0, age - start_age))
        total += amt
    return total


def annual_event_expenses(age, inputs):
    total = annual_healthcare(age, inputs)
    inflation = as_float(inputs.settings["inflation"], 0.025)
    for _, row in inputs.expenses.iterrows():
        if not as_bool(row["enabled"], True):
            continue
        start_age = as_int(row["start_age"], 0)
        end_age = as_int(row["end_age"], 0)
        if age < start_age or age > end_age:
            continue
        expense_type = str(row["expense_type"])
        if expense_type == "One off" and age != start_age:
            continue
        amt = as_float(row["amount"], 0.0)
        if expense_type == "Monthly":
            amt *= 12.0
        if as_bool(row["inflation_linked"], False):
            amt *= ((1.0 + inflation) ** max(0, age - start_age))
        total += amt
    return total


def amortize_one_year(balance, annual_rate, monthly_payment):
    bal = as_float(balance, 0.0)
    paid = 0.0
    for _ in range(12):
        if bal <= 0:
            break
        bal += bal * (annual_rate / 12.0)
        payment = min(monthly_payment, bal) if monthly_payment > 0 else 0.0
        bal -= payment
        paid += payment
    return max(0.0, bal), paid


def build_projection(inputs, retirement_returns=None):
    assets = inputs.assets.copy()
    debts = inputs.debts.copy()
    enabled_assets = assets[assets["enabled"] == True].copy()
    enabled_debts = debts[debts["enabled"] == True].copy()

    if enabled_assets.empty:
        enabled_assets = pd.DataFrame(columns=assets.columns)

    enabled_assets["current_value"] = enabled_assets["value"].apply(as_float)
    enabled_debts["current_balance"] = enabled_debts["balance"].apply(as_float)

    sale_done = {str(name): False for name in enabled_assets.get("name", pd.Series(dtype=str)).tolist()}
    rows = []

    for age in range(inputs.current_age, inputs.life_expectancy + 1):
        retired = age >= inputs.retirement_age
        contributions = 0.0
        sale_inflow = 0.0

        liquid_mask = enabled_assets["asset_type"].astype(str).isin(["Investment", "Cash", "Other"]) if not enabled_assets.empty else pd.Series(dtype=bool)

        for idx, row in enabled_assets.iterrows():
            value = as_float(enabled_assets.at[idx, "current_value"], 0.0)
            asset_type = str(row["asset_type"])
            if not retired and asset_type in ["Investment", "Cash", "Other"]:
                contrib = as_float(row["monthly_contribution"], 0.0) * 12.0
                value += contrib
                contributions += contrib
            annual_return = as_float(row["annual_return"], 0.0)
            if retirement_returns is not None and retired and asset_type in ["Investment", "Cash", "Other"]:
                annual_return = as_float(retirement_returns[age - inputs.retirement_age], annual_return)
            value += value * annual_return
            sale_age = as_int(row["sale_age"], 0)
            sale_proceeds = as_float(row["sale_proceeds"], 0.0)
            name = str(row["name"])
            if sale_age > 0 and age >= sale_age and not sale_done.get(name, False):
                if sale_proceeds > 0:
                    sale_inflow += sale_proceeds
                    if asset_type == "Property":
                        value = max(0.0, value - sale_proceeds)
                sale_done[name] = True
            enabled_assets.at[idx, "current_value"] = max(0.0, value)

        debt_paid = 0.0
        liabilities_end = 0.0
        for idx, row in enabled_debts.iterrows():
            new_balance, paid = amortize_one_year(as_float(enabled_debts.at[idx, "current_balance"], 0.0), as_float(row["interest_rate"], 0.0), as_float(row["monthly_payment"], 0.0))
            enabled_debts.at[idx, "current_balance"] = new_balance
            debt_paid += paid
            if as_bool(row["include_in_net_worth"], True):
                liabilities_end += new_balance

        pensions = annual_pension_income(age, inputs)
        other_income = annual_other_income(age, inputs)
        base_spending = annual_base_spending(age, inputs) if retired else 0.0
        event_expenses = annual_event_expenses(age, inputs)
        total_spending = base_spending + debt_paid + event_expenses

        liquid_total = float(enabled_assets.loc[liquid_mask, "current_value"].sum()) if not enabled_assets.empty and liquid_mask.any() else 0.0
        liquid_total += sale_inflow
        reserve_floor = as_float(inputs.settings["emergency_cash_years"], 2.0) * (base_spending + annual_healthcare(age, inputs))
        net_need = max(0.0, total_spending - pensions - other_income)
        available_draw = max(0.0, liquid_total - reserve_floor)
        draw = min(available_draw, net_need)

        if not enabled_assets.empty and liquid_mask.any() and draw > 0:
            liquid_before = float(enabled_assets.loc[liquid_mask, "current_value"].sum())
            if liquid_before > 0:
                ratio = max(0.0, (liquid_before - draw) / liquid_before)
                enabled_assets.loc[liquid_mask, "current_value"] = enabled_assets.loc[liquid_mask, "current_value"] * ratio

        liquid_end = float(enabled_assets.loc[liquid_mask, "current_value"].sum()) if not enabled_assets.empty and liquid_mask.any() else 0.0
        non_liquid_end = float(enabled_assets.loc[~liquid_mask, "current_value"].sum()) if not enabled_assets.empty else 0.0
        net_worth = liquid_end + non_liquid_end - liabilities_end

        rows.append({
            "age": age,
            "phase": "Retirement" if retired else "Pre retirement",
            "liquid_assets_end": liquid_end,
            "non_liquid_assets_end": non_liquid_end,
            "liabilities_end": liabilities_end,
            "net_worth_end": net_worth,
            "contributions": contributions,
            "pensions": pensions,
            "other_income": other_income,
            "base_spending": base_spending,
            "debt_paid": debt_paid,
            "event_expenses": event_expenses,
            "total_spending": total_spending,
            "net_portfolio_draw": draw,
            "reserve_floor": reserve_floor,
        })

    return pd.DataFrame(rows)


def monte_carlo(inputs, progress_bar=None, progress_text=None):
    rng = np.random.default_rng(inputs.random_seed)
    enabled_assets = inputs.assets[inputs.assets["enabled"] == True].copy()
    liquid_assets = enabled_assets[enabled_assets["asset_type"].isin(["Investment", "Cash", "Other"])]

    if liquid_assets.empty:
        avg_return, avg_vol = 0.0, 0.0
    else:
        values = liquid_assets["value"].apply(as_float).to_numpy(dtype=float)
        total = values.sum()
        if total <= 0:
            avg_return = float(liquid_assets["annual_return"].apply(as_float).mean())
            avg_vol = float(liquid_assets["volatility"].apply(as_float).mean())
        else:
            weights = values / total
            avg_return = float((weights * liquid_assets["annual_return"].apply(as_float).to_numpy(dtype=float)).sum())
            avg_vol = float((weights * liquid_assets["volatility"].apply(as_float).to_numpy(dtype=float)).sum())

    years = max(1, inputs.life_expectancy - inputs.retirement_age + 1)
    paths = []
    runs = inputs.mc_runs
    for i in range(runs):
        returns = rng.normal(avg_return, avg_vol, years)
        projection = build_projection(inputs, returns)
        retirement_projection = projection[projection["age"] >= inputs.retirement_age]
        paths.append(retirement_projection["liquid_assets_end"].to_numpy())
        if progress_bar is not None and runs > 0:
            fraction = (i + 1) / runs
            progress_bar.progress(min(fraction, 1.0))
            if progress_text is not None:
                progress_text.caption(f"Monte Carlo progress: {i + 1:,} / {runs:,}")
    return pd.DataFrame(paths, columns=list(range(inputs.retirement_age, inputs.life_expectancy + 1)))


def make_snapshot():
    snapshot = {
        "settings": dict(st.session_state["settings"]),
        "household": list(st.session_state["household_records"]),
        "assets": list(st.session_state["asset_records"]),
        "debts": list(st.session_state["debt_records"]),
        "extra_income": list(st.session_state["extra_income_records"]),
        "expenses": list(st.session_state["expense_records"]),
    }
    return json.dumps(snapshot, sort_keys=True)


def optimize(inputs):
    rows = []
    base_projection = build_projection(inputs)
    base_ret = base_projection[base_projection["age"] == inputs.retirement_age].iloc[0]
    rows.append({"strategy": "Current plan", "retirement_age": inputs.retirement_age, "liquid_at_retirement": float(base_ret["liquid_assets_end"])})
    max_extra = as_int(inputs.settings["optimizer_max_extra_years"], 10)
    for extra in range(1, max_extra + 1):
        trial_household = inputs.household.copy()
        trial_household["retirement_age"] = trial_household["retirement_age"].apply(as_int) + extra
        trial_inputs = Inputs(inputs.settings, trial_household, inputs.assets, inputs.debts, inputs.extra_income, inputs.expenses)
        trial_projection = build_projection(trial_inputs)
        trial_ret = trial_projection[trial_projection["age"] == trial_inputs.retirement_age].iloc[0]
        rows.append({"strategy": f"Work {extra} more year(s)", "retirement_age": trial_inputs.retirement_age, "liquid_at_retirement": float(trial_ret["liquid_assets_end"])})
    result = pd.DataFrame(rows).sort_values(["liquid_at_retirement"], ascending=[False]).reset_index(drop=True)
    notes = [
        "The optimizer only uses the deterministic projection.",
        "It does not run Monte Carlo. That keeps it fast and easier to understand.",
        "Use Results first. Then use Optimizer to see which simple lever improves the outcome most."
    ]
    return result, notes


with st.sidebar:
    settings = st.session_state["settings"]
    st.header("Planner")
    settings["display_currency"] = st.selectbox("Display currency", list(CURRENCIES.keys()), index=list(CURRENCIES.keys()).index(clean_currency(settings["display_currency"])))
    settings["base_currency"] = st.selectbox("Model base currency", list(CURRENCIES.keys()), index=list(CURRENCIES.keys()).index(clean_currency(settings["base_currency"])))
    settings["scenario_name"] = st.text_input("Scenario name", value=str(settings["scenario_name"]))
    st.session_state["settings"] = settings

    st.divider()
    st.header("Section")
    st.session_state["page"] = st.radio("Go to", PAGES, index=PAGES.index(st.session_state["page"]))

    st.divider()
    st.header("Calculation")
    st.session_state["auto_recalc"] = st.checkbox("Auto recalculate", value=as_bool(st.session_state["auto_recalc"], False))
    if st.button("Run plan", use_container_width=True, type="primary"):
        st.session_state["run_counter"] += 1


page = st.session_state["page"]
st.title("Retirement Planner")
st.caption("Stable v1. Faster by default. Monte Carlo runs only when you explicitly enable it.")

if page == "Basic Inputs":
    st.subheader("Basic inputs")
    st.markdown("Enter the core facts of the plan. Keep this simple first.")
    st.markdown("**Household**")
    st.session_state["household_records"] = st.data_editor(df_from_records(st.session_state["household_records"]), num_rows="dynamic", use_container_width=True, key="household_editor_v1").to_dict("records")
    st.markdown("**Assets**")
    st.session_state["asset_records"] = st.data_editor(df_from_records(st.session_state["asset_records"]), num_rows="dynamic", use_container_width=True, key="asset_editor_v1").to_dict("records")
    st.markdown("**Debts**")
    st.session_state["debt_records"] = st.data_editor(df_from_records(st.session_state["debt_records"]), num_rows="dynamic", use_container_width=True, key="debt_editor_v1").to_dict("records")
    st.markdown("**Extra income**")
    st.session_state["extra_income_records"] = st.data_editor(df_from_records(st.session_state["extra_income_records"]), num_rows="dynamic", use_container_width=True, key="income_editor_v1").to_dict("records")
    st.markdown("**Large expenses**")
    st.session_state["expense_records"] = st.data_editor(df_from_records(st.session_state["expense_records"]), num_rows="dynamic", use_container_width=True, key="expense_editor_v1").to_dict("records")

elif page == "Advanced Assumptions":
    st.subheader("Advanced assumptions")
    st.markdown("Use number inputs instead of sliders so values are precise and stable.")
    s = st.session_state["settings"]
    c1, c2, c3 = st.columns(3)
    s["spending_pre75"] = c1.number_input("Annual spending before 75", min_value=0.0, value=as_float(s["spending_pre75"], 100000.0), step=1000.0)
    s["spending_post75"] = c2.number_input("Annual spending after 75", min_value=0.0, value=as_float(s["spending_post75"], 80000.0), step=1000.0)
    s["inflation"] = c3.number_input("Inflation", min_value=0.0, max_value=0.20, value=as_float(s["inflation"], 0.025), step=0.001, format="%.3f")
    c4, c5, c6 = st.columns(3)
    s["healthcare_enabled"] = c4.checkbox("Healthcare cost ramp", value=as_bool(s["healthcare_enabled"], True))
    s["healthcare_start_age"] = c5.number_input("Healthcare start age", min_value=50, max_value=110, value=as_int(s["healthcare_start_age"], 75), step=1)
    s["healthcare_annual"] = c6.number_input("Healthcare annual cost", min_value=0.0, value=as_float(s["healthcare_annual"], 8000.0), step=1000.0)
    c7, c8, c9 = st.columns(3)
    s["healthcare_extra_inflation"] = c7.number_input("Extra healthcare inflation", min_value=0.0, max_value=0.20, value=as_float(s["healthcare_extra_inflation"], 0.02), step=0.001, format="%.3f")
    s["legacy_target"] = c8.number_input("Legacy target", min_value=0.0, value=as_float(s["legacy_target"], 250000.0), step=10000.0)
    s["emergency_cash_years"] = c9.number_input("Emergency cash floor in years", min_value=0.0, max_value=5.0, value=as_float(s["emergency_cash_years"], 2.0), step=0.5)

    st.markdown("**Monte Carlo**")
    c10, c11, c12 = st.columns(3)
    s["enable_monte_carlo"] = c10.checkbox("Enable Monte Carlo", value=as_bool(s.get("enable_monte_carlo", False), False))
    s["mc_runs"] = c11.number_input("Monte Carlo runs", min_value=200, max_value=5000, value=as_int(s["mc_runs"], 200), step=100)
    c12.info("Higher run counts will slow the plan down.")
    s["optimizer_max_extra_years"] = st.number_input("Max extra work years to test", min_value=1, max_value=20, value=as_int(s["optimizer_max_extra_years"], 10), step=1)
    st.session_state["settings"] = s

elif page == "Results":
    st.subheader("Results")
    if not (st.session_state["auto_recalc"] or st.session_state["run_counter"] > 0):
        st.info("Click **Run plan** in the sidebar first.")
    else:
        inputs, warnings = get_inputs_from_state()

        status = st.empty()
        progress = st.progress(0.0)

        status.caption("Step 1 of 2: running core projection...")
        projection = build_projection(inputs)
        progress.progress(0.35)

        if inputs.monte_carlo_enabled:
            status.caption("Step 2 of 2: running Monte Carlo...")
            mc_text = st.empty()
            mc = monte_carlo(inputs, progress_bar=progress, progress_text=mc_text)
            finals = mc.iloc[:, -1]
            summary = {
                "success_rate": float((finals > 0).mean()),
                "median_final_liquid": float(finals.median()),
                "p10_final_liquid": float(finals.quantile(0.10)),
            }
            mc_text.empty()
        else:
            status.caption("Step 2 of 2: Monte Carlo skipped.")
            progress.progress(1.0)
            mc = None
            final_liquid = float(projection.iloc[-1]["liquid_assets_end"])
            summary = {
                "success_rate": float("nan"),
                "median_final_liquid": final_liquid,
                "p10_final_liquid": final_liquid,
            }

        status.empty()
        progress.empty()

        retirement_row = projection[projection["age"] == inputs.retirement_age].iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current net worth", fmt_money(as_float(projection.iloc[0]["net_worth_end"], 0.0), inputs.display_currency))
        m2.metric("Liquid at retirement", fmt_money(as_float(retirement_row["liquid_assets_end"], 0.0), inputs.display_currency))
        m3.metric("Success rate", "Monte Carlo off" if not inputs.monte_carlo_enabled else fmt_pct(summary["success_rate"]))
        m4.metric("Withdrawal rate at retirement", fmt_pct(safe_div(as_float(retirement_row["net_portfolio_draw"], 0.0), as_float(retirement_row["liquid_assets_end"], 0.0))))

        if warnings:
            st.warning("Some incomplete values were defaulted safely.")
            for w in warnings:
                st.markdown(f"- {w}")

        chart = projection[["age", "liquid_assets_end", "net_worth_end"]].copy().set_index("age")
        st.line_chart(chart)

        if mc is not None:
            st.markdown("**Monte Carlo**")
            mc_chart = pd.DataFrame({"Age": mc.columns.astype(int), "Median": mc.median(axis=0).values, "10th percentile": mc.quantile(0.10, axis=0).values, "90th percentile": mc.quantile(0.90, axis=0).values}).set_index("Age")
            st.line_chart(mc_chart)

elif page == "Optimizer":
    st.subheader("Optimizer")
    st.markdown("Run Results first so you understand the baseline. Then use this page to test simple levers.")
    if not (st.session_state["auto_recalc"] or st.session_state["run_counter"] > 0):
        st.info("Click **Run plan** in the sidebar first.")
    else:
        inputs, _ = get_inputs_from_state()
        with st.spinner("Running optimizer..."):
            results, notes = optimize(inputs)
        show = results.copy()
        show["liquid_at_retirement"] = show["liquid_at_retirement"].map(lambda x: fmt_money(as_float(x, 0.0), inputs.display_currency))
        st.dataframe(show.head(12), use_container_width=True, hide_index=True)
        for note in notes:
            st.markdown(f"- {note}")

st.divider()
st.markdown("This version keeps Monte Carlo off by default, starts at 200 runs, adds progress feedback, and keeps the optimizer deterministic so it stays fast.")
