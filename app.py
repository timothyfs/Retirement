
import json
from io import StringIO
import hashlib
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


def as_float(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, str):
            s = v.strip().replace(",", "")
            return default if s == "" else float(s)
        return float(v)
    except Exception:
        return default


def as_int(v, default: int = 0) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, str):
            s = v.strip().replace(",", "")
            return default if s == "" else int(float(s))
        return int(float(v))
    except Exception:
        return default


def as_bool(v, default: bool = False) -> bool:
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


def df_from_records(records: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def clean_currency(currency: str, default: str = "EUR") -> str:
    return currency if currency in CURRENCIES else default


def _safe_col(df: pd.DataFrame, col: str, default):
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


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
        "cached_inputs_hash": None,
        "baseline_completed_hash": None,
        "cached_projection": None,
        "cached_mc": None,
        "cached_summary": None,
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

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "settings": self.settings,
                "household": self.household.to_dict("records"),
                "assets": self.assets.to_dict("records"),
                "debts": self.debts.to_dict("records"),
                "extra_income": self.extra_income.to_dict("records"),
                "expenses": self.expenses.to_dict("records"),
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.md5(payload.encode()).hexdigest()


def clean_household(records):
    df = df_from_records(records)
    if df.empty:
        return pd.DataFrame(columns=["enabled","name","current_age","retirement_age","life_expectancy","pension_age","pension_annual","pension_currency"])
    df["enabled"] = _safe_col(df, "enabled", False).apply(as_bool)
    df["name"] = _safe_col(df, "name", "").apply(str)
    for col in ["current_age", "retirement_age", "life_expectancy", "pension_age"]:
        df[col] = _safe_col(df, col, 0).apply(as_int)
    df["pension_annual"] = _safe_col(df, "pension_annual", 0.0).apply(as_float)
    df["pension_currency"] = _safe_col(df, "pension_currency", "EUR").apply(clean_currency)
    return df


def clean_assets(records):
    df = df_from_records(records)
    if df.empty:
        return pd.DataFrame(columns=["enabled","name","asset_type","currency","value","annual_return","volatility","monthly_contribution","sale_age","sale_proceeds","income_annual","income_start_age","income_end_age","inflation_linked_income"])
    df["enabled"] = _safe_col(df, "enabled", False).apply(as_bool)
    df["name"] = _safe_col(df, "name", "").apply(str)
    df["asset_type"] = _safe_col(df, "asset_type", "Other").apply(lambda x: x if x in ASSET_TYPES else "Other")
    df["currency"] = _safe_col(df, "currency", "EUR").apply(clean_currency)
    for col in ["value","annual_return","volatility","monthly_contribution","sale_proceeds","income_annual"]:
        df[col] = _safe_col(df, col, 0.0).apply(as_float)
    for col in ["sale_age","income_start_age","income_end_age"]:
        df[col] = _safe_col(df, col, 0).apply(as_int)
    df["inflation_linked_income"] = _safe_col(df, "inflation_linked_income", False).apply(as_bool)
    return df


def clean_debts(records):
    df = df_from_records(records)
    if df.empty:
        return pd.DataFrame(columns=["enabled","name","debt_type","currency","balance","interest_rate","monthly_payment","include_in_net_worth"])
    df["enabled"] = _safe_col(df, "enabled", False).apply(as_bool)
    df["name"] = _safe_col(df, "name", "").apply(str)
    df["debt_type"] = _safe_col(df, "debt_type", "Other").apply(lambda x: x if x in DEBT_TYPES else "Other")
    df["currency"] = _safe_col(df, "currency", "EUR").apply(clean_currency)
    for col in ["balance","interest_rate","monthly_payment"]:
        df[col] = _safe_col(df, col, 0.0).apply(as_float)
    df["include_in_net_worth"] = _safe_col(df, "include_in_net_worth", True).apply(as_bool)
    return df


def clean_extra_income(records):
    df = df_from_records(records)
    if df.empty:
        return pd.DataFrame(columns=["enabled","name","currency","annual_amount","start_age","end_age","inflation_linked"])
    df["enabled"] = _safe_col(df, "enabled", False).apply(as_bool)
    df["name"] = _safe_col(df, "name", "").apply(str)
    df["currency"] = _safe_col(df, "currency", "EUR").apply(clean_currency)
    df["annual_amount"] = _safe_col(df, "annual_amount", 0.0).apply(as_float)
    df["start_age"] = _safe_col(df, "start_age", 0).apply(as_int)
    df["end_age"] = _safe_col(df, "end_age", 0).apply(as_int)
    df["inflation_linked"] = _safe_col(df, "inflation_linked", False).apply(as_bool)
    return df


def clean_expenses(records):
    df = df_from_records(records)
    if df.empty:
        return pd.DataFrame(columns=["enabled","name","currency","expense_type","amount","start_age","end_age","inflation_linked"])
    df["enabled"] = _safe_col(df, "enabled", False).apply(as_bool)
    df["name"] = _safe_col(df, "name", "").apply(str)
    df["currency"] = _safe_col(df, "currency", "EUR").apply(clean_currency)
    df["expense_type"] = _safe_col(df, "expense_type", "One off").apply(lambda x: x if x in EXPENSE_TYPES else "One off")
    df["amount"] = _safe_col(df, "amount", 0.0).apply(as_float)
    df["start_age"] = _safe_col(df, "start_age", 0).apply(as_int)
    df["end_age"] = _safe_col(df, "end_age", 0).apply(as_int)
    df["inflation_linked"] = _safe_col(df, "inflation_linked", False).apply(as_bool)
    return df


def get_inputs_from_state():
    warnings: List[str] = []
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
    if (inputs.assets["name"].astype(str).str.strip() == "").any():
        warnings.append("One or more asset rows have no name.")
    return inputs, warnings


def annual_base_spending(age: int, inputs: Inputs) -> float:
    base = as_float(inputs.settings["spending_pre75"], 0.0) if age < 75 else as_float(inputs.settings["spending_post75"], 0.0)
    years = max(0, age - inputs.retirement_age)
    return base * ((1.0 + as_float(inputs.settings["inflation"], 0.025)) ** years)


def annual_healthcare(age: int, inputs: Inputs) -> float:
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


def annual_pension_income(age: int, inputs: Inputs) -> float:
    total = 0.0
    inflation = as_float(inputs.settings["inflation"], 0.025)
    for row in inputs.household.itertuples(index=False):
        if not as_bool(row.enabled, True):
            continue
        pension_age = as_int(row.pension_age, 0)
        if age < pension_age:
            continue
        years = age - pension_age
        total += as_float(row.pension_annual, 0.0) * ((1.0 + inflation) ** years)
    return total


def annual_other_income(age: int, inputs: Inputs) -> float:
    total = 0.0
    inflation = as_float(inputs.settings["inflation"], 0.025)
    for row in inputs.extra_income.itertuples(index=False):
        if not as_bool(row.enabled, True):
            continue
        start_age = as_int(row.start_age, 0)
        end_age = as_int(row.end_age, 0)
        if age < start_age or age > end_age:
            continue
        amt = as_float(row.annual_amount, 0.0)
        if as_bool(row.inflation_linked, False):
            amt *= ((1.0 + inflation) ** max(0, age - start_age))
        total += amt
    for row in inputs.assets.itertuples(index=False):
        if not as_bool(row.enabled, True):
            continue
        start_age = as_int(row.income_start_age, 0)
        end_age = as_int(row.income_end_age, 0)
        end_age = end_age if end_age > 0 else inputs.life_expectancy
        if age < start_age or age > end_age:
            continue
        amt = as_float(row.income_annual, 0.0)
        if as_bool(row.inflation_linked_income, False):
            amt *= ((1.0 + inflation) ** max(0, age - start_age))
        total += amt
    return total


def annual_event_expenses(age: int, inputs: Inputs) -> float:
    total = annual_healthcare(age, inputs)
    inflation = as_float(inputs.settings["inflation"], 0.025)
    for row in inputs.expenses.itertuples(index=False):
        if not as_bool(row.enabled, True):
            continue
        start_age = as_int(row.start_age, 0)
        end_age = as_int(row.end_age, 0)
        if age < start_age or age > end_age:
            continue
        expense_type = str(row.expense_type)
        if expense_type == "One off" and age != start_age:
            continue
        amt = as_float(row.amount, 0.0)
        if expense_type == "Monthly":
            amt *= 12.0
        if as_bool(row.inflation_linked, False):
            amt *= ((1.0 + inflation) ** max(0, age - start_age))
        total += amt
    return total


def amortize_one_year(balance: float, annual_rate: float, monthly_payment: float) -> Tuple[float, float]:
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


def build_projection(inputs: Inputs, retirement_returns=None) -> pd.DataFrame:
    assets = inputs.assets.copy()
    debts = inputs.debts.copy()
    enabled_assets = assets[assets["enabled"] == True].copy()
    enabled_debts = debts[debts["enabled"] == True].copy()

    if enabled_assets.empty:
        enabled_assets = pd.DataFrame(columns=assets.columns)

    enabled_assets["current_value"] = enabled_assets["value"].apply(as_float)
    if not enabled_debts.empty:
        enabled_debts["current_balance"] = enabled_debts["balance"].apply(as_float)

    sale_done = {str(name): False for name in enabled_assets.get("name", pd.Series(dtype=str)).tolist()}
    rows: List[dict] = []

    for age in range(inputs.current_age, inputs.life_expectancy + 1):
        retired = age >= inputs.retirement_age
        contributions = 0.0
        sale_inflow = 0.0

        liquid_mask = enabled_assets["asset_type"].astype(str).isin(["Investment", "Cash", "Other"]) if not enabled_assets.empty else pd.Series(dtype=bool)

        for idx, row in zip(enabled_assets.index, enabled_assets.itertuples(index=False)):
            value = as_float(enabled_assets.at[idx, "current_value"], 0.0)
            asset_type = str(row.asset_type)

            if not retired and asset_type in ["Investment", "Cash", "Other"]:
                contrib = as_float(row.monthly_contribution, 0.0) * 12.0
                value += contrib
                contributions += contrib

            annual_return = as_float(row.annual_return, 0.0)
            if retirement_returns is not None and retired and asset_type in ["Investment", "Cash", "Other"]:
                annual_return = as_float(retirement_returns[age - inputs.retirement_age], annual_return)

            value += value * annual_return

            sale_age = as_int(row.sale_age, 0)
            sale_proceeds = as_float(row.sale_proceeds, 0.0)
            name = str(row.name)
            if sale_age > 0 and age >= sale_age and not sale_done.get(name, False):
                if sale_proceeds > 0:
                    sale_inflow += sale_proceeds
                    if asset_type == "Property":
                        value = max(0.0, value - sale_proceeds)
                sale_done[name] = True

            enabled_assets.at[idx, "current_value"] = max(0.0, value)

        debt_paid = 0.0
        liabilities_end = 0.0
        if not enabled_debts.empty:
            for idx, row in zip(enabled_debts.index, enabled_debts.itertuples(index=False)):
                new_balance, paid = amortize_one_year(
                    as_float(enabled_debts.at[idx, "current_balance"], 0.0),
                    as_float(row.interest_rate, 0.0),
                    as_float(row.monthly_payment, 0.0),
                )
                enabled_debts.at[idx, "current_balance"] = new_balance
                debt_paid += paid
                if as_bool(row.include_in_net_worth, True):
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


def monte_carlo(inputs: Inputs, progress_bar=None, progress_text=None):
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
            fraction = 0.35 + ((i + 1) / runs) * 0.65
            progress_bar.progress(min(fraction, 1.0))
            if progress_text is not None:
                progress_text.caption(f"Monte Carlo progress: {i + 1:,} / {runs:,}")
    return pd.DataFrame(paths, columns=list(range(inputs.retirement_age, inputs.life_expectancy + 1)))


def _inputs_to_json_args(inputs: Inputs):
    return (
        json.dumps(inputs.settings, sort_keys=True, default=str),
        inputs.household.to_json(orient="records"),
        inputs.assets.to_json(orient="records"),
        inputs.debts.to_json(orient="records"),
        inputs.extra_income.to_json(orient="records"),
        inputs.expenses.to_json(orient="records"),
    )


@st.cache_data(show_spinner=False)
def _cached_snapshot(
    settings_json: str,
    household_json: str,
    assets_json: str,
    debts_json: str,
    extra_income_json: str,
    expenses_json: str,
) -> str:
    snapshot = {
        "settings": json.loads(settings_json),
        "household": json.loads(household_json),
        "assets": json.loads(assets_json),
        "debts": json.loads(debts_json),
        "extra_income": json.loads(extra_income_json),
        "expenses": json.loads(expenses_json),
    }
    return json.dumps(snapshot, sort_keys=True, indent=2)


def get_snapshot_json() -> str:
    return _cached_snapshot(
        json.dumps(st.session_state["settings"], sort_keys=True, default=str),
        json.dumps(st.session_state["household_records"], default=str),
        json.dumps(st.session_state["asset_records"], default=str),
        json.dumps(st.session_state["debt_records"], default=str),
        json.dumps(st.session_state["extra_income_records"], default=str),
        json.dumps(st.session_state["expense_records"], default=str),
    )


@st.cache_data(show_spinner=False)
def _cached_projection(
    settings_json: str,
    household_json: str,
    assets_json: str,
    debts_json: str,
    extra_income_json: str,
    expenses_json: str,
) -> pd.DataFrame:
    settings = json.loads(settings_json)
    inputs = Inputs(
        settings,
        pd.read_json(StringIO(household_json), orient="records"),
        pd.read_json(StringIO(assets_json), orient="records"),
        pd.read_json(StringIO(debts_json), orient="records"),
        pd.read_json(StringIO(extra_income_json), orient="records"),
        pd.read_json(StringIO(expenses_json), orient="records"),
    )
    return build_projection(inputs)


@st.cache_data(show_spinner=False)
def _cached_optimize(
    settings_json: str,
    household_json: str,
    assets_json: str,
    debts_json: str,
    extra_income_json: str,
    expenses_json: str,
) -> Tuple[pd.DataFrame, List[str]]:
    settings = json.loads(settings_json)
    inputs = Inputs(
        settings,
        pd.read_json(StringIO(household_json), orient="records"),
        pd.read_json(StringIO(assets_json), orient="records"),
        pd.read_json(StringIO(debts_json), orient="records"),
        pd.read_json(StringIO(extra_income_json), orient="records"),
        pd.read_json(StringIO(expenses_json), orient="records"),
    )
    return _run_optimize(inputs)


def _run_optimize(inputs: Inputs):
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
        "Run Results first. Then use Optimizer to see which simple lever improves the outcome most.",
    ]
    return result, notes


def invalidate_result_cache() -> None:
    st.session_state["cached_inputs_hash"] = None
    st.session_state["cached_projection"] = None
    st.session_state["cached_mc"] = None
    st.session_state["cached_summary"] = None


# Sidebar

def assumptions_summary(inputs: Inputs) -> pd.DataFrame:
    enabled_household = inputs.household[inputs.household["enabled"] == True]
    enabled_assets = inputs.assets[inputs.assets["enabled"] == True]
    enabled_debts = inputs.debts[inputs.debts["enabled"] == True]
    enabled_income = inputs.extra_income[inputs.extra_income["enabled"] == True]
    enabled_expenses = inputs.expenses[inputs.expenses["enabled"] == True]

    total_assets = float(enabled_assets["value"].apply(as_float).sum()) if not enabled_assets.empty else 0.0
    total_debts = float(enabled_debts["balance"].apply(as_float).sum()) if not enabled_debts.empty else 0.0
    total_contrib = float(enabled_assets["monthly_contribution"].apply(as_float).sum()) * 12.0 if not enabled_assets.empty else 0.0
    total_other_income = float(enabled_income["annual_amount"].apply(as_float).sum()) if not enabled_income.empty else 0.0
    total_event_expenses = float(enabled_expenses["amount"].apply(as_float).sum()) if not enabled_expenses.empty else 0.0

    rows = [
        ("Scenario", str(inputs.settings.get("scenario_name", "Base"))),
        ("People included", str(int(enabled_household.shape[0]))),
        ("Current age used", str(inputs.current_age)),
        ("Retirement age used", str(inputs.retirement_age)),
        ("Life expectancy used", str(inputs.life_expectancy)),
        ("Annual spending before 75", fmt_money(as_float(inputs.settings.get("spending_pre75", 0.0)), inputs.display_currency)),
        ("Annual spending after 75", fmt_money(as_float(inputs.settings.get("spending_post75", 0.0)), inputs.display_currency)),
        ("Inflation", fmt_pct(as_float(inputs.settings.get("inflation", 0.025)))),
        ("Healthcare ramp", "On" if as_bool(inputs.settings.get("healthcare_enabled", False)) else "Off"),
        ("Healthcare annual cost", fmt_money(as_float(inputs.settings.get("healthcare_annual", 0.0)), inputs.display_currency)),
        ("Emergency cash floor", f"{as_float(inputs.settings.get('emergency_cash_years', 0.0)):.1f} years"),
        ("Legacy target", fmt_money(as_float(inputs.settings.get("legacy_target", 0.0)), inputs.display_currency)),
        ("Total assets entered", fmt_money(total_assets, inputs.display_currency)),
        ("Total debts entered", fmt_money(total_debts, inputs.display_currency)),
        ("Annual contributions", fmt_money(total_contrib, inputs.display_currency)),
        ("Other annual income", fmt_money(total_other_income, inputs.display_currency)),
        ("Enabled extra expense rows", str(int(enabled_expenses.shape[0]))),
        ("Monte Carlo", "On" if inputs.monte_carlo_enabled else "Off"),
        ("Monte Carlo runs", str(inputs.mc_runs if inputs.monte_carlo_enabled else 0)),
    ]
    return pd.DataFrame(rows, columns=["Assumption", "Value"])


def input_explainers():
    st.markdown("### What each input section means")
    with st.expander("Household"):
        st.write("Add the people included in the plan. Current age, retirement age, life expectancy, and pension age are the dates the model uses for the timeline. Pension annual is the yearly state pension or other guaranteed pension you expect.")
    with st.expander("Assets"):
        st.write("Add savings, portfolios, property, and cash here. Monthly contribution is what you are still adding before retirement. Sale age and sale proceeds are optional and only matter if you plan to sell an asset later.")
    with st.expander("Debts"):
        st.write("Add mortgages and loans here. Monthly payment is what you expect to pay. The model reduces the debt balance each year using the interest rate and payment you enter.")
    with st.expander("Extra income"):
        st.write("Use this for consulting, rent, part time work, or any other income stream that starts and stops at certain ages.")
    with st.expander("Large expenses"):
        st.write("Use this for weddings, cars, school fees, holidays, or major one off spending. Choose One off, Monthly, or Annual depending on how the cost happens.")
    with st.expander("Advanced assumptions"):
        st.write("These are the plan-wide assumptions. Spending before and after 75 are your baseline retirement spending targets. Inflation increases future spending. Healthcare adds an extra late-life cost ramp. Monte Carlo is optional and stress tests the plan with random return paths.")
    with st.expander("Results"):
        st.write("Results show the baseline plan outcome. Run this first before using Optimizer. If Monte Carlo is off, the results are deterministic only.")
    with st.expander("Optimizer"):
        st.write("Optimizer answers a simple question in plain English: what happens if you work longer? It does not run Monte Carlo. It compares straightforward scenarios like working 1, 2, or more extra years and shows which one improves liquid assets at retirement the most.")


def current_balance_sheet(inputs: Inputs) -> pd.DataFrame:
    assets = inputs.assets[inputs.assets["enabled"] == True].copy()
    debts = inputs.debts[inputs.debts["enabled"] == True].copy()
    liquid_now = float(assets.loc[assets["asset_type"].isin(["Investment", "Cash", "Other"]), "value"].apply(as_float).sum()) if not assets.empty else 0.0
    property_now = float(assets.loc[assets["asset_type"] == "Property", "value"].apply(as_float).sum()) if not assets.empty else 0.0
    debts_now = float(debts["balance"].apply(as_float).sum()) if not debts.empty else 0.0
    net_now = liquid_now + property_now - debts_now
    return pd.DataFrame([
        ("Liquid assets today", liquid_now),
        ("Property value today", property_now),
        ("Debt today", debts_now),
        ("Net worth today", net_now),
    ], columns=["Item", "Value"])


def build_projection_with_sale_lines(inputs: Inputs, retirement_returns=None):
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
        property_mask = enabled_assets["asset_type"].astype(str).isin(["Property"]) if not enabled_assets.empty else pd.Series(dtype=bool)

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

        if sale_inflow > 0 and not enabled_assets.empty:
            cash_candidates = enabled_assets.index[enabled_assets["asset_type"] == "Cash"].tolist()
            if cash_candidates:
                idx = cash_candidates[0]
                enabled_assets.at[idx, "current_value"] = as_float(enabled_assets.at[idx, "current_value"], 0.0) + sale_inflow
            else:
                # create a synthetic cash bucket so property sales visibly become cash
                new_row = {col: None for col in enabled_assets.columns}
                new_row["enabled"] = True
                new_row["name"] = "Sale proceeds cash"
                new_row["asset_type"] = "Cash"
                new_row["currency"] = inputs.display_currency
                new_row["current_value"] = sale_inflow
                new_row["value"] = 0.0
                new_row["annual_return"] = 0.0
                new_row["volatility"] = 0.0
                new_row["monthly_contribution"] = 0.0
                new_row["sale_age"] = 0
                new_row["sale_proceeds"] = 0.0
                new_row["income_annual"] = 0.0
                new_row["income_start_age"] = 0
                new_row["income_end_age"] = 0
                new_row["inflation_linked_income"] = False
                enabled_assets = pd.concat([enabled_assets, pd.DataFrame([new_row])], ignore_index=True)
                liquid_mask = enabled_assets["asset_type"].astype(str).isin(["Investment", "Cash", "Other"])
                property_mask = enabled_assets["asset_type"].astype(str).isin(["Property"])

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

        liquid_total_pre_draw = float(enabled_assets.loc[liquid_mask, "current_value"].sum()) if not enabled_assets.empty and liquid_mask.any() else 0.0
        property_total = float(enabled_assets.loc[property_mask, "current_value"].sum()) if not enabled_assets.empty and property_mask.any() else 0.0
        reserve_floor = as_float(inputs.settings["emergency_cash_years"], 2.0) * (base_spending + annual_healthcare(age, inputs))
        net_need = max(0.0, total_spending - pensions - other_income)
        available_draw = max(0.0, liquid_total_pre_draw - reserve_floor)
        draw = min(available_draw, net_need)

        if not enabled_assets.empty and liquid_mask.any() and draw > 0:
            liquid_before = float(enabled_assets.loc[liquid_mask, "current_value"].sum())
            if liquid_before > 0:
                ratio = max(0.0, (liquid_before - draw) / liquid_before)
                enabled_assets.loc[liquid_mask, "current_value"] = enabled_assets.loc[liquid_mask, "current_value"] * ratio

        liquid_end = float(enabled_assets.loc[liquid_mask, "current_value"].sum()) if not enabled_assets.empty and liquid_mask.any() else 0.0
        property_end = float(enabled_assets.loc[property_mask, "current_value"].sum()) if not enabled_assets.empty and property_mask.any() else 0.0
        net_worth = liquid_end + property_end - liabilities_end

        rows.append({
            "age": age,
            "phase": "Retirement" if retired else "Pre retirement",
            "liquid_assets_end": liquid_end,
            "property_assets_end": property_end,
            "sale_inflow": sale_inflow,
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
        invalidate_result_cache()

    st.divider()
    st.header("Export")
    st.download_button(
        label="Download snapshot (JSON)",
        data=get_snapshot_json(),
        file_name=f"retirement_plan_{settings['scenario_name']}.json",
        mime="application/json",
        use_container_width=True,
    )


# Main
page = st.session_state["page"]
st.title("Retirement Planner")
st.caption("Stable v1. Faster by default. Monte Carlo runs only when you explicitly enable it.")

if page == "Basic Inputs":
    st.subheader("Basic inputs")
    st.markdown("Enter the core facts of the plan. Keep this simple first.")
    input_explainers()
    st.info("These tables are for the main financial facts only. Advanced assumptions live on the next page.")
    st.markdown("**Household**")
    edited = st.data_editor(df_from_records(st.session_state["household_records"]), num_rows="dynamic", use_container_width=True, key="household_editor_v2")
    st.session_state["household_records"] = edited.to_dict("records")
    st.markdown("**Assets**")
    edited = st.data_editor(df_from_records(st.session_state["asset_records"]), num_rows="dynamic", use_container_width=True, key="asset_editor_v2")
    st.session_state["asset_records"] = edited.to_dict("records")
    st.markdown("**Debts**")
    edited = st.data_editor(df_from_records(st.session_state["debt_records"]), num_rows="dynamic", use_container_width=True, key="debt_editor_v2")
    st.session_state["debt_records"] = edited.to_dict("records")
    st.markdown("**Extra income**")
    edited = st.data_editor(df_from_records(st.session_state["extra_income_records"]), num_rows="dynamic", use_container_width=True, key="income_editor_v2")
    st.session_state["extra_income_records"] = edited.to_dict("records")
    st.markdown("**Large expenses**")
    edited = st.data_editor(df_from_records(st.session_state["expense_records"]), num_rows="dynamic", use_container_width=True, key="expense_editor_v2")
    st.session_state["expense_records"] = edited.to_dict("records")

elif page == "Advanced Assumptions":
    st.subheader("Advanced assumptions")
    st.markdown("Use number inputs instead of sliders so values are precise and stable.")
    st.info("These assumptions control how hard the model pushes your plan over time. Higher inflation and higher spending make retirement harder. Monte Carlo is optional and should usually stay off while you are still editing.")
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
    c12.warning("Higher run counts will slow the plan down.")
    c13, c14 = st.columns(2)
    s["random_seed"] = c13.number_input("Random seed", min_value=0, max_value=99999, value=as_int(s.get("random_seed", 42), 42), step=1)
    s["optimizer_max_extra_years"] = c14.number_input("Max extra work years to test", min_value=1, max_value=20, value=as_int(s["optimizer_max_extra_years"], 10), step=1)
    st.session_state["settings"] = s

elif page == "Results":
    st.subheader("Results")
    st.markdown("This page runs the baseline plan. The optimizer uses this as the starting point.")
    if not (st.session_state["auto_recalc"] or st.session_state["run_counter"] > 0):
        st.info("Click **Run plan** in the sidebar first.")
    else:
        inputs, warnings = get_inputs_from_state()

        status = st.empty()
        progress = st.progress(0.0)

        status.caption("Step 1 of 2: running core projection...")
        projection = build_projection_with_sale_lines(inputs)
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

        today_df = current_balance_sheet(inputs)
        retirement_row = projection[projection["age"] == inputs.retirement_age].iloc[0]
        prev_retirement_row = projection[projection["age"] == max(inputs.current_age, inputs.retirement_age - 1)].iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Net worth today", fmt_money(as_float(today_df.loc[today_df["Item"] == "Net worth today", "Value"].iloc[0], 0.0), inputs.display_currency))
        m2.metric("Liquid assets at retirement start", fmt_money(as_float(prev_retirement_row["liquid_assets_end"], 0.0), inputs.display_currency))
        m3.metric("Success rate", "Monte Carlo off" if not inputs.monte_carlo_enabled else fmt_pct(summary["success_rate"]))
        m4.metric("Withdrawal rate in first retirement year", fmt_pct(safe_div(as_float(retirement_row["net_portfolio_draw"], 0.0), max(as_float(prev_retirement_row["liquid_assets_end"], 0.0), 1e-9))))

        m5, m6, m7 = st.columns(3)
        m5.metric("Liquid assets at end of first retirement year", fmt_money(as_float(retirement_row["liquid_assets_end"], 0.0), inputs.display_currency))
        m6.metric("Median final liquid", fmt_money(as_float(summary["median_final_liquid"], 0.0), inputs.display_currency))
        m7.metric("10th percentile final liquid", fmt_money(as_float(summary["p10_final_liquid"], 0.0), inputs.display_currency))

        st.markdown("### Run settings used")
        st.dataframe(assumptions_summary(inputs), use_container_width=True, hide_index=True)

        if inputs.monte_carlo_enabled:
            st.success(f"Monte Carlo was ON for this run with {inputs.mc_runs:,} simulations.")
        else:
            st.info("Monte Carlo was OFF for this run. These results use the deterministic projection only.")

        if warnings:
            st.warning("Some incomplete values were defaulted safely.")
            for w in warnings:
                st.markdown(f"- {w}")

        st.markdown("### Balance sheet today")
        today_display = today_df.copy()
        today_display["Value"] = today_display["Value"].map(lambda x: fmt_money(as_float(x, 0.0), inputs.display_currency))
        st.dataframe(today_display, use_container_width=True, hide_index=True)

        st.markdown("### Projection")
        chart = projection[["age", "liquid_assets_end", "property_assets_end", "net_worth_end"]].copy().set_index("age")
        st.line_chart(chart)

        sale_rows = projection[projection["sale_inflow"] > 0][["age", "sale_inflow"]].copy()
        if not sale_rows.empty:
            st.markdown("### Property sales converted to cash")
            sale_rows["sale_inflow"] = sale_rows["sale_inflow"].map(lambda x: fmt_money(as_float(x, 0.0), inputs.display_currency))
            sale_rows.columns = ["Age", "Sale proceeds moved into liquid assets"]
            st.dataframe(sale_rows, use_container_width=True, hide_index=True)

        if mc is not None:
            st.markdown("### Monte Carlo")
            mc_chart = pd.DataFrame({"Age": mc.columns.astype(int), "Median": mc.median(axis=0).values, "10th percentile": mc.quantile(0.10, axis=0).values, "90th percentile": mc.quantile(0.90, axis=0).values}).set_index("Age")
            st.line_chart(mc_chart)

elif page == "Optimizer":
    st.subheader("Optimizer")
    st.markdown("Run Results first so you understand the baseline. Then use this page to test simple levers.")
    st.info("Plain English explanation: the Optimizer checks simple versions of one idea only — working longer. It shows how much more liquid money you may have at retirement if you work 1, 2, 3, or more extra years. It is fast because it does not run Monte Carlo.")
    inputs, _ = get_inputs_from_state()
    current_hash = inputs.fingerprint()

    if st.session_state.get("baseline_completed_hash") != current_hash:
        st.info("Run **Results** for the current inputs first. The optimizer is only enabled after the baseline plan is complete.")
    else:
        with st.spinner("Running optimizer..."):
            results, notes = _cached_optimize(*_inputs_to_json_args(inputs))
        show = results.copy()
        show["liquid_at_retirement"] = show["liquid_at_retirement"].map(lambda x: fmt_money(as_float(x, 0.0), inputs.display_currency))
        st.dataframe(show.head(12), use_container_width=True, hide_index=True)
        for note in notes:
            st.markdown(f"- {note}")

st.divider()
st.markdown("Updated version: safer column handling, faster row iteration, cached deterministic projection, clearer gating between Results and Optimizer, and better progress feedback.")
