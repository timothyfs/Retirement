
import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import Dict, Tuple

st.set_page_config(page_title="Retirement Planner", page_icon="📈", layout="wide")

CURRENCIES: Dict[str, Dict[str, str]] = {
    "EUR": {"symbol": "€", "name": "Euro"},
    "GBP": {"symbol": "£", "name": "British Pound"},
    "USD": {"symbol": "$", "name": "US Dollar"},
    "CHF": {"symbol": "CHF ", "name": "Swiss Franc"},
}

PAGES = ["Welcome", "Household", "Assets", "Debt", "Income", "Expenses", "FX & Tax", "Settings", "Results", "Optimizer"]


def fmt_money(value: float, currency: str) -> str:
    return f"{CURRENCIES.get(currency, CURRENCIES['EUR'])['symbol']}{value:,.0f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}"


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def as_bool(v, default=False) -> bool:
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"true", "yes", "1", "y"}:
        return True
    if s in {"false", "no", "0", "n", ""}:
        return False
    return default


def as_float(v, default=0.0) -> float:
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    try:
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
        if pd.isna(v):
            return default
    except Exception:
        pass
    try:
        if isinstance(v, str):
            s = v.strip().replace(",", "")
            if s == "":
                return default
            return int(float(s))
        return int(float(v))
    except Exception:
        return default


def as_str(v, default="") -> str:
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    s = str(v).strip()
    return s if s else default


def to_records(df: pd.DataFrame) -> list[dict]:
    return df.to_dict("records")


def from_records(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


DEFAULT_SETTINGS = {
    "scenario_name": "Base",
    "display_currency": "EUR",
    "base_currency": "EUR",
    "inflation": 0.025,
    "stress_uplift": 0.00,
    "base_spending_pre75": 100_000.0,
    "base_spending_post75": 80_000.0,
    "target_success_rate": 0.85,
    "mc_runs": 1000,
    "random_seed": 42,
    "healthcare_enabled": True,
    "healthcare_start_age": 75,
    "healthcare_base_annual": 8_000.0,
    "healthcare_inflation_extra": 0.02,
    "legacy_target": 250_000.0,
    "emergency_cash_years": 2.0,
    "tax_enabled": True,
    "optimizer_max_extra_work_years": 10,
}

DEFAULT_HOUSEHOLD = pd.DataFrame([
    {"enabled": True, "name": "You", "current_age": 53, "retirement_age": 55, "life_expectancy": 95, "pension_age": 67, "pension_annual": 14000.0, "pension_currency": "GBP"},
    {"enabled": True, "name": "Wife", "current_age": 51, "retirement_age": 55, "life_expectancy": 95, "pension_age": 67, "pension_annual": 13000.0, "pension_currency": "GBP"},
])

DEFAULT_ASSETS = pd.DataFrame([
    {"enabled": True, "name": "Investment portfolio", "category": "investment", "account_type": "taxable", "currency": "EUR", "value": 1_070_000.0, "annual_return": 0.060, "volatility": 0.140, "monthly_contribution": 5_000.0, "sale_age": 0, "sale_proceeds": 0.0, "income_annual": 0.0, "income_start_age": 0, "income_end_age": 0, "inflation_linked_income": False},
    {"enabled": True, "name": "Main property", "category": "property", "account_type": "property", "currency": "EUR", "value": 1_020_000.0, "annual_return": 0.025, "volatility": 0.070, "monthly_contribution": 0.0, "sale_age": 75, "sale_proceeds": 500_000.0, "income_annual": 0.0, "income_start_age": 0, "income_end_age": 0, "inflation_linked_income": False},
    {"enabled": True, "name": "Rental property", "category": "property", "account_type": "property", "currency": "EUR", "value": 0.0, "annual_return": 0.025, "volatility": 0.070, "monthly_contribution": 0.0, "sale_age": 0, "sale_proceeds": 0.0, "income_annual": 24_000.0, "income_start_age": 55, "income_end_age": 95, "inflation_linked_income": False},
    {"enabled": True, "name": "Emergency cash", "category": "cash", "account_type": "cash", "currency": "EUR", "value": 50_000.0, "annual_return": 0.020, "volatility": 0.010, "monthly_contribution": 0.0, "sale_age": 0, "sale_proceeds": 0.0, "income_annual": 0.0, "income_start_age": 0, "income_end_age": 0, "inflation_linked_income": False},
])

DEFAULT_DEBT = pd.DataFrame([
    {"enabled": True, "name": "Main mortgage", "linked_asset": "Main property", "currency": "EUR", "balance": 50_000.0, "interest_rate": 0.035, "monthly_payment": 1_200.0, "include_in_net_worth": True}
])

DEFAULT_INCOME = pd.DataFrame([
    {"enabled": True, "name": "Consulting", "currency": "EUR", "annual_amount": 10_000.0, "start_age": 55, "end_age": 60, "inflation_linked": False}
])

DEFAULT_EXPENSES = pd.DataFrame([
    {"enabled": False, "name": "Car purchase", "currency": "EUR", "mode": "one_off", "amount": 50_000.0, "start_age": 56, "end_age": 56, "inflation_linked": False}
])

DEFAULT_FX = pd.DataFrame([
    {"currency": "EUR", "to_base": 1.00},
    {"currency": "GBP", "to_base": 1.17},
    {"currency": "USD", "to_base": 0.92},
    {"currency": "CHF", "to_base": 1.04},
])

DEFAULT_TAX = pd.DataFrame([
    {"account_type": "taxable", "withdrawal_tax_rate": 0.20, "growth_tax_drag": 0.00},
    {"account_type": "pension", "withdrawal_tax_rate": 0.15, "growth_tax_drag": 0.00},
    {"account_type": "cash", "withdrawal_tax_rate": 0.00, "growth_tax_drag": 0.00},
    {"account_type": "property", "withdrawal_tax_rate": 0.10, "growth_tax_drag": 0.00},
    {"account_type": "other", "withdrawal_tax_rate": 0.10, "growth_tax_drag": 0.00},
    {"account_type": "isa", "withdrawal_tax_rate": 0.00, "growth_tax_drag": 0.00},
])


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
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


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
        return 55 if h.empty else as_int(h["current_age"].max(), 55)

    @property
    def retirement_age(self) -> int:
        h = self.household[self.household["enabled"] == True]
        return 55 if h.empty else as_int(h["retirement_age"].max(), 55)

    @property
    def life_expectancy(self) -> int:
        h = self.household[self.household["enabled"] == True]
        return 95 if h.empty else as_int(h["life_expectancy"].max(), 95)

    @property
    def mc_runs(self) -> int:
        return as_int(self.settings["mc_runs"], 1000)

    @property
    def random_seed(self) -> int:
        return as_int(self.settings["random_seed"], 42)


def clean_df(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df

    if kind == "household":
        df["enabled"] = df.get("enabled", False).apply(as_bool)
        for c in ["name", "pension_currency"]:
            df[c] = df.get(c, "").apply(as_str)
        for c in ["current_age", "retirement_age", "life_expectancy", "pension_age"]:
            df[c] = df.get(c, 0).apply(as_int)
        df["pension_annual"] = df.get("pension_annual", 0.0).apply(as_float)

    elif kind == "assets":
        for c in ["enabled", "inflation_linked_income"]:
            df[c] = df.get(c, False).apply(as_bool)
        for c in ["name", "category", "account_type", "currency"]:
            df[c] = df.get(c, "").apply(as_str)
        for c in ["sale_age", "income_start_age", "income_end_age"]:
            df[c] = df.get(c, 0).apply(as_int)
        for c in ["value", "annual_return", "volatility", "monthly_contribution", "sale_proceeds", "income_annual"]:
            df[c] = df.get(c, 0.0).apply(as_float)

    elif kind == "debt":
        for c in ["enabled", "include_in_net_worth"]:
            df[c] = df.get(c, False).apply(as_bool)
        for c in ["name", "linked_asset", "currency"]:
            df[c] = df.get(c, "").apply(as_str)
        for c in ["balance", "interest_rate", "monthly_payment"]:
            df[c] = df.get(c, 0.0).apply(as_float)

    elif kind == "income":
        for c in ["enabled", "inflation_linked"]:
            df[c] = df.get(c, False).apply(as_bool)
        for c in ["name", "currency"]:
            df[c] = df.get(c, "").apply(as_str)
        for c in ["start_age", "end_age"]:
            df[c] = df.get(c, 0).apply(as_int)
        df["annual_amount"] = df.get("annual_amount", 0.0).apply(as_float)

    elif kind == "expenses":
        for c in ["enabled", "inflation_linked"]:
            df[c] = df.get(c, False).apply(as_bool)
        for c in ["name", "currency", "mode"]:
            df[c] = df.get(c, "").apply(as_str)
        for c in ["start_age", "end_age"]:
            df[c] = df.get(c, 0).apply(as_int)
        df["amount"] = df.get("amount", 0.0).apply(as_float)

    elif kind == "fx":
        df["currency"] = df.get("currency", "").apply(as_str)
        df["to_base"] = df.get("to_base", 1.0).apply(lambda x: max(as_float(x, 1.0), 0.000001))

    elif kind == "tax":
        df["account_type"] = df.get("account_type", "").apply(as_str)
        df["withdrawal_tax_rate"] = df.get("withdrawal_tax_rate", 0.0).apply(as_float)
        df["growth_tax_drag"] = df.get("growth_tax_drag", 0.0).apply(as_float)

    return df


def get_inputs() -> Inputs:
    return Inputs(
        settings=dict(st.session_state["settings"]),
        household=clean_df(from_records(st.session_state["household_records"]), "household"),
        assets=clean_df(from_records(st.session_state["asset_records"]), "assets"),
        debt=clean_df(from_records(st.session_state["debt_records"]), "debt"),
        income=clean_df(from_records(st.session_state["income_records"]), "income"),
        expenses=clean_df(from_records(st.session_state["expense_records"]), "expenses"),
        fx=clean_df(from_records(st.session_state["fx_records"]), "fx"),
        tax=clean_df(from_records(st.session_state["tax_records"]), "tax"),
    )


def fx_rate(inputs: Inputs, currency: str) -> float:
    row = inputs.fx[inputs.fx["currency"] == currency]
    return 1.0 if row.empty else as_float(row.iloc[0]["to_base"], 1.0)


def to_base(inputs: Inputs, amount: float, currency: str) -> float:
    return as_float(amount, 0.0) * fx_rate(inputs, currency)


def from_base(inputs: Inputs, amount: float, currency: str) -> float:
    rate = fx_rate(inputs, currency)
    return as_float(amount, 0.0) / rate if rate else as_float(amount, 0.0)


def tax_rates(inputs: Inputs, account_type: str) -> Tuple[float, float]:
    if not as_bool(inputs.settings.get("tax_enabled", True), True):
        return 0.0, 0.0
    row = inputs.tax[inputs.tax["account_type"] == account_type]
    if row.empty:
        return 0.0, 0.0
    return as_float(row.iloc[0]["withdrawal_tax_rate"], 0.0), as_float(row.iloc[0]["growth_tax_drag"], 0.0)


def annual_base_spending(age: int, inputs: Inputs) -> float:
    base = as_float(inputs.settings["base_spending_pre75"], 0.0) if age < 75 else as_float(inputs.settings["base_spending_post75"], 0.0)
    base *= (1.0 + as_float(inputs.settings["stress_uplift"], 0.0))
    years = max(0, age - inputs.retirement_age)
    return base * ((1.0 + as_float(inputs.settings["inflation"], 0.025)) ** years)


def annual_healthcare(age: int, inputs: Inputs) -> float:
    if not as_bool(inputs.settings["healthcare_enabled"], True) or age < as_int(inputs.settings["healthcare_start_age"], 75):
        return 0.0
    years = age - as_int(inputs.settings["healthcare_start_age"], 75)
    rate = as_float(inputs.settings["inflation"], 0.025) + as_float(inputs.settings["healthcare_inflation_extra"], 0.02)
    return as_float(inputs.settings["healthcare_base_annual"], 0.0) * ((1.0 + rate) ** years)


def annual_expenses(age: int, inputs: Inputs) -> float:
    total = annual_healthcare(age, inputs)
    for _, row in inputs.expenses.iterrows():
        if not as_bool(row.get("enabled", True), True):
            continue
        start_age = as_int(row.get("start_age", 0), 0)
        end_age = as_int(row.get("end_age", 0), 0)
        if age < start_age or age > end_age:
            continue
        mode = as_str(row.get("mode", "one_off"), "one_off")
        if mode == "one_off" and age != start_age:
            continue
        amt = as_float(row.get("amount", 0.0), 0.0)
        if mode == "monthly":
            amt *= 12.0
        if as_bool(row.get("inflation_linked", False), False):
            amt *= (1.0 + as_float(inputs.settings["inflation"], 0.025)) ** max(0, age - start_age)
        total += to_base(inputs, amt, as_str(row.get("currency", inputs.base_currency), inputs.base_currency))
    return total


def annual_pensions(age: int, inputs: Inputs) -> float:
    total = 0.0
    for _, row in inputs.household.iterrows():
        if not as_bool(row.get("enabled", True), True):
            continue
        pension_age = as_int(row.get("pension_age", 0), 0)
        if age < pension_age:
            continue
        amt = as_float(row.get("pension_annual", 0.0), 0.0) * ((1.0 + as_float(inputs.settings["inflation"], 0.025)) ** max(0, age - pension_age))
        total += to_base(inputs, amt, as_str(row.get("pension_currency", inputs.base_currency), inputs.base_currency))
    return total


def annual_other_income(age: int, inputs: Inputs) -> float:
    total = 0.0
    for _, row in inputs.income.iterrows():
        if not as_bool(row.get("enabled", True), True):
            continue
        start_age = as_int(row.get("start_age", 0), 0)
        end_age = as_int(row.get("end_age", 0), 0)
        if age < start_age or age > end_age:
            continue
        amt = as_float(row.get("annual_amount", 0.0), 0.0)
        if as_bool(row.get("inflation_linked", False), False):
            amt *= (1.0 + as_float(inputs.settings["inflation"], 0.025)) ** max(0, age - start_age)
        total += to_base(inputs, amt, as_str(row.get("currency", inputs.base_currency), inputs.base_currency))
    for _, row in inputs.assets.iterrows():
        if not as_bool(row.get("enabled", True), True):
            continue
        start_age = as_int(row.get("income_start_age", 0), 0)
        end_age = as_int(row.get("income_end_age", 0), 0)
        end_age = end_age if end_age > 0 else inputs.life_expectancy
        if age < start_age or age > end_age:
            continue
        amt = as_float(row.get("income_annual", 0.0), 0.0)
        if as_bool(row.get("inflation_linked_income", False), False):
            amt *= (1.0 + as_float(inputs.settings["inflation"], 0.025)) ** max(0, age - start_age)
        total += to_base(inputs, amt, as_str(row.get("currency", inputs.base_currency), inputs.base_currency))
    return total


def amortize_one_year(balance: float, annual_rate: float, monthly_payment: float) -> Tuple[float, float]:
    bal = as_float(balance, 0.0)
    paid = 0.0
    for _ in range(12):
        if bal <= 0:
            break
        bal += bal * (as_float(annual_rate, 0.0) / 12.0)
        payment = min(as_float(monthly_payment, 0.0), bal) if as_float(monthly_payment, 0.0) > 0 else 0.0
        bal -= payment
        paid += payment
    return max(0.0, bal), paid


def build_projection(inputs: Inputs, retirement_returns: np.ndarray | None = None) -> pd.DataFrame:
    assets = inputs.assets.copy()
    debt = inputs.debt.copy()
    assets = assets[assets["enabled"] == True].copy() if "enabled" in assets.columns else assets.copy()
    debt = debt[debt["enabled"] == True].copy() if "enabled" in debt.columns else debt.copy()

    if assets.empty:
        assets = pd.DataFrame(columns=["category", "currency", "value", "name", "account_type", "annual_return", "monthly_contribution", "sale_age", "sale_proceeds", "income_annual", "income_start_age", "income_end_age", "inflation_linked_income"])

    assets["base_value"] = [to_base(inputs, as_float(v, 0.0), as_str(c, inputs.base_currency)) for v, c in zip(assets.get("value", []), assets.get("currency", []))]
    debt["base_balance"] = [to_base(inputs, as_float(v, 0.0), as_str(c, inputs.base_currency)) for v, c in zip(debt.get("balance", []), debt.get("currency", []))]
    rows = []
    sale_done = {as_str(name, ""): False for name in assets.get("name", pd.Series(dtype=str)).tolist()}

    for age in range(inputs.current_age, inputs.life_expectancy + 1):
        retired = age >= inputs.retirement_age
        contributions = 0.0
        liquid_growth = 0.0
        sale_inflow = 0.0
        liquid_mask = assets["category"].astype(str).isin(["investment", "cash", "other"]) if not assets.empty else pd.Series(dtype=bool)

        for idx, row in assets.iterrows():
            value = as_float(assets.at[idx, "base_value"], 0.0)
            category = as_str(row.get("category", "other"), "other")
            _, growth_tax_drag = tax_rates(inputs, as_str(row.get("account_type", "other"), "other"))

            if not retired and category in ["investment", "cash", "other"]:
                contrib = to_base(inputs, as_float(row.get("monthly_contribution", 0.0), 0.0) * 12.0, as_str(row.get("currency", inputs.base_currency), inputs.base_currency))
                value += contrib
                contributions += contrib

            annual_return = as_float(row.get("annual_return", 0.0), 0.0) - growth_tax_drag
            if retirement_returns is not None and retired and category in ["investment", "cash", "other"]:
                annual_return = as_float(retirement_returns[age - inputs.retirement_age], annual_return)

            growth = value * annual_return
            value += growth
            if category in ["investment", "cash", "other"]:
                liquid_growth += growth

            sale_age = as_int(row.get("sale_age", 0), 0)
            sale_proceeds = to_base(inputs, as_float(row.get("sale_proceeds", 0.0), 0.0), as_str(row.get("currency", inputs.base_currency), inputs.base_currency))
            name = as_str(row.get("name", ""), "")
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
            bal = as_float(debt.at[idx, "base_balance"], 0.0)
            new_bal, paid = amortize_one_year(bal, as_float(row.get("interest_rate", 0.0), 0.0), to_base(inputs, as_float(row.get("monthly_payment", 0.0), 0.0), as_str(row.get("currency", inputs.base_currency), inputs.base_currency)))
            debt.at[idx, "base_balance"] = new_bal
            debt_paid += paid
            if as_bool(row.get("include_in_net_worth", True), True):
                liabilities_end += new_bal

        pensions = annual_pensions(age, inputs)
        other_income = annual_other_income(age, inputs)
        base_spending = annual_base_spending(age, inputs) if retired else 0.0
        life_expenses = annual_expenses(age, inputs)
        total_spending = base_spending + debt_paid + life_expenses

        liquid_total = float(assets.loc[liquid_mask, "base_value"].sum()) if not assets.empty and liquid_mask.any() else 0.0
        liquid_total += sale_inflow
        reserve_floor = as_float(inputs.settings["emergency_cash_years"], 2.0) * (base_spending + annual_healthcare(age, inputs))
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

        rows.append({
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
        })
    return pd.DataFrame(rows)


def monte_carlo(inputs: Inputs) -> pd.DataFrame:
    rng = np.random.default_rng(inputs.random_seed)
    assets = inputs.assets.copy()
    assets = assets[assets["enabled"] == True].copy() if "enabled" in assets.columns else assets.copy()
    liquid_assets = assets[assets["category"].astype(str).isin(["investment", "cash", "other"])] if not assets.empty else pd.DataFrame()

    if liquid_assets.empty:
        avg_return, avg_vol = 0.0, 0.0
    else:
        vals = np.array([to_base(inputs, as_float(v, 0.0), as_str(c, inputs.base_currency)) for v, c in zip(liquid_assets["value"], liquid_assets["currency"])], dtype=float)
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
        settings=dict(snapshot["settings"]),
        household=clean_df(from_records(snapshot["household"]), "household"),
        assets=clean_df(from_records(snapshot["assets"]), "assets"),
        debt=clean_df(from_records(snapshot["debt"]), "debt"),
        income=clean_df(from_records(snapshot["income"]), "income"),
        expenses=clean_df(from_records(snapshot["expenses"]), "expenses"),
        fx=clean_df(from_records(snapshot["fx"]), "fx"),
        tax=clean_df(from_records(snapshot["tax"]), "tax"),
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


def optimize(inputs: Inputs) -> Tuple[pd.DataFrame, list[str]]:
    rows = []
    base_proj = build_projection(inputs)
    base_mc = monte_carlo(inputs)
    base_success = float((base_mc.iloc[:, -1] > 0).mean())
    base_ret = base_proj[base_proj["age"] == inputs.retirement_age].iloc[0]
    rows.append({"strategy": "Current plan", "retirement_age": inputs.retirement_age, "success_rate": base_success, "liquid_at_retirement": float(base_ret["liquid_assets_end"])})

    max_extra = as_int(inputs.settings.get("optimizer_max_extra_work_years", 10), 10)
    for extra in range(1, max_extra + 1):
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
st.title("Retirement Planner")
st.caption("This file removes the direct int(row['sale_age']) pattern that was crashing your Results page.")

if page == "Welcome":
    st.write("Use the left sidebar to work through the plan. Keep Auto recalculate off while entering data.")
elif page == "Household":
    st.session_state["household_records"] = st.data_editor(from_records(st.session_state["household_records"]), num_rows="dynamic", use_container_width=True, key="household_editor").to_dict("records")
elif page == "Assets":
    st.session_state["asset_records"] = st.data_editor(from_records(st.session_state["asset_records"]), num_rows="dynamic", use_container_width=True, key="asset_editor").to_dict("records")
elif page == "Debt":
    st.session_state["debt_records"] = st.data_editor(from_records(st.session_state["debt_records"]), num_rows="dynamic", use_container_width=True, key="debt_editor").to_dict("records")
elif page == "Income":
    st.session_state["income_records"] = st.data_editor(from_records(st.session_state["income_records"]), num_rows="dynamic", use_container_width=True, key="income_editor").to_dict("records")
elif page == "Expenses":
    st.session_state["expense_records"] = st.data_editor(from_records(st.session_state["expense_records"]), num_rows="dynamic", use_container_width=True, key="expense_editor").to_dict("records")
elif page == "FX & Tax":
    c1, c2 = st.columns(2)
    with c1:
        st.session_state["fx_records"] = st.data_editor(from_records(st.session_state["fx_records"]), num_rows="dynamic", use_container_width=True, key="fx_editor").to_dict("records")
    with c2:
        st.session_state["tax_records"] = st.data_editor(from_records(st.session_state["tax_records"]), num_rows="dynamic", use_container_width=True, key="tax_editor").to_dict("records")
elif page == "Settings":
    s = st.session_state["settings"]
    s["base_spending_pre75"] = st.number_input("Base spending before 75", min_value=0.0, value=as_float(s["base_spending_pre75"], 100000.0), step=1000.0)
    s["base_spending_post75"] = st.number_input("Base spending after 75", min_value=0.0, value=as_float(s["base_spending_post75"], 80000.0), step=1000.0)
    s["inflation"] = st.slider("Inflation", 0.0, 0.10, as_float(s["inflation"], 0.025), step=0.001, format="%.1f%%")
    s["mc_runs"] = st.number_input("Monte Carlo runs", 500, 5000, as_int(s["mc_runs"], 1000), step=500)
    s["random_seed"] = st.number_input("Random seed", 0, 999999, as_int(s["random_seed"], 42), step=1)
    s["optimizer_max_extra_work_years"] = st.number_input("Max extra work years", 1, 20, as_int(s["optimizer_max_extra_work_years"], 10), step=1)
    st.session_state["settings"] = s
elif page == "Results":
    if not (st.session_state["auto_recalc"] or st.session_state["apply_counter"] > 0):
        st.info("Click Apply inputs and recalculate in the sidebar first.")
    else:
        snapshot = snapshot_state()
        proj, mc, summary = run_results(snapshot)
        inputs = get_inputs()
        ret_row = proj[proj["age"] == inputs.retirement_age].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Liquid at retirement", fmt_money(from_base(inputs, as_float(ret_row["liquid_assets_end"], 0.0), inputs.display_currency), inputs.display_currency))
        c2.metric("Success rate", fmt_pct(summary["success_rate"]))
        c3.metric("Median final liquid", fmt_money(from_base(inputs, as_float(summary["median_final_liquid"], 0.0), inputs.display_currency), inputs.display_currency))
        chart = proj[["age", "liquid_assets_end", "net_worth_end"]].copy().set_index("age")
        chart = chart.apply(lambda col: col.map(lambda x: from_base(inputs, as_float(x, 0.0), inputs.display_currency)))
        st.line_chart(chart)
elif page == "Optimizer":
    if not (st.session_state["auto_recalc"] or st.session_state["apply_counter"] > 0):
        st.info("Click Apply inputs and recalculate in the sidebar first.")
    else:
        inputs = get_inputs()
        results, notes = optimize(inputs)
        st.dataframe(results, use_container_width=True, hide_index=True)
        for note in notes:
            st.markdown(f"- {note}")
