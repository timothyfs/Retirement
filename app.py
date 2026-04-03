
import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Retirement Planning Cockpit",
    page_icon="📈",
    layout="wide",
)

st.title("Retirement Planning Cockpit")
st.caption(
    "Scenario-led retirement planning with deterministic projections, Monte Carlo analysis, "
    "bridge-to-pension logic, and decision dashboards."
)

CURRENCIES: Dict[str, Dict[str, str]] = {
    "EUR": {"symbol": "€", "name": "Euro"},
    "GBP": {"symbol": "£", "name": "British Pound"},
    "USD": {"symbol": "$", "name": "US Dollar"},
    "CHF": {"symbol": "CHF ", "name": "Swiss Franc"},
}

DEFAULTS = {
    "scenario_name": "Base",
    "load_scenario": "Base",
    "currency": "EUR",
    "your_age": 53,
    "wife_age": 51,
    "retirement_age": 55,
    "life_expectancy": 95,
    "liquid_portfolio": 1_070_000.0,
    "property_value": 1_020_000.0,
    "mortgage": 50_000.0,
    "monthly_contribution": 5_000.0,
    "spending_before_75": 100_000.0,
    "spending_after_75": 80_000.0,
    "inflation": 0.025,
    "stress_uplift": 0.00,
    "your_pension_age": 67,
    "your_pension_annual": 14_000.0,
    "wife_pension_age": 67,
    "wife_pension_annual": 13_000.0,
    "rental_income_annual": 24_000.0,
    "rental_income_start_age": 55,
    "consulting_income_annual": 10_000.0,
    "consulting_income_start_age": 55,
    "consulting_income_end_age": 60,
    "cash_weight": 0.08,
    "equity_weight": 0.62,
    "property_weight": 0.30,
    "cash_return": 0.02,
    "equity_return": 0.065,
    "property_growth": 0.025,
    "cash_volatility": 0.01,
    "equity_volatility": 0.16,
    "property_volatility": 0.07,
    "property_sale_age": 75,
    "property_sale_proceeds": 500_000.0,
    "mc_runs": 5000,
    "random_seed": 42,
}


def init_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def fmt_money(value: float, currency: str) -> str:
    symbol = CURRENCIES[currency]["symbol"]
    return f"{symbol}{value:,.0f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}"


def safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


@dataclass
class Inputs:
    scenario_name: str
    currency: str
    your_age: int
    wife_age: int
    retirement_age: int
    life_expectancy: int
    liquid_portfolio: float
    property_value: float
    mortgage: float
    monthly_contribution: float
    spending_before_75: float
    spending_after_75: float
    inflation: float
    stress_uplift: float
    your_pension_age: int
    your_pension_annual: float
    wife_pension_age: int
    wife_pension_annual: float
    rental_income_annual: float
    rental_income_start_age: int
    consulting_income_annual: float
    consulting_income_start_age: int
    consulting_income_end_age: int
    cash_weight: float
    equity_weight: float
    property_weight: float
    cash_return: float
    equity_return: float
    property_growth: float
    cash_volatility: float
    equity_volatility: float
    property_volatility: float
    property_sale_age: int
    property_sale_proceeds: float
    mc_runs: int
    random_seed: int

    @property
    def total_weight(self) -> float:
        return self.cash_weight + self.equity_weight + self.property_weight

    @property
    def blended_liquid_return(self) -> float:
        return self.cash_weight * self.cash_return + self.equity_weight * self.equity_return

    @property
    def blended_total_return(self) -> float:
        return (
            self.cash_weight * self.cash_return
            + self.equity_weight * self.equity_return
            + self.property_weight * self.property_growth
        )

    @property
    def blended_liquid_volatility(self) -> float:
        return self.cash_weight * self.cash_volatility + self.equity_weight * self.equity_volatility

    @property
    def current_net_worth(self) -> float:
        return self.liquid_portfolio + self.property_value - self.mortgage


def get_inputs() -> Inputs:
    return Inputs(
        scenario_name=st.session_state.scenario_name,
        currency=st.session_state.currency,
        your_age=int(st.session_state.your_age),
        wife_age=int(st.session_state.wife_age),
        retirement_age=int(st.session_state.retirement_age),
        life_expectancy=int(st.session_state.life_expectancy),
        liquid_portfolio=float(st.session_state.liquid_portfolio),
        property_value=float(st.session_state.property_value),
        mortgage=float(st.session_state.mortgage),
        monthly_contribution=float(st.session_state.monthly_contribution),
        spending_before_75=float(st.session_state.spending_before_75),
        spending_after_75=float(st.session_state.spending_after_75),
        inflation=float(st.session_state.inflation),
        stress_uplift=float(st.session_state.stress_uplift),
        your_pension_age=int(st.session_state.your_pension_age),
        your_pension_annual=float(st.session_state.your_pension_annual),
        wife_pension_age=int(st.session_state.wife_pension_age),
        wife_pension_annual=float(st.session_state.wife_pension_annual),
        rental_income_annual=float(st.session_state.rental_income_annual),
        rental_income_start_age=int(st.session_state.rental_income_start_age),
        consulting_income_annual=float(st.session_state.consulting_income_annual),
        consulting_income_start_age=int(st.session_state.consulting_income_start_age),
        consulting_income_end_age=int(st.session_state.consulting_income_end_age),
        cash_weight=float(st.session_state.cash_weight),
        equity_weight=float(st.session_state.equity_weight),
        property_weight=float(st.session_state.property_weight),
        cash_return=float(st.session_state.cash_return),
        equity_return=float(st.session_state.equity_return),
        property_growth=float(st.session_state.property_growth),
        cash_volatility=float(st.session_state.cash_volatility),
        equity_volatility=float(st.session_state.equity_volatility),
        property_volatility=float(st.session_state.property_volatility),
        property_sale_age=int(st.session_state.property_sale_age),
        property_sale_proceeds=float(st.session_state.property_sale_proceeds),
        mc_runs=int(st.session_state.mc_runs),
        random_seed=int(st.session_state.random_seed),
    )


def annual_spending(age: int, inputs: Inputs) -> float:
    base = inputs.spending_before_75 if age < 75 else inputs.spending_after_75
    base *= 1.0 + inputs.stress_uplift
    years_after_retirement = max(0, age - inputs.retirement_age)
    return base * ((1.0 + inputs.inflation) ** years_after_retirement)


def indexed_income(age: int, start_age: int, base_amount: float, inputs: Inputs) -> float:
    if age < start_age:
        return 0.0
    years_since_start = age - start_age
    return base_amount * ((1.0 + inputs.inflation) ** years_since_start)


def total_pension_income(age: int, inputs: Inputs) -> float:
    return indexed_income(age, inputs.your_pension_age, inputs.your_pension_annual, inputs) + indexed_income(
        age, inputs.wife_pension_age, inputs.wife_pension_annual, inputs
    )


def total_other_income(age: int, inputs: Inputs) -> float:
    rental = inputs.rental_income_annual if age >= inputs.rental_income_start_age else 0.0
    consulting = (
        inputs.consulting_income_annual
        if inputs.consulting_income_start_age <= age <= inputs.consulting_income_end_age
        else 0.0
    )
    return rental + consulting


def deterministic_projection(inputs: Inputs) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    liquid = inputs.liquid_portfolio
    property_value = inputs.property_value
    property_sold = False

    for age in range(inputs.your_age, inputs.retirement_age):
        contribution = inputs.monthly_contribution * 12
        growth = liquid * inputs.blended_liquid_return
        liquid_end = liquid + contribution + growth

        rows.append(
            {
                "phase": "Pre-retirement",
                "age": age,
                "liquid_start": liquid,
                "contribution": contribution,
                "spending": 0.0,
                "income": 0.0,
                "growth": growth,
                "liquid_end": liquid_end,
                "property_value": property_value,
                "net_worth_end": liquid_end + property_value - inputs.mortgage,
            }
        )
        liquid = liquid_end

    for age in range(inputs.retirement_age, inputs.life_expectancy + 1):
        spending = annual_spending(age, inputs)
        income = total_pension_income(age, inputs) + total_other_income(age, inputs)
        withdrawal = max(0.0, spending - income)

        if inputs.property_sale_age > 0 and age >= inputs.property_sale_age and not property_sold:
            liquid += inputs.property_sale_proceeds
            property_value = max(0.0, property_value - inputs.property_sale_proceeds)
            property_sold = True

        growth = liquid * inputs.blended_liquid_return
        liquid_end = max(0.0, liquid + growth - withdrawal)

        rows.append(
            {
                "phase": "Retirement",
                "age": age,
                "liquid_start": liquid,
                "contribution": 0.0,
                "spending": spending,
                "income": income,
                "growth": growth,
                "liquid_end": liquid_end,
                "property_value": property_value,
                "net_worth_end": liquid_end + property_value - inputs.mortgage,
            }
        )
        liquid = liquid_end

    return pd.DataFrame(rows)


def monte_carlo_projection(inputs: Inputs, start_liquid: float) -> pd.DataFrame:
    rng = np.random.default_rng(inputs.random_seed)
    ages = list(range(inputs.retirement_age, inputs.life_expectancy + 1))
    results = np.zeros((inputs.mc_runs, len(ages)), dtype=float)

    for run in range(inputs.mc_runs):
        liquid = start_liquid
        property_sold = False

        for idx, age in enumerate(ages):
            spending = annual_spending(age, inputs)
            income = total_pension_income(age, inputs) + total_other_income(age, inputs)
            withdrawal = max(0.0, spending - income)

            if inputs.property_sale_age > 0 and age >= inputs.property_sale_age and not property_sold:
                liquid += inputs.property_sale_proceeds
                property_sold = True

            cash_r = rng.normal(inputs.cash_return, inputs.cash_volatility)
            equity_r = rng.normal(inputs.equity_return, inputs.equity_volatility)
            blended_liquid_return = inputs.cash_weight * cash_r + inputs.equity_weight * equity_r

            liquid = max(0.0, liquid * (1.0 + blended_liquid_return) - withdrawal)
            results[run, idx] = liquid

    return pd.DataFrame(results, columns=ages)


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


init_state()

with st.sidebar:
    st.header("Display")
    st.selectbox(
        "Currency",
        options=list(CURRENCIES.keys()),
        format_func=lambda code: f"{code} — {CURRENCIES[code]['name']}",
        key="currency",
        help="Changes display symbol and formatting only. It does not convert values or FX rates.",
    )
    st.markdown(
        "All money fields use **thousands separators** for easier reading. "
        "Use **Apply inputs** after edits so the app updates in one pass."
    )

with st.form("retirement_input_form", clear_on_submit=False):
    tab_cockpit, tab_plan, tab_monte_carlo, tab_bridge, tab_scenarios = st.tabs(
        ["Cockpit", "Plan", "Monte Carlo", "Bridge", "Scenarios"]
    )

    with tab_cockpit:
        st.subheader("Scenario")
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Scenario name", key="scenario_name")
        with c2:
            st.selectbox("Load scenario", options=["Base"], key="load_scenario")

        st.subheader("Household")
        h1, h2, h3, h4 = st.columns(4)
        h1.number_input("Your age", min_value=18, max_value=100, step=1, format="%d", key="your_age")
        h2.number_input("Wife age", min_value=18, max_value=100, step=1, format="%d", key="wife_age")
        h3.number_input("Retirement age", min_value=18, max_value=100, step=1, format="%d", key="retirement_age")
        h4.number_input("Life expectancy", min_value=18, max_value=110, step=1, format="%d", key="life_expectancy")

        st.subheader("Assets and liabilities")
        a1, a2, a3, a4 = st.columns(4)
        a1.number_input("Liquid portfolio", min_value=0.0, step=1_000.0, format="%0.0f", key="liquid_portfolio")
        a2.number_input("Property value", min_value=0.0, step=1_000.0, format="%0.0f", key="property_value")
        a3.number_input("Mortgage", min_value=0.0, step=1_000.0, format="%0.0f", key="mortgage")
        a4.number_input("Monthly contribution", min_value=0.0, step=100.0, format="%0.0f", key="monthly_contribution")

        st.subheader("Spending")
        s1, s2, s3, s4 = st.columns(4)
        s1.number_input("Spending before 75", min_value=0.0, step=1_000.0, format="%0.0f", key="spending_before_75")
        s2.number_input("Spending after 75", min_value=0.0, step=1_000.0, format="%0.0f", key="spending_after_75")
        s3.slider("Inflation", min_value=0.0, max_value=0.10, step=0.001, format="%.1f%%", key="inflation")
        s4.slider("Stress uplift to spending", min_value=-0.20, max_value=0.30, step=0.01, format="%.0f%%", key="stress_uplift")

        st.subheader("Pensions and income")
        p1, p2, p3, p4 = st.columns(4)
        p1.number_input("Your pension age", min_value=18, max_value=100, step=1, format="%d", key="your_pension_age")
        p2.number_input("Your pension annual", min_value=0.0, step=500.0, format="%0.0f", key="your_pension_annual")
        p3.number_input("Wife pension age", min_value=18, max_value=100, step=1, format="%d", key="wife_pension_age")
        p4.number_input("Wife pension annual", min_value=0.0, step=500.0, format="%0.0f", key="wife_pension_annual")

        i1, i2, i3, i4 = st.columns(4)
        i1.number_input("Rental income annual", min_value=0.0, step=1_000.0, format="%0.0f", key="rental_income_annual")
        i2.number_input("Rental income start age", min_value=18, max_value=100, step=1, format="%d", key="rental_income_start_age")
        i3.number_input("Consulting income annual", min_value=0.0, step=1_000.0, format="%0.0f", key="consulting_income_annual")
        i4.number_input("Consulting start age", min_value=18, max_value=100, step=1, format="%d", key="consulting_income_start_age")

        ce1, ce2 = st.columns([1, 3])
        ce1.number_input("Consulting end age", min_value=18, max_value=100, step=1, format="%d", key="consulting_income_end_age")
        ce2.markdown("")

    with tab_plan:
        st.subheader("Allocation")
        w1, w2, w3 = st.columns(3)
        w1.slider("Cash weight", min_value=0.0, max_value=1.0, step=0.01, key="cash_weight")
        w2.slider("Equity weight", min_value=0.0, max_value=1.0, step=0.01, key="equity_weight")
        w3.slider("Property weight", min_value=0.0, max_value=1.0, step=0.01, key="property_weight")

        r1, r2, r3 = st.columns(3)
        r1.slider("Cash return", min_value=-0.02, max_value=0.10, step=0.001, format="%.1f%%", key="cash_return")
        r2.slider("Equity return", min_value=-0.05, max_value=0.15, step=0.001, format="%.1f%%", key="equity_return")
        r3.slider("Property growth", min_value=-0.05, max_value=0.12, step=0.001, format="%.1f%%", key="property_growth")

        v1, v2, v3 = st.columns(3)
        v1.slider("Cash volatility", min_value=0.0, max_value=0.10, step=0.001, format="%.1f%%", key="cash_volatility")
        v2.slider("Equity volatility", min_value=0.0, max_value=0.50, step=0.001, format="%.1f%%", key="equity_volatility")
        v3.slider("Property volatility", min_value=0.0, max_value=0.25, step=0.001, format="%.1f%%", key="property_volatility")

    with tab_monte_carlo:
        st.subheader("Simulation")
        m1, m2, m3, m4 = st.columns(4)
        m1.number_input("Property sale age (0 = off)", min_value=0, max_value=110, step=1, format="%d", key="property_sale_age")
        m2.number_input("Property sale proceeds", min_value=0.0, step=10_000.0, format="%0.0f", key="property_sale_proceeds")
        m3.number_input("Monte Carlo runs", min_value=200, max_value=10000, step=100, format="%d", key="mc_runs")
        m4.number_input("Random seed", min_value=0, max_value=999999, step=1, format="%d", key="random_seed")

        st.info(
            "This model uses annual normal return assumptions for liquid assets, "
            "bridge-to-pension income logic, and optional property-sale proceeds."
        )

    with tab_bridge:
        st.subheader("Bridge logic")
        st.markdown(
            """
            - Spending inflates from retirement onward.
            - State pensions start at the ages you set and are indexed using the inflation assumption.
            - Rental and consulting income reduce portfolio withdrawals.
            - Property sale proceeds are injected into liquid assets once at the chosen sale age.
            """
        )

    with tab_scenarios:
        st.subheader("Scenario ideas")
        st.markdown(
            """
            - **Base**: your central case.
            - **Delay retirement**: push retirement age by 2 to 5 years.
            - **Defensive**: lower returns, lower spending, more cash.
            - **Stress test**: higher inflation, lower returns, spending uplift.
            """
        )

    st.form_submit_button("Apply inputs", type="primary", use_container_width=True)

inputs = get_inputs()

if not math.isclose(inputs.total_weight, 1.0, abs_tol=0.01):
    st.warning(
        f"Allocation total is {inputs.total_weight:.0%}. Aim for 100% so returns and volatility are coherent."
    )

projection = deterministic_projection(inputs)
retirement_row = projection.query("phase == 'Retirement' and age == @inputs.retirement_age").iloc[0]
retirement_liquid = float(retirement_row["liquid_start"])
retirement_spending = annual_spending(inputs.retirement_age, inputs)
retirement_income = total_pension_income(inputs.retirement_age, inputs) + total_other_income(inputs.retirement_age, inputs)
net_withdrawal = max(0.0, retirement_spending - retirement_income)
withdrawal_rate = safe_div(net_withdrawal, retirement_liquid)

mc_df = monte_carlo_projection(inputs, retirement_liquid)
mc_summary = summarize_mc(mc_df)
safety = safety_label(mc_summary["success_rate"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current net worth", fmt_money(inputs.current_net_worth, inputs.currency))
m2.metric("Liquid at retirement", fmt_money(retirement_liquid, inputs.currency))
m3.metric("Success rate", fmt_pct(mc_summary["success_rate"]))
m4.metric("Retirement withdrawal rate", fmt_pct(withdrawal_rate))

m5, m6, m7, m8 = st.columns(4)
m5.metric("Net withdrawal at retirement", fmt_money(net_withdrawal, inputs.currency))
m6.metric("Median final liquid", fmt_money(mc_summary["median_final_liquid"], inputs.currency))
m7.metric("10th percentile final liquid", fmt_money(mc_summary["p10_final_liquid"], inputs.currency))
m8.metric("Blended liquid return", fmt_pct(inputs.blended_liquid_return))

st.subheader("Retirement safety")
if safety == "Healthy":
    st.success(f"{safety} — current plan looks resilient on these assumptions.")
elif safety == "Watchlist":
    st.warning(f"{safety} — the plan may work, but spending pressure and sequence risk matter.")
else:
    st.error(f"{safety} — retirement age, spending, or allocation likely need work.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Years to both pensions", str(max(inputs.your_pension_age, inputs.wife_pension_age) - inputs.retirement_age))
k2.metric("Spending before 75", fmt_money(inputs.spending_before_75, inputs.currency))
k3.metric("Spending after 75", fmt_money(inputs.spending_after_75, inputs.currency))
k4.metric("Property sale enabled", "Yes" if inputs.property_sale_age > 0 else "No")

st.subheader("Retirement year cash flow")
cashflow_df = pd.DataFrame(
    {
        "Item": [
            "Spending need",
            "State pensions",
            "Rental income",
            "Consulting income",
            "Net portfolio withdrawal",
        ],
        "Value": [
            retirement_spending,
            total_pension_income(inputs.retirement_age, inputs),
            inputs.rental_income_annual if inputs.retirement_age >= inputs.rental_income_start_age else 0.0,
            inputs.consulting_income_annual
            if inputs.consulting_income_start_age <= inputs.retirement_age <= inputs.consulting_income_end_age
            else 0.0,
            net_withdrawal,
        ],
    }
)
cashflow_df["Display"] = cashflow_df["Value"].map(lambda value: fmt_money(value, inputs.currency))
st.dataframe(cashflow_df[["Item", "Display"]], use_container_width=True, hide_index=True)

st.subheader("Projection")
projection_chart = projection[["age", "liquid_end", "net_worth_end"]].copy().set_index("age")
st.line_chart(projection_chart)

st.subheader("Monte Carlo distribution")
mc_chart = pd.DataFrame(
    {
        "Age": mc_df.columns.astype(int),
        "Median": mc_df.median(axis=0).values,
        "10th percentile": mc_df.quantile(0.10, axis=0).values,
        "90th percentile": mc_df.quantile(0.90, axis=0).values,
    }
).set_index("Age")
st.line_chart(mc_chart)

with st.expander("Detailed annual projection"):
    display_df = projection.copy()
    money_cols = [
        "liquid_start",
        "contribution",
        "spending",
        "income",
        "growth",
        "liquid_end",
        "property_value",
        "net_worth_end",
    ]
    for col in money_cols:
        display_df[col] = display_df[col].map(lambda value: fmt_money(value, inputs.currency))
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with st.expander("Model notes"):
    st.markdown(
        """
        - Currency switching changes the displayed symbol only. It does not convert values.
        - Comma separators are applied in all displayed outputs, cards, and tables.
        - Streamlit number inputs do not reliably render comma-separated values during typing on every client, so the app formats all outputs consistently after input.
        - For full FX support next, add exchange-rate inputs and a base-currency calculation layer.
        """
    )
