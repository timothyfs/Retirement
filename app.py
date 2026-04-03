import json
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

SCENARIO_FILE = Path("scenarios.json")
DEFAULT_SCENARIO_NAME = "Base"


@dataclass
class Inputs:
    # Household
    age: int = 53
    wife_age: int = 51
    retirement_age: int = 55
    life_expectancy: int = 95

    # Assets and liabilities
    liquid_portfolio: float = 1_000_000
    property_value: float = 1_000_000
    mortgage: float = 50_000
    monthly_contribution: float = 5_000

    # Spending
    spending_pre75: float = 100_000
    spending_post75: float = 80_000
    inflation: float = 0.025
    spending_stress: float = 0.00

    # Pensions
    your_pension_age: int = 67
    your_pension: float = 14_000
    wife_pension_age: int = 67
    wife_pension: float = 13_000

    # Other income
    rental_income: float = 24_000
    rental_income_start_age: int = 55
    consulting_income: float = 0.0
    consulting_start_age: int = 55
    consulting_end_age: int = 60

    # Allocation
    cash_weight: float = 0.39
    equity_weight: float = 0.11
    property_weight: float = 0.50

    cash_return: float = 0.02
    equity_return: float = 0.07
    property_growth: float = 0.03

    cash_vol: float = 0.01
    equity_vol: float = 0.18

    # Options
    property_sale_age: int = 0
    property_sale_proceeds: float = 0.0

    # Simulation
    simulations: int = 2000
    seed: int = 42


def load_scenarios():
    if SCENARIO_FILE.exists():
        try:
            return json.loads(SCENARIO_FILE.read_text())
        except Exception:
            pass
    base = {DEFAULT_SCENARIO_NAME: asdict(Inputs())}
    save_scenarios(base)
    return base


def save_scenarios(data):
    SCENARIO_FILE.write_text(json.dumps(data, indent=2))


def normalize_inputs(inp: Inputs) -> Inputs:
    float_fields = [
        "liquid_portfolio",
        "property_value",
        "mortgage",
        "monthly_contribution",
        "spending_pre75",
        "spending_post75",
        "inflation",
        "spending_stress",
        "your_pension",
        "wife_pension",
        "rental_income",
        "consulting_income",
        "cash_weight",
        "equity_weight",
        "property_weight",
        "cash_return",
        "equity_return",
        "property_growth",
        "cash_vol",
        "equity_vol",
        "property_sale_proceeds",
    ]

    int_fields = [
        "age",
        "wife_age",
        "retirement_age",
        "life_expectancy",
        "your_pension_age",
        "wife_pension_age",
        "rental_income_start_age",
        "consulting_start_age",
        "consulting_end_age",
        "property_sale_age",
        "simulations",
        "seed",
    ]

    for field_name in float_fields:
        setattr(inp, field_name, float(getattr(inp, field_name)))

    for field_name in int_fields:
        setattr(inp, field_name, int(getattr(inp, field_name)))

    return inp


def blended_liquid_return(inp: Inputs) -> float:
    return inp.cash_weight * inp.cash_return + inp.equity_weight * inp.equity_return


def blended_liquid_vol(inp: Inputs) -> float:
    return inp.cash_weight * inp.cash_vol + inp.equity_weight * inp.equity_vol


def indexed(value: float, inflation: float, years: int) -> float:
    return value * ((1 + inflation) ** max(0, years))


def annual_spending(inp: Inputs, age: int) -> float:
    if age < inp.retirement_age:
        return 0.0
    years = age - inp.retirement_age
    base = inp.spending_pre75 if age < 75 else inp.spending_post75
    return indexed(base, inp.inflation, years) * (1 + inp.spending_stress)


def annual_income(inp: Inputs, age: int) -> dict:
    your_pension = indexed(inp.your_pension, inp.inflation, age - inp.your_pension_age) if age >= inp.your_pension_age else 0.0
    wife_pension = indexed(inp.wife_pension, inp.inflation, age - inp.wife_pension_age) if age >= inp.wife_pension_age else 0.0
    rental = indexed(inp.rental_income, inp.inflation, age - inp.rental_income_start_age) if age >= inp.rental_income_start_age else 0.0
    consulting = 0.0
    if inp.consulting_start_age <= age <= inp.consulting_end_age:
        consulting = indexed(inp.consulting_income, inp.inflation, age - inp.consulting_start_age)
    return {
        "your_pension": your_pension,
        "wife_pension": wife_pension,
        "rental": rental,
        "consulting": consulting,
        "total": your_pension + wife_pension + rental + consulting,
    }


def deterministic(inp: Inputs) -> pd.DataFrame:
    liq_return = blended_liquid_return(inp)
    liquid = inp.liquid_portfolio
    property_val = inp.property_value
    property_sold = False

    rows = []

    for age in range(inp.age, inp.life_expectancy + 1):
        liquid_start = liquid
        property_start = property_val

        contribution = inp.monthly_contribution * 12 if age < inp.retirement_age else 0.0
        spending = annual_spending(inp, age)
        income = annual_income(inp, age)
        withdrawal = max(0.0, spending - income["total"])

        liquid_growth = liquid_start * liq_return
        property_growth_amt = property_start * inp.property_growth if not property_sold else 0.0

        sale_cash = 0.0
        if (
            not property_sold
            and inp.property_sale_age > 0
            and age >= inp.property_sale_age
            and inp.property_sale_proceeds > 0
        ):
            sale_cash = inp.property_sale_proceeds
            property_val = 0.0
            property_sold = True
            property_growth_amt = 0.0

        liquid_end = max(0.0, liquid_start + contribution + liquid_growth - withdrawal + sale_cash)
        property_end = 0.0 if property_sold else max(0.0, property_start + property_growth_amt)

        liquid = liquid_end
        property_val = property_end
        net_worth = liquid_end + property_end - inp.mortgage

        rows.append(
            {
                "Age": age,
                "Liquid Start": liquid_start,
                "Property Start": property_start,
                "Contribution": contribution,
                "Liquid Growth": liquid_growth,
                "Property Growth": property_growth_amt,
                "Spending": spending,
                "Your Pension": income["your_pension"],
                "Wife Pension": income["wife_pension"],
                "Rental Income": income["rental"],
                "Consulting Income": income["consulting"],
                "Non-Portfolio Income": income["total"],
                "Net Withdrawal": withdrawal,
                "Property Sale Cash": sale_cash,
                "Liquid End": liquid_end,
                "Property End": property_end,
                "Net Worth": net_worth,
            }
        )

    return pd.DataFrame(rows)


def monte_carlo(inp: Inputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(inp.seed)

    det = deterministic(inp)
    retirement_row = det.loc[det["Age"] == inp.retirement_age].iloc[0]
    retirement_liquid = float(retirement_row["Liquid Start"])
    retirement_property = float(retirement_row["Property Start"])

    age_slice = det[det["Age"] >= inp.retirement_age].copy()
    ages = age_slice["Age"].tolist()
    withdrawals = dict(zip(age_slice["Age"], age_slice["Net Withdrawal"]))

    mean_ret = blended_liquid_return(inp)
    vol = blended_liquid_vol(inp)

    summary_rows = []
    paths = []

    for sim in range(inp.simulations):
        liquid = retirement_liquid
        property_val = retirement_property
        property_sold = False
        fail_age = None
        min_liquid = liquid
        path = []

        for age in ages:
            annual_ret = max(-0.95, rng.normal(mean_ret, vol))
            withdrawal = withdrawals[age]

            sale_cash = 0.0
            if (
                not property_sold
                and inp.property_sale_age > 0
                and age >= inp.property_sale_age
                and inp.property_sale_proceeds > 0
            ):
                sale_cash = inp.property_sale_proceeds
                property_val = 0.0
                property_sold = True

            liquid = max(0.0, liquid * (1 + annual_ret) - withdrawal + sale_cash)

            if not property_sold:
                property_val = max(0.0, property_val * (1 + inp.property_growth))

            if liquid <= 0 and fail_age is None:
                fail_age = age

            min_liquid = min(min_liquid, liquid)
            path.append(liquid)

        summary_rows.append(
            {
                "Simulation": sim + 1,
                "Success": fail_age is None,
                "Fail Age": fail_age,
                "Final Liquid": liquid,
                "Final Property": property_val,
                "Final Net Worth": liquid + property_val - inp.mortgage,
                "Min Liquid": min_liquid,
            }
        )
        paths.append(path)

    summary = pd.DataFrame(summary_rows)
    paths_df = pd.DataFrame(paths, columns=ages)
    return summary, paths_df


def bridge_analysis(det: pd.DataFrame, inp: Inputs) -> pd.DataFrame:
    both_pensions_age = max(inp.your_pension_age, inp.wife_pension_age)
    bridge = det[(det["Age"] >= inp.retirement_age) & (det["Age"] < both_pensions_age)].copy()

    if bridge.empty:
        return pd.DataFrame(
            [{"Metric": "Bridge years", "Value": 0, "Notes": "No bridge gap"}]
        )

    first = det.loc[det["Age"] == inp.retirement_age].iloc[0]
    rows = [
        ["Retirement age", inp.retirement_age, ""],
        ["Your pension age", inp.your_pension_age, ""],
        ["Wife pension age", inp.wife_pension_age, ""],
        ["Years to both pensions", both_pensions_age - inp.retirement_age, ""],
        ["Liquid at retirement", first["Liquid Start"], ""],
        ["Spending at retirement", first["Spending"], ""],
        ["Income at retirement", first["Non-Portfolio Income"], ""],
        ["Net withdrawal at retirement", first["Net Withdrawal"], ""],
        ["Cumulative bridge withdrawals", bridge["Net Withdrawal"].sum(), "Until both pensions start"],
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value", "Notes"])


def safety_score(success_rate: float, p10_final: float, retirement_withdrawal_rate: float) -> tuple[str, str, str]:
    score = 0
    if success_rate >= 0.90:
        score += 2
    elif success_rate >= 0.75:
        score += 1

    if p10_final > 0:
        score += 1

    if retirement_withdrawal_rate <= 0.04:
        score += 2
    elif retirement_withdrawal_rate <= 0.055:
        score += 1

    if score >= 4:
        return "Strong", "#1b7f3b", "Plan looks resilient under current assumptions."
    if score >= 2:
        return "Borderline", "#c98a00", "Plan may work, but the margin for error is thinner."
    return "At Risk", "#b42318", "Current plan is exposed. Retirement age, spending, or allocation likely need work."


def optimizer_retirement_age(inp: Inputs, min_age=50, max_age=65) -> pd.DataFrame:
    rows = []
    for age in range(min_age, max_age + 1):
        test = Inputs(**{**asdict(inp), "retirement_age": age})
        det = deterministic(test)
        mc, _ = monte_carlo(test)
        retirement_row = det.loc[det["Age"] == age].iloc[0]
        withdrawal_rate = (
            retirement_row["Net Withdrawal"] / retirement_row["Liquid Start"]
            if retirement_row["Liquid Start"] > 0
            else np.nan
        )
        rows.append(
            {
                "Retirement Age": age,
                "Success Rate": mc["Success"].mean(),
                "Liquid at Retirement": retirement_row["Liquid Start"],
                "Net Withdrawal at Retirement": retirement_row["Net Withdrawal"],
                "Withdrawal Rate": withdrawal_rate,
            }
        )
    return pd.DataFrame(rows)


def spending_sensitivity(inp: Inputs, spend_values, retire_values) -> pd.DataFrame:
    grid = []
    for r_age in retire_values:
        row = []
        for spend in spend_values:
            test = Inputs(
                **{
                    **asdict(inp),
                    "retirement_age": int(r_age),
                    "spending_pre75": float(spend),
                    "spending_post75": float(spend * 0.8),
                }
            )
            mc, _ = monte_carlo(test)
            row.append(mc["Success"].mean())
        grid.append(row)

    return pd.DataFrame(grid, index=retire_values, columns=spend_values)


def scenario_compare(inp: Inputs) -> pd.DataFrame:
    variants = {
        "Base": inp,
        "Retire 57": Inputs(**{**asdict(inp), "retirement_age": 57}),
        "Spend 90k": Inputs(**{**asdict(inp), "spending_pre75": 90_000, "spending_post75": 72_000}),
        "Sell property at 75": Inputs(**{**asdict(inp), "property_sale_age": 75, "property_sale_proceeds": inp.property_value}),
    }

    rows = []
    for name, test in variants.items():
        det = deterministic(test)
        mc, _ = monte_carlo(test)
        r = det.loc[det["Age"] == test.retirement_age].iloc[0]
        rows.append(
            {
                "Scenario": name,
                "Retirement Age": test.retirement_age,
                "Success Rate": mc["Success"].mean(),
                "Liquid at Retirement": r["Liquid Start"],
                "Net Withdrawal": r["Net Withdrawal"],
                "Median Final Net Worth": mc["Final Net Worth"].median(),
            }
        )
    return pd.DataFrame(rows)


def money(x):
    return f"€{x:,.0f}"


def percent(x):
    return f"{x:.1%}"


def render_gauge(label: str, color: str, subtitle: str):
    st.markdown(
        f"""
        <div style="
            border-radius:18px;
            padding:18px 20px;
            background:linear-gradient(135deg, {color}22, #ffffff);
            border:1px solid {color}55;
            box-shadow:0 4px 16px rgba(0,0,0,0.06);
        ">
            <div style="font-size:14px; color:#555;">Retirement safety</div>
            <div style="font-size:34px; font-weight:700; color:{color}; margin-top:4px;">{label}</div>
            <div style="font-size:14px; color:#666; margin-top:8px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Retirement Cockpit", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            padding: 14px;
            border-radius: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    scenarios = load_scenarios()
    names = list(scenarios.keys())

    st.title("Retirement Planning Cockpit")
    st.caption("Scenario-led retirement planning with deterministic projections, Monte Carlo analysis, bridge-to-pension logic, and decision dashboards.")

    with st.sidebar:
        st.header("Scenario")
        selected_name = st.selectbox("Load scenario", names, index=names.index(DEFAULT_SCENARIO_NAME) if DEFAULT_SCENARIO_NAME in names else 0)
        inp = normalize_inputs(Inputs(**scenarios[selected_name]))

        st.subheader("Household")
        inp.age = st.number_input(
            "Your age",
            min_value=18,
            max_value=100,
            value=int(inp.age),
            step=1,
        )
        inp.wife_age = st.number_input(
            "Wife age",
            min_value=18,
            max_value=100,
            value=int(inp.wife_age),
            step=1,
        )
        inp.retirement_age = st.number_input(
            "Retirement age",
            min_value=40,
            max_value=100,
            value=int(inp.retirement_age),
            step=1,
        )
        inp.life_expectancy = st.number_input(
            "Life expectancy",
            min_value=60,
            max_value=110,
            value=int(inp.life_expectancy),
            step=1,
        )

        st.subheader("Assets and liabilities")
        inp.liquid_portfolio = st.number_input(
            "Liquid portfolio (€)",
            min_value=0.0,
            max_value=100_000_000.0,
            value=float(inp.liquid_portfolio),
            step=10_000.0,
        )
        inp.property_value = st.number_input(
            "Property value (€)",
            min_value=0.0,
            max_value=100_000_000.0,
            value=float(inp.property_value),
            step=10_000.0,
        )
        inp.mortgage = st.number_input(
            "Mortgage (€)",
            min_value=0.0,
            max_value=100_000_000.0,
            value=float(inp.mortgage),
            step=5_000.0,
        )
        inp.monthly_contribution = st.number_input(
            "Monthly contribution (€)",
            min_value=0.0,
            max_value=1_000_000.0,
            value=float(inp.monthly_contribution),
            step=500.0,
        )

        st.subheader("Spending")
        inp.spending_pre75 = st.number_input(
            "Spending before 75 (€)",
            min_value=0.0,
            max_value=10_000_000.0,
            value=float(inp.spending_pre75),
            step=5_000.0,
        )
        inp.spending_post75 = st.number_input(
            "Spending after 75 (€)",
            min_value=0.0,
            max_value=10_000_000.0,
            value=float(inp.spending_post75),
            step=5_000.0,
        )
        inp.inflation = st.slider("Inflation", 0.0, 0.10, inp.inflation, 0.001)
        inp.spending_stress = st.slider("Stress uplift to spending", -0.20, 0.30, inp.spending_stress, 0.01)

        st.subheader("Pensions and income")
        inp.your_pension_age = st.number_input(
            "Your pension age",
            min_value=40,
            max_value=100,
            value=int(inp.your_pension_age),
            step=1,
        )
        inp.your_pension = st.number_input(
            "Your pension annual (€)",
            min_value=0.0,
            max_value=1_000_000.0,
            value=float(inp.your_pension),
            step=1_000.0,
        )
        inp.wife_pension_age = st.number_input(
            "Wife pension age",
            min_value=40,
            max_value=100,
            value=int(inp.wife_pension_age),
            step=1,
        )
        inp.wife_pension = st.number_input(
            "Wife pension annual (€)",
            min_value=0.0,
            max_value=1_000_000.0,
            value=float(inp.wife_pension),
            step=1_000.0,
        )
        inp.rental_income = st.number_input(
            "Rental income annual (€)",
            min_value=0.0,
            max_value=5_000_000.0,
            value=float(inp.rental_income),
            step=1_000.0,
        )
        inp.rental_income_start_age = st.number_input(
            "Rental income start age",
            min_value=18,
            max_value=100,
            value=int(inp.rental_income_start_age),
            step=1,
        )
        inp.consulting_income = st.number_input(
            "Consulting income annual (€)",
            min_value=0.0,
            max_value=5_000_000.0,
            value=float(inp.consulting_income),
            step=1_000.0,
        )
        inp.consulting_start_age = st.number_input(
            "Consulting start age",
            min_value=18,
            max_value=100,
            value=int(inp.consulting_start_age),
            step=1,
        )
        inp.consulting_end_age = st.number_input(
            "Consulting end age",
            min_value=18,
            max_value=100,
            value=int(inp.consulting_end_age),
            step=1,
        )

        st.subheader("Allocation")
        inp.cash_weight = st.slider("Cash weight", 0.0, 1.0, inp.cash_weight, 0.01)
        inp.equity_weight = st.slider("Equity weight", 0.0, 1.0, inp.equity_weight, 0.01)
        inp.property_weight = st.slider("Property weight", 0.0, 1.0, inp.property_weight, 0.01)
        total_alloc = inp.cash_weight + inp.equity_weight + inp.property_weight
        st.write(f"Allocation total: {total_alloc:.0%}")
        inp.cash_return = st.slider("Cash return", -0.02, 0.10, inp.cash_return, 0.001)
        inp.equity_return = st.slider("Equity return", -0.05, 0.15, inp.equity_return, 0.001)
        inp.property_growth = st.slider("Property growth", -0.05, 0.12, inp.property_growth, 0.001)
        inp.cash_vol = st.slider("Cash volatility", 0.0, 0.10, inp.cash_vol, 0.001)
        inp.equity_vol = st.slider("Equity volatility", 0.0, 0.50, inp.equity_vol, 0.001)

        st.subheader("Simulation")
        inp.property_sale_age = st.number_input(
            "Property sale age (0 = off)",
            min_value=0,
            max_value=110,
            value=int(inp.property_sale_age),
            step=1,
        )
        inp.property_sale_proceeds = st.number_input(
            "Property sale proceeds (€)",
            min_value=0.0,
            max_value=100_000_000.0,
            value=float(inp.property_sale_proceeds),
            step=10_000.0,
        )
        inp.simulations = st.slider("Monte Carlo runs", 200, 10000, inp.simulations, 100)
        inp.seed = st.number_input(
            "Random seed",
            min_value=1,
            max_value=999999,
            value=int(inp.seed),
            step=1,
        )

        new_name = st.text_input("Scenario name", value=selected_name)
        c1, c2 = st.columns(2)
        if c1.button("Save"):
            scenarios[new_name] = asdict(inp)
            save_scenarios(scenarios)
            st.success(f"Saved '{new_name}'")
        if c2.button("Delete"):
            if selected_name != DEFAULT_SCENARIO_NAME:
                scenarios.pop(selected_name, None)
                save_scenarios(scenarios)
                st.success(f"Deleted '{selected_name}'")
            else:
                st.warning("Base cannot be deleted.")

    det = deterministic(inp)
    mc_summary, mc_paths = monte_carlo(inp)
    bridge = bridge_analysis(det, inp)
    compare = scenario_compare(inp)
    optimizer = optimizer_retirement_age(inp, 50, 65)

    retirement_row = det.loc[det["Age"] == inp.retirement_age].iloc[0]
    current_net_worth = inp.liquid_portfolio + inp.property_value - inp.mortgage
    success_rate = float(mc_summary["Success"].mean())
    median_final_net_worth = float(mc_summary["Final Net Worth"].median())
    p10_final = float(mc_summary["Final Liquid"].quantile(0.10))
    withdrawal_rate = float(retirement_row["Net Withdrawal"] / retirement_row["Liquid Start"]) if retirement_row["Liquid Start"] > 0 else 0.0
    safe_label, safe_color, safe_text = safety_score(success_rate, p10_final, withdrawal_rate)

    spend_values = [80_000, 90_000, 100_000, 110_000, 120_000]
    retire_values = list(range(53, 61))
    sens = spending_sensitivity(inp, spend_values, retire_values)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Cockpit", "Plan", "Monte Carlo", "Bridge", "Scenarios"]
    )

    with tab1:
        a, b, c, d = st.columns(4)
        a.metric("Current net worth", money(current_net_worth))
        b.metric("Liquid at retirement", money(float(retirement_row["Liquid Start"])))
        c.metric("Success rate", percent(success_rate))
        d.metric("Retirement withdrawal rate", percent(withdrawal_rate))

        e, f, g, h = st.columns(4)
        e.metric("Net withdrawal at retirement", money(float(retirement_row["Net Withdrawal"])))
        f.metric("Median final net worth", money(median_final_net_worth))
        g.metric("10th percentile final liquid", money(p10_final))
        h.metric("Blended liquid return", percent(blended_liquid_return(inp)))

        left, right = st.columns((2, 1))
        with left:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(det["Age"], det["Liquid End"], label="Liquid portfolio")
            ax.plot(det["Age"], det["Property End"], label="Property")
            ax.plot(det["Age"], det["Net Worth"], label="Net worth")
            ax.set_xlabel("Age")
            ax.set_ylabel("€")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)

        with right:
            render_gauge(safe_label, safe_color, safe_text)

            st.markdown("### Key pressure points")
            st.write(f"Years to both pensions: **{max(0, max(inp.your_pension_age, inp.wife_pension_age) - inp.retirement_age)}**")
            st.write(f"Spending before 75: **{money(inp.spending_pre75)}**")
            st.write(f"Spending after 75: **{money(inp.spending_post75)}**")
            st.write(f"Rental income: **{money(inp.rental_income)}**")
            st.write(f"Property sale enabled: **{'Yes' if inp.property_sale_age > 0 else 'No'}**")

        st.markdown("### Retirement year cash flow")
        cashflow = pd.DataFrame(
            {
                "Component": [
                    "Spending",
                    "Your pension",
                    "Wife pension",
                    "Rental income",
                    "Consulting income",
                    "Net withdrawal",
                ],
                "Amount": [
                    retirement_row["Spending"],
                    retirement_row["Your Pension"],
                    retirement_row["Wife Pension"],
                    retirement_row["Rental Income"],
                    retirement_row["Consulting Income"],
                    retirement_row["Net Withdrawal"],
                ],
            }
        )
        st.dataframe(cashflow, use_container_width=True)

    with tab2:
        st.subheader("Deterministic lifetime plan")
        st.dataframe(
            det[
                [
                    "Age",
                    "Liquid Start",
                    "Contribution",
                    "Spending",
                    "Non-Portfolio Income",
                    "Net Withdrawal",
                    "Liquid End",
                    "Property End",
                    "Net Worth",
                ]
            ],
            use_container_width=True,
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(det["Age"], det["Net Withdrawal"])
        ax.set_xlabel("Age")
        ax.set_ylabel("Net withdrawal (€)")
        ax.grid(True, axis="y", alpha=0.3)
        st.pyplot(fig)

    with tab3:
        left, right = st.columns((2, 1))
        with left:
            fig, ax = plt.subplots(figsize=(10, 5))
            sample = mc_paths.sample(min(100, len(mc_paths)), random_state=inp.seed)
            for _, row in sample.iterrows():
                ax.plot(mc_paths.columns, row.values, alpha=0.08)
            ax.set_xlabel("Age")
            ax.set_ylabel("Liquid portfolio (€)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with right:
            st.markdown("### Monte Carlo summary")
            st.write(f"Runs: **{inp.simulations}**")
            st.write(f"Success rate: **{percent(success_rate)}**")
            st.write(f"Median final liquid: **{money(float(mc_summary['Final Liquid'].median()))}**")
            st.write(f"10th percentile final liquid: **{money(float(mc_summary['Final Liquid'].quantile(0.10)))}**")
            fail_ages = mc_summary["Fail Age"].dropna()
            st.write(f"Median failure age: **{int(fail_ages.median()) if not fail_ages.empty else 'No failures'}**")

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.hist(mc_summary["Final Net Worth"], bins=40)
        ax2.set_xlabel("Final net worth (€)")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    with tab4:
        st.subheader("Bridge to pension")
        st.dataframe(bridge, use_container_width=True)

        bridge_slice = det[
            (det["Age"] >= inp.retirement_age)
            & (det["Age"] <= max(inp.your_pension_age, inp.wife_pension_age))
        ]
        if not bridge_slice.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(bridge_slice["Age"], bridge_slice["Net Withdrawal"])
            ax.set_xlabel("Age")
            ax.set_ylabel("Bridge withdrawal (€)")
            ax.grid(True, axis="y", alpha=0.3)
            st.pyplot(fig)

    with tab5:
        st.subheader("Scenario comparison")
        st.dataframe(compare, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(compare["Scenario"], compare["Success Rate"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Success rate")
        ax.grid(True, axis="y", alpha=0.3)
        st.pyplot(fig)

        st.subheader("Retirement age optimizer")
        st.dataframe(optimizer, use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(optimizer["Retirement Age"], optimizer["Success Rate"], marker="o")
        ax2.set_xlabel("Retirement age")
        ax2.set_ylabel("Success rate")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

        st.subheader("Spending sensitivity")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        im = ax3.imshow(sens.values, aspect="auto")
        ax3.set_xticks(range(len(sens.columns)))
        ax3.set_xticklabels([f"€{int(v/1000)}k" for v in sens.columns])
        ax3.set_yticks(range(len(sens.index)))
        ax3.set_yticklabels(sens.index)
        ax3.set_xlabel("Spending before 75")
        ax3.set_ylabel("Retirement age")
        plt.colorbar(im, ax=ax3, label="Success rate")
        st.pyplot(fig3)


if __name__ == "__main__":
    main()