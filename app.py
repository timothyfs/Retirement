import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from pathlib import Path

SCENARIO_FILE = "scenarios.json"

@dataclass
class Inputs:
    age:int=53
    wife_age:int=51
    retirement_age:int=55
    life_expectancy:int=95
    liquid_portfolio:float=1_000_000
    property_value:float=1_000_000
    mortgage:float=50_000
    monthly_contribution:float=5000
    spending_pre75:float=100000
    spending_post75:float=80000
    inflation:float=0.025
    your_pension_age:int=67
    your_pension:float=14000
    wife_pension_age:int=67
    wife_pension:float=13000
    rental_income:float=24000
    equity_return:float=0.07
    volatility:float=0.12
    simulations:int=2000

def load_scenarios():
    if Path(SCENARIO_FILE).exists():
        return json.load(open(SCENARIO_FILE))
    return {"Base":asdict(Inputs())}

def save_scenarios(data):
    json.dump(data,open(SCENARIO_FILE,"w"),indent=2)

def deterministic(inputs:Inputs):

    ages=list(range(inputs.age,inputs.life_expectancy+1))

    liquid=inputs.liquid_portfolio
    property_val=inputs.property_value

    rows=[]

    for age in ages:

        contrib=0
        if age<inputs.retirement_age:
            contrib=inputs.monthly_contribution*12

        spending=0
        if age>=inputs.retirement_age:
            if age<75:
                spending=inputs.spending_pre75
            else:
                spending=inputs.spending_post75

        pension=0
        if age>=inputs.your_pension_age:
            pension+=inputs.your_pension
        if age>=inputs.wife_pension_age:
            pension+=inputs.wife_pension

        income=pension+inputs.rental_income

        withdrawal=max(0,spending-income)

        growth=liquid*inputs.equity_return

        liquid=max(0,liquid+contrib+growth-withdrawal)

        property_val*=1.03

        networth=liquid+property_val-inputs.mortgage

        rows.append({
            "Age":age,
            "Liquid":liquid,
            "Property":property_val,
            "Spending":spending,
            "Income":income,
            "Withdrawal":withdrawal,
            "NetWorth":networth
        })

    return pd.DataFrame(rows)

def montecarlo(inputs:Inputs):

    start_liquid=inputs.liquid_portfolio

    success=0
    results=[]

    for i in range(inputs.simulations):

        liquid=start_liquid

        for age in range(inputs.retirement_age,inputs.life_expectancy):

            r=np.random.normal(inputs.equity_return,inputs.volatility)

            spending=inputs.spending_pre75 if age<75 else inputs.spending_post75

            pension=0
            if age>=inputs.your_pension_age:
                pension+=inputs.your_pension
            if age>=inputs.wife_pension_age:
                pension+=inputs.wife_pension

            income=pension+inputs.rental_income

            withdrawal=max(0,spending-income)

            liquid=max(0,liquid*(1+r)-withdrawal)

        results.append(liquid)

        if liquid>0:
            success+=1

    return np.array(results),success/inputs.simulations

st.set_page_config(layout="wide")

scenarios=load_scenarios()

st.sidebar.title("Scenario")

scenario_name=st.sidebar.selectbox("Load scenario",list(scenarios.keys()))

inputs=Inputs(**scenarios[scenario_name])

st.sidebar.header("Household")

inputs.age=st.sidebar.number_input("Your age",value=inputs.age)
inputs.wife_age=st.sidebar.number_input("Wife age",value=inputs.wife_age)
inputs.retirement_age=st.sidebar.number_input("Retirement age",value=inputs.retirement_age)

st.sidebar.header("Portfolio")

inputs.liquid_portfolio=st.sidebar.number_input("Liquid portfolio",value=inputs.liquid_portfolio)
inputs.property_value=st.sidebar.number_input("Property value",value=inputs.property_value)
inputs.monthly_contribution=st.sidebar.number_input("Monthly contribution",value=inputs.monthly_contribution)

st.sidebar.header("Spending")

inputs.spending_pre75=st.sidebar.number_input("Spending before 75",value=inputs.spending_pre75)
inputs.spending_post75=st.sidebar.number_input("Spending after 75",value=inputs.spending_post75)

st.sidebar.header("Simulation")

inputs.equity_return=st.sidebar.slider("Expected return",0.0,0.12,inputs.equity_return)
inputs.volatility=st.sidebar.slider("Volatility",0.0,0.3,inputs.volatility)
inputs.simulations=st.sidebar.slider("Monte Carlo runs",500,10000,inputs.simulations)

if st.sidebar.button("Save scenario"):
    scenarios[scenario_name]=asdict(inputs)
    save_scenarios(scenarios)

st.title("Retirement Cockpit")

det=deterministic(inputs)

mc,success=montecarlo(inputs)

c1,c2,c3=st.columns(3)

c1.metric("Success probability",f"{success:.1%}")
c2.metric("Median ending wealth",f"€{np.median(mc):,.0f}")
c3.metric("Worst 10%",f"€{np.percentile(mc,10):,.0f}")

st.subheader("Net worth projection")

fig,ax=plt.subplots()

ax.plot(det["Age"],det["NetWorth"])

st.pyplot(fig)

st.subheader("Monte Carlo distribution")

fig,ax=plt.subplots()

ax.hist(mc,40)

st.pyplot(fig)

st.subheader("Projection table")

st.dataframe(det)