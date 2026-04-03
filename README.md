# Retirement Planning Cockpit

A Streamlit-based retirement planning and scenario analysis app for modelling household wealth, retirement timing, bridge-to-pension risk, and long-term portfolio sustainability.

This app is designed to be a practical decision cockpit rather than a spreadsheet clone. It combines deterministic planning, Monte Carlo simulation, scenario management, retirement age optimisation, and spending sensitivity analysis in one interface.

---

## What the app does

The app helps you explore questions like:

- Can I retire at 55?
- How exposed am I before state pensions begin?
- How much spending is sustainable?
- How sensitive is my plan to bad market returns?
- How much safer would my plan be if I retired later, spent less, or sold a property?

It currently includes:

- sidebar-driven input model
- saved scenarios
- deterministic year-by-year retirement projection
- Monte Carlo simulation
- bridge-to-pension analysis
- retirement safety gauge
- retirement age optimiser
- spending sensitivity heatmap
- scenario comparison view

---

## Core features

### 1. Scenario-driven planning
You can create, save, load, update, and delete named scenarios.

Examples:
- Base
- Retire at 55
- Retire at 57
- Lower Spending
- Sell Property at 75
- Consulting Bridge

Scenarios are stored locally in a JSON file.

---

### 2. Deterministic projection
The deterministic model gives a year-by-year projection from current age to life expectancy.

It models:
- liquid portfolio growth
- property growth
- mortgage drag on net worth
- pre-retirement contributions
- retirement spending
- spending reduction after age 75
- your state pension
- your wife’s state pension
- rental income
- consulting or bridge income
- optional property sale proceeds

This is the straight-line planning case.

---

### 3. Monte Carlo simulation
The Monte Carlo engine runs many simulated retirement paths to estimate how robust the plan is under uncertain returns.

It produces:
- success rate
- failure age distribution
- median final liquid wealth
- downside outcomes
- median final net worth

This helps quantify sequence-of-returns risk and downside exposure.

---

### 4. Bridge-to-pension analysis
This view focuses on the years between retirement and the point when both pensions are in payment.

It highlights:
- years to both pensions
- liquid portfolio at retirement
- spending at retirement
- income at retirement
- net withdrawal at retirement
- cumulative bridge withdrawals

This is especially important for early retirement planning.

---

### 5. Retirement age optimiser
This module runs the model across a range of retirement ages and compares:
- success rate
- liquid assets at retirement
- withdrawal rate at retirement

It helps answer:
- how much safer is retiring at 57 than 55?
- what age gets me above a target probability of success?

---

### 6. Spending sensitivity analysis
This module tests combinations of:
- retirement age
- spending level

It shows how success probability changes as you adjust both.

This is one of the most useful decision tools in the app.

---

## Project structure

A simple single-file version may look like this:

```text
retirement_cockpit/
├── app.py
├── requirements.txt
├── scenarios.json
└── README.md