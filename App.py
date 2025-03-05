import streamlit as st
import numpy as np
import math
import sympy as sp  

def future_value(present_value, r, n):
    return present_value * (1 + r)**n

def present_value(fv, r, n):
    return fv / ((1 + r)**n)

def annuity_pv(C, r, n):
    return C * (1 - (1 + r)**(-n)) / r

def annuity_fv(C, r, n):
    return C * ((1 + r)**n - 1) / r

def growing_annuity_pv(C, r, g, n):
    if abs(r - g) < 1e-12:
        return C * n / (1 + r)
    return C * (1 - ((1 + g)/(1 + r))**n) / (r - g)

def perpetuity_pv(C, r):
    return C / r

def growing_perpetuity_pv(C, r, g):
    if r <= g:
        return None
    return C / (r - g)

def npv_calc(cash_flows, r):
    total = 0.0
    for i, cf in enumerate(cash_flows):
        total += cf / ((1 + r)**i)
    return total

def irr_calc(cash_flows, guess=0.1, max_iterations=100, tolerance=1e-6):
    def npv_rate(rate):
        total = 0.0
        for i, cf in enumerate(cash_flows):
            total += cf / ((1 + rate)**i)
        return total
    
    rate0 = guess
    rate1 = guess + 0.01  
    npv0 = npv_rate(rate0)
    for _ in range(max_iterations):
        npv1 = npv_rate(rate1)
        if (npv1 - npv0) == 0:
            break
        rate2 = rate1 - npv1 * (rate1 - rate0)/(npv1 - npv0)
        if abs(rate2 - rate1) < tolerance:
            return rate2
        rate0, rate1 = rate1, rate2
        npv0 = npv1
    return rate1

def payback_period(cash_flows):
    cumulative = 0.0
    for i, cf in enumerate(cash_flows):
        cumulative += cf
        if cumulative >= 0:
            return i
    return None

def discounted_payback_period(cash_flows, r):
    cumulative = 0.0
    for i, cf in enumerate(cash_flows):
        discounted_cf = cf / ((1 + r)**i)
        cumulative += discounted_cf
        if cumulative >= 0:
            return i
    return None

def bond_price_coupon(face_value, coupon_rate, years_to_maturity, coupons_per_year, ytm_annual):
    C = (coupon_rate * face_value) / coupons_per_year
    r = ytm_annual / coupons_per_year
    T = int(coupons_per_year * years_to_maturity)
    price = annuity_pv(C, r, T) + present_value(face_value, r, T)
    return price

def bond_ytm_coupon(face_value, coupon_rate, years_to_maturity, coupons_per_year, bond_price):
    C = (coupon_rate * face_value) / coupons_per_year
    T = int(coupons_per_year * years_to_maturity)
    cf_list = [-bond_price] + [C]*(T-1) + [C + face_value]
    period_irr = irr_calc(cf_list, guess=0.05)
    annual_irr = period_irr * coupons_per_year
    return period_irr, annual_irr

def bond_coupon_rate(bond_price, face_value, annual_ytm, years_to_maturity, coupons_per_year):
    r = annual_ytm / coupons_per_year
    T = coupons_per_year * years_to_maturity
    discounted_face = present_value(face_value, r, T)
    annuity_factor = annuity_pv(1, r, T)
    numerator = bond_price - discounted_face
    denominator = (face_value / coupons_per_year) * annuity_factor
    if abs(denominator) < 1e-12:
        return None
    c = numerator / denominator
    return c

st.set_page_config(page_title="Finance Formulas App", layout="wide")
st.title("Finance Formulas App")
st.write("Use the selectors below to navigate formulas by category.")

categories = {
    "Time Value of Money": ["Future Value (FV)", "Present Value (PV)", "Annuity (PV)", "Annuity (FV)", "Growing Annuity (PV)", "Perpetuity", "Growing Perpetuity"],
    "Bond": ["Bond Price (Coupon)", "Bond YTM (Coupon)", "Bond Coupon Rate"],
    "Capital Budgeting": ["Net Present Value (NPV)", "Internal Rate of Return (IRR)", "Payback Period", "Discounted Payback Period"],
    "Stock Valuation": ["Stock - Constant Dividend Price", "Stock - Constant Growth Dividend Price (Gordon Growth)", "Stock - Required Return (Gordon Growth)", "Stock - Non-Constant Growth Dividend Price", "Corporate Value (FCF) Model", "Stock Price from PE Multiple", "Free Cash Flow (FCF)"],
    "Risk Valuation": ["Expected Return", "Average Return", "CAPM Expected Return", "Beta Calculation"],
    "Cost of Capital": ["WACC Calculation", "Cost of Equity (CAPM)", "After-Tax Cost of Debt"]
}

selected_category = st.selectbox("Select a Category:", list(categories.keys()))
selected_formula = st.selectbox("Select a Formula:", categories[selected_category])
st.write("---")

if st.button("Calculate"):
    st.write(f"Selected formula: {selected_formula}")
    st.success("Feature implementation coming soon!")
