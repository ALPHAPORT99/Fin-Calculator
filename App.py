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

def stock_price_constant_dividend(dividend, r):
    return dividend / r

def stock_price_constant_growth(D0, r, g):
    if r <= g:
        return None
    return (D0 * (1 + g)) / (r - g)

def stock_required_return_gordon(D1, P0, g):
    return (D1 / P0) + g

def stock_price_nonconstant(dividends, r, g, start_growth_year):
    pv_sum = 0.0
    for i, div in enumerate(dividends):
        year = i + 1
        pv_sum += div / ((1 + r)**year)
    Dn = dividends[-1]
    D_next = Dn * (1 + g)
    if r <= g:
        return None
    tv = D_next / (r - g)
    pv_tv = tv / ((1 + r)**start_growth_year)
    return pv_sum + pv_tv

def corporate_value_model(fcf_list, r, g, debt, shares):
    pv_sum = 0.0
    for i, fcf in enumerate(fcf_list):
        year = i + 1
        pv_sum += fcf / ((1 + r)**year)
    if len(fcf_list) == 0 or r <= g:
        return None
    FCFn = fcf_list[-1]
    FCF_next = FCFn * (1 + g)
    tv = FCF_next / (r - g)
    N = len(fcf_list)
    pv_tv = tv / ((1 + r)**N)
    mv_firm = pv_sum + pv_tv
    mv_equity = mv_firm - debt
    return mv_equity / shares

def pe_multiple_valuation(eps, pe_ratio):
    return eps * pe_ratio

def free_cash_flow(ebit, tax_rate, depreciation, capex, delta_nwc):
    return ebit * (1 - tax_rate) + depreciation - capex - delta_nwc

def expected_return(probabilities, returns):
    return np.sum(np.array(probabilities) * np.array(returns))

def wacc(equity, debt, re, rd, tax_rate):
    return (equity / (equity + debt)) * re + (debt / (equity + debt)) * rd * (1 - tax_rate)

def after_tax_cost_of_debt(rd, tax_rate):
    return rd * (1 - tax_rate)
