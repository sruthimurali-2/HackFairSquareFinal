import numpy as np
import pandas as pd

# --- OIS Curve Mapping ---
ois_curve_map = {
    "USD": "USD.OIS",
    "EUR": "EUR.OIS",
    "GBP": "GBP.OIS",
    "JPY": "JPY.OIS"
}
#--- Load Observability Grids ---
ir_grid = pd.read_csv("ir_delta_observability_grid.csv")
ir_grid.columns = ir_grid.columns.str.strip()
ir_grid["Observable Tenor (Years)"] = pd.to_numeric(ir_grid["Observable Tenor (Years)"], errors="coerce")
vol_grid = pd.read_csv("volatility_observability_grid.csv")
vol_grid.columns = vol_grid.columns.str.strip()


# Generate Risk Factors
def simulate_greeks(trade):

    base = trade["notional"] / 1_000_000
    tenor_factor = trade["maturity_tenor"] / 10
    vol_factor = 0.01 * np.random.uniform(0.8, 1.2)
    return {
        "OIS Curve": ois_curve_map.get(trade["currency"], "UNKNOWN"),
        "IRDelta 1Y": round(base * 0.5 * tenor_factor, 2),
        "IRDelta 5Y": round(base * 1.0 * tenor_factor, 2),
        "IRDelta 10Y": round(base * 1.5 * tenor_factor, 2),
        "IRDelta 30Y": round(base * 2.0 * tenor_factor, 2),
        "Vega": round(base * 0.6 * vol_factor, 2),
        "Vanna": round(base * 0.3 * vol_factor, 2),
        "Volga": round(base * 0.4 * vol_factor, 2),
    }

# Generate Trade PV and Risk Factor PVs
def generate_trade_pv_and_risk_pvs(greeks):
    pv_greeks = {}
    total_pv = 0
    for key in ["IRDelta 1Y", "IRDelta 5Y", "IRDelta 10Y", "IRDelta 30Y", "Vega", "Vanna", "Volga"]:
        pv = round(np.random.uniform(5000, 20000), 2)
        pv_greeks[key + " PV"] = pv
        total_pv += abs(pv)
    return total_pv, pv_greeks

# IR Delta Stress Test
def ir_delta_stress_test(trade, greeks):
    messages = []
    stressed = {}
    report = {}
    total_stress_pv = 0

    curve_id = ois_curve_map[trade["currency"]]

    # ✅ Rename before using the column
    ir_grid.rename(columns={"Stress Factor (%)": "Stress Factor"}, inplace=True)
    ir_grid["Stress Factor"] = pd.to_numeric(ir_grid["Stress Factor"], errors="coerce")
    ir_grid["Observable Tenor (Years)"] = pd.to_numeric(ir_grid["Observable Tenor (Years)"], errors="coerce")

    for tenor in ["1Y", "5Y", "10Y", "30Y"]:
        col_name = f"IRDelta {tenor}"
        tenor_years = int(tenor.strip("Y"))
        base_pv = greeks.get(col_name + " PV", 0)

        grid_row = ir_grid[
            (ir_grid["Curve ID"] == curve_id) &
            (ir_grid["Observable Tenor (Years)"] >= tenor_years)
        ]

        observable = not grid_row.empty
        if observable:
            stress_factor = 0.0
            stressed_pv = base_pv * stress_factor
        else:
            fallback_row = ir_grid[ir_grid["Curve ID"] == curve_id]
            stress_factor = fallback_row["Stress Factor"].values[0] if not fallback_row.empty else 1.0
            stressed_pv = base_pv * stress_factor
            messages.append(f"⚠️ {col_name} for {curve_id} risk considered Unobservable")
            total_stress_pv += abs(stressed_pv)

        stressed[col_name] = stressed_pv
        report[col_name] = {
            "Observable": observable,
            "Base PV": base_pv,
            "Stressed PV": stressed_pv,
            "StressFactor": stress_factor if not observable else 0.0
        }

    return stressed, report, total_stress_pv, messages

# Volatility Risk Stress Test
def vol_risk_stress_test(trade, greeks):
    messages = []
    stressed = {}
    report = {}
    total_stress_pv = 0

    for risk in ["Vega", "Vanna", "Volga"]:
        base_pv = greeks.get(risk + " PV", 0)

        grid_row = vol_grid[
            (vol_grid["Risk Type"] == risk) &
            (vol_grid["Currency"] == trade["currency"]) &
            (vol_grid["Max Observable Tenor"] >= trade["maturity_tenor"]) &
            (vol_grid["Max Observable Expiry"] >= trade["expiry_tenor"])
        ]

        observable = not grid_row.empty
        if observable:
            stress_factor = 0.0
            stressed_pv = base_pv * stress_factor
        else:
            fallback_row = vol_grid[
                (vol_grid["Risk Type"] == risk) &
                (vol_grid["Currency"] == trade["currency"])
            ]
            stress_factor = fallback_row["Stress Factor"].values[0] if not fallback_row.empty else 1.0
            stressed_pv = base_pv * stress_factor
            messages.append(f"⚠️ {risk} risk considered Unobservable")
            total_stress_pv += abs(stressed_pv)

        stressed[risk] = stressed_pv
        report[risk] = {
            "Observable": observable,
            "Base PV": base_pv,
            "Stressed PV": stressed_pv,
            "StressFactor": stress_factor if not observable else 0.0
        }

    return stressed, report, total_stress_pv, messages

# Decion Maker - Combine & Final Assessment
def run_full_observability_stress_test(trade, greeks):
    # Ensure PVs are generated
    if "trade_pv" not in trade:
        trade["trade_pv"], generated_pvs = generate_trade_pv_and_risk_pvs(greeks)
        greeks.update(generated_pvs)

    # Run individual stress tests
    ir_stressed, ir_report, ir_stress_pv, ir_msgs = ir_delta_stress_test(trade, greeks)
    vol_stressed, vol_report, vol_stress_pv, vol_msgs = vol_risk_stress_test(trade, greeks)

    total_stress_pv = ir_stress_pv + vol_stress_pv
    final_level = "Level 3" if total_stress_pv > 0.1 * trade["trade_pv"] else "Level 2"

    # Combine all
    final_stressed = {**ir_stressed, **vol_stressed}
    final_report = {**ir_report, **vol_report}
    final_stressed["Total Stress PV"] = round(total_stress_pv, 2)
    final_stressed["Total Trade PV"] = round(trade["trade_pv"], 2)
    final_stressed["Final IFRS13 Level"] = final_level

    messages = ir_msgs + vol_msgs
    return final_stressed, final_report, messages