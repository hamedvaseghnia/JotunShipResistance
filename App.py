
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, dash_table

# -----------------------------
# Constants & Defaults
# -----------------------------
K_FORM_CONST = 0.15         # constant form factor
NU_CONST     = 1.05e-6      # constant kinematic viscosity [m^2/s]
G_CONST      = 9.81         # gravity [m/s^2]
KS_SPC_M     = 150e-6       # SPC roughness height [m] (fixed)

DEFAULTS = {
    "L_ship": 230.0,   # m
    "B_ship": 32.2,    # m
    "T_ship": 10.8,    # m
    "rho":    1025.0,  # kg/m^3
    "v_min_kn": 11.0,  # knots
    "v_max_kn": 24.0,  # knots
    "v_step_kn": 0.5,  # knots
}

# -----------------------------
# Hydrodynamic models (from user code)
# -----------------------------
def wetted_area(L, B, T):
    return 1.7 * L * (B + T)

def cf_ittc(Re):
    return 0.075 / ((np.log10(Re) - 2.0) ** 2)

def delta_cf(Re, ks, LWL):
    return 0.044 * ((ks / LWL) ** (1/3)) - 0.44 * Re**(-1/3) + 0.000125

# --- FRC (Granville with S(L+)) ---
Re_flow = np.array([45350.57965, 92735.86107, 142110.199, 191913.4081,
                    244680.8013, 295904.4004, 347541.3412, 399163.7682])
Cf_rough_flow = np.array([0.004688874, 0.003962825, 0.003665055, 0.0034877,
                          0.003331024, 0.00324357, 0.003189113, 0.003119389])
Cf_smooth_measured = 0.027 * Re_flow**(-0.17)
Ub_rough  = np.sqrt(2.0 / Cf_rough_flow)
Ub_smooth = np.sqrt(2.0 / Cf_smooth_measured)
DeltaU    = Ub_smooth - Ub_rough

kappa = 0.41
S_points = (kappa * DeltaU) / np.log(10.0)
Lplus_points = Re_flow * np.sqrt(Cf_rough_flow / 2.0)

order = np.argsort(Lplus_points)
logL_sorted = np.log10(Lplus_points[order])
S_sorted    = S_points[order]

def S_of_Lplus(Lp):
    Lp = np.asarray(Lp)
    Lp = np.maximum(Lp, 1.0)
    return np.interp(np.log10(Lp), logL_sorted, S_sorted,
                     left=S_sorted[0], right=S_sorted[-1])

def f_smooth(Cf, Re):
    return 0.242 / np.sqrt(Cf) - np.log10(Re * Cf)

def df_smooth(Cf):
    return -0.242 / (2.0 * Cf**1.5) - 1.0 / (np.log(10.0) * Cf)

def cf_smooth_implicit(Re, tol=1e-12, max_iter=80):
    Cf = float(cf_ittc(Re))
    if not np.isfinite(Cf) or Cf <= 0:
        Cf = 1e-3
    for _ in range(max_iter):
        r  = f_smooth(Cf, Re)
        dr = df_smooth(Cf)
        Cf_new = Cf - r/dr
        if not np.isfinite(Cf_new) or Cf_new <= 0:
            Cf_new = 0.5 * Cf
        if abs(Cf_new - Cf) <= tol * max(Cf, 1e-16):
            return Cf_new
        Cf = 0.5 * Cf + 0.5 * Cf_new
    return Cf

def cf_granville(Re, tol=1e-12, max_iter=60):
    Cf = cf_smooth_implicit(Re)
    S = 0.0
    for _ in range(max_iter):
        Lp   = Re * np.sqrt(Cf / 2.0)
        S    = S_of_Lplus(Lp)
        Cf_n = cf_smooth_implicit(Re * (10.0 ** (-S)))
        if abs(Cf_n - Cf) <= tol * max(Cf, 1e-16):
            return Cf_n, S
        Cf = 0.5 * Cf + 0.5 * Cf_n
    return Cf, S

# -----------------------------
# Computation
# -----------------------------
def compute_results(L_ship, B_ship, T_ship, rho, v_min_kn, v_max_kn, v_step_kn):
    speeds_knots = np.arange(v_min_kn, v_max_kn + 1e-9, v_step_kn)
    speeds = speeds_knots * 0.51445  # m/s

    S_ship = wetted_area(L_ship, B_ship, T_ship)
    Re_ship = (speeds * L_ship) / NU_CONST
    Fn = speeds / np.sqrt(G_CONST * L_ship)

    Cf_FRC, _ = zip(*(cf_granville(R) for R in Re_ship))
    Cf_FRC = np.array(Cf_FRC)

    dCf_SPC = delta_cf(Re_ship, KS_SPC_M, L_ship)
    Cf_SPC = cf_ittc(Re_ship) + dCf_SPC

    CT_FRC = (1.0 + K_FORM_CONST) * Cf_FRC
    CT_SPC = (1.0 + K_FORM_CONST) * Cf_SPC

    # We no longer plot PE vs speed, but keep it in the table if useful
    PE_FRC = 0.5 * rho * S_ship * (speeds**3) * CT_FRC
    PE_SPC = 0.5 * rho * S_ship * (speeds**3) * CT_SPC

    percent_delta_PE = 100.0 * (PE_FRC - PE_SPC) / np.where(PE_SPC==0, np.nan, PE_SPC)

    df = pd.DataFrame({
        "Speed [kn]": speeds_knots,
        "Fn": Fn,
        "Re": Re_ship,
        "Cf_FRC": Cf_FRC,
        "Cf_SPC": Cf_SPC,
        "PE_FRC [MW]": PE_FRC / 1e6,
        "PE_SPC [MW]": PE_SPC / 1e6,
        "ΔPE [%] (FRC vs SPC)": percent_delta_PE,
    })
    return df

# -----------------------------
# App
# -----------------------------
app = Dash(__name__, title="App")
server = app.server

def number_input(id_, label, value, step, min_=None):
    return html.Div([
        html.Label(label, style={"fontWeight":"600"}),
        dcc.Input(id=id_, type="number", value=value, step=step, min=min_, debounce=True, style={"width":"100%"}),
    ], style={"marginBottom":"10px"})

app.layout = html.Div([
    html.Div([
        html.H2("🚢 FRC vs SPC — Ship Characteristics (k, ν constant; SPC ks = 150 μm)",
                style={"margin":"0 0 8px 0"}),
        html.Div(f"Constants: k = {K_FORM_CONST}, ν = {NU_CONST} m²/s, g = {G_CONST} m/s²; SPC ks = 150 μm.",
                 style={"color":"#6b7280","fontSize":"12px"}),
    ], style={"marginBottom":"10px"}),

    html.Div([
        # Sidebar
        html.Div([
            html.H4("Inputs"),
            number_input("L_ship", "Length L [m]", DEFAULTS["L_ship"], 0.1, 1.0),
            number_input("B_ship", "Beam B [m]",   DEFAULTS["B_ship"], 0.1, 1.0),
            number_input("T_ship", "Draft T [m]",  DEFAULTS["T_ship"], 0.1, 0.1),
            number_input("rho", "Density ρ [kg/m³]", DEFAULTS["rho"], 1.0, 1.0),
            html.H4("Speed range (knots)"),
            number_input("v_min_kn", "Min",  DEFAULTS["v_min_kn"], 0.5, 0.1),
            number_input("v_max_kn", "Max",  DEFAULTS["v_max_kn"], 0.5, 0.2),
            number_input("v_step_kn","Step", DEFAULTS["v_step_kn"], 0.1, 0.1),
            html.Button("Recalculate", id="calc_btn", style={"marginTop":"6px"}),
            html.Div(id="status", style={"color":"#6b7280","fontSize":"12px","marginTop":"6px"}),
        ], style={"width":"320px","padding":"14px","background":"#fff","borderRadius":"12px",
                  "boxShadow":"0 8px 24px rgba(2,8,20,0.06)"}),

        # Content
        html.Div([
            dcc.Graph(id="fig_cf", config={"displaylogo": False}, style={"height":"520px"}),
            dcc.Graph(id="fig_bar", config={"displaylogo": False}, style={"height":"520px"}),
            html.H4("Results table", style={"marginTop":"6px"}),
            dash_table.DataTable(
                id="results_table",
                columns=[{"name": c, "id": c} for c in [
                    "Speed [kn]","Fn","Re","Cf_FRC","Cf_SPC","PE_FRC [MW]","PE_SPC [MW]","ΔPE [%] (FRC vs SPC)"
                ]],
                data=[],
                page_size=15,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX":"auto"},
                style_data={"fontFamily":"Segoe UI, Inter, sans-serif","fontSize":"13px"},
                style_header={"fontWeight":"600","backgroundColor":"#f3f4f6"}
            )
        ], style={"flex":"1","display":"grid","gridTemplateColumns":"1fr","gap":"14px"}),

    ], style={"display":"flex","gap":"18px","padding":"16px","background":"#f0f2f7",
              "maxWidth":"1400px","margin":"0 auto"})
])

@app.callback(
    Output("fig_cf", "figure"),
    Output("fig_bar", "figure"),
    Output("results_table", "data"),
    Output("status", "children"),
    Input("calc_btn", "n_clicks"),
    State("L_ship", "value"),
    State("B_ship", "value"),
    State("T_ship", "value"),
    State("rho", "value"),
    State("v_min_kn", "value"),
    State("v_max_kn", "value"),
    State("v_step_kn", "value"),
    prevent_initial_call=False
)
def recalc(_, L_ship, B_ship, T_ship, rho, vmin, vmax, vstep):
    try:
        df = compute_results(float(L_ship), float(B_ship), float(T_ship),
                             float(rho), float(vmin), float(vmax), float(vstep))

        # Round for table
        table = df.copy()
        table["Fn"] = table["Fn"].round(5)
        table["Re"] = table["Re"].round(0)
        for col in ["Cf_FRC","Cf_SPC"]:
            table[col] = table[col].astype(float).round(6)
        for col in ["PE_FRC [MW]","PE_SPC [MW]","ΔPE [%] (FRC vs SPC)"]:
            table[col] = table[col].astype(float).round(4)

        x = df["Speed [kn]"]

        # Wider, bolder plots
        fig_cf = go.Figure([
            go.Scatter(x=x, y=df["Cf_FRC"], mode="lines+markers", name="FRC",
                       line=dict(width=3), marker=dict(size=8)),
            go.Scatter(x=x, y=df["Cf_SPC"], mode="lines+markers", name="SPC",
                       line=dict(width=3), marker=dict(size=8)),
        ])
        fig_cf.update_layout(
            title="Friction coefficient Cf vs Speed (Wide)",
            xaxis_title="Speed [kn]",
            yaxis_title="Cf",
            template="plotly_white",
            legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
            margin=dict(l=60, r=30, t=60, b=60),
            height=520
        )

        fig_bar = go.Figure([
            go.Bar(x=x, y=-df["ΔPE [%] (FRC vs SPC)"], name="-(ΔPE%)")
        ])
        fig_bar.update_layout(
            title="Effective Power difference by speed (−ΔPE% = SPC reference)",
            xaxis_title="Speed [kn]",
            yaxis_title="Effective Power difference [%]",
            template="plotly_white",
            legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
            margin=dict(l=60, r=30, t=60, b=60),
            height=520
        )

        status = (f"Computed {len(df)} speeds. Constants → k = {K_FORM_CONST}, ν = {NU_CONST} m²/s, "
                  f"g = {G_CONST} m/s²; SPC ks = 150 μm.")
        return fig_cf, fig_bar, table.to_dict("records"), status
    except Exception as e:
        empty = go.Figure().update_layout(template="plotly_white", height=520)
        return empty, empty, [], f"Error: {e}"

if __name__ == "__main__":
    # Dash >= 2.16
    app.run(debug=True)


