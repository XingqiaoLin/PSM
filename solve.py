"""
Solve the multi-period memory-chip allocation problem with Pyomo + Gurobi.

Model summary
-------------
Sets        I = suppliers (fabs)
            J = customers
            P = products  (HBM3e, DDR5, DDR4)
            T = planning periods
Decisions   x[i,j,p,t]   shipments (kWE)                      continuous ≥ 0
            u[j,p,t]     unmet demand (kWE)                   continuous ≥ 0
            I[i,p,t]     end-of-period inventory at the fab   continuous ≥ 0
            y[i,j]       lane (i,j) opened over horizon       binary
Parameters  S[i,p,t]     supply, D[j,p,t] demand,
            c[i,j,p]     transport cost,  π[j,p] shortage penalty,
            h[p]         holding cost,     e[i,j] kg CO2/unit,
            F            lane fixed cost,  K[i] lane cap per fab,
            B            total-horizon carbon budget,
            φ[j,p]       minimum SLA fill-rate (0 if best-effort).

Objective   min   Σ c x + Σ π u + Σ h I + F Σ y

Constraints
(1) Supply balance       Σ_j x + I[i,p,t] = S[i,p,t] + I[i,p,t-1]
(2) Demand balance       Σ_i x + u[j,p,t] = D[j,p,t]
(3) SLA floor            Σ_i x[i,j,p,t] ≥ φ[j,p] · D[j,p,t]                (if φ>0)
(4) Lane linking         Σ_p,t x[i,j,p,t] ≤ M · y[i,j]
(5) Lane cardinality     Σ_j y[i,j] ≤ K[i]
(6) Carbon budget        Σ_{i,j,p,t} e[i,j] x[i,j,p,t] ≤ B
(7) Non-negativity / inventory continuity

The script
  1. Loads the 12 CSVs under ./data/
  2. Builds (P1)  — the LP relaxation obtained by dropping integrality of y
  3. Builds (P2)  — the full MILP
  4. Solves both with Gurobi (via Pyomo)
  5. Writes ./output/results.json and prints a summary.
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import pyomo.environ as pyo


ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT  = ROOT / "output"
OUT.mkdir(exist_ok=True)


# --------------------------------------------------------------------- load
def load_data() -> dict:
    d = {
        "suppliers":       pd.read_csv(DATA / "suppliers.csv"),
        "customers":       pd.read_csv(DATA / "customers.csv"),
        "products":        pd.read_csv(DATA / "products.csv"),
        "periods":         pd.read_csv(DATA / "periods.csv"),
        "supply":          pd.read_csv(DATA / "supply.csv"),
        "demand":          pd.read_csv(DATA / "demand.csv"),
        "transport_cost":  pd.read_csv(DATA / "transport_cost.csv"),
        "shortage_pen":    pd.read_csv(DATA / "shortage_penalty.csv"),
        "sla_min":         pd.read_csv(DATA / "sla_min.csv"),
        "holding":         pd.read_csv(DATA / "holding_cost.csv"),
        "emissions":       pd.read_csv(DATA / "emissions.csv"),
    }
    params = (
        pd.read_csv(DATA / "params.csv")
          .set_index("param_id")["value"].to_dict()
    )
    d["params"] = params
    return d


# --------------------------------------------------------------------- model
def build_model(d: dict, relax_binary: bool = False) -> pyo.ConcreteModel:
    """Return a Pyomo ConcreteModel.

    relax_binary=True  →  y is continuous in [0,1]   (LP relaxation)
    relax_binary=False →  y is {0,1}                 (MILP)
    """
    m = pyo.ConcreteModel(name="memory_allocation")

    # ---- sets
    m.I = pyo.Set(initialize=d["suppliers"]["supplier_id"].tolist())
    m.J = pyo.Set(initialize=d["customers"]["customer_id"].tolist())
    m.P = pyo.Set(initialize=d["products"]["product_id"].tolist())
    m.T = pyo.Set(initialize=d["periods"]["period_id"].tolist(), ordered=True)

    # ---- parameters
    S = {(r.supplier_id, r.product_id, r.period_id): r.supply_kwe
         for r in d["supply"].itertuples()}
    D = {(r.customer_id, r.product_id, r.period_id): r.demand_kwe
         for r in d["demand"].itertuples()}
    C = {(r.supplier_id, r.customer_id, r.product_id): r.cost_per_unit
         for r in d["transport_cost"].itertuples()}
    pen = {(r.customer_id, r.product_id): r.penalty_per_unit
           for r in d["shortage_pen"].itertuples()}
    phi = {(r.customer_id, r.product_id): r.min_fill_rate
           for r in d["sla_min"].itertuples()}
    h   = {r.product_id: r.cost_per_unit_period for r in d["holding"].itertuples()}
    emis = {(r.supplier_id, r.customer_id): r.kg_co2_per_unit
            for r in d["emissions"].itertuples()}

    F   = float(d["params"]["lane_fixed_cost"])
    K   = int(d["params"]["lanes_per_fab"])
    B   = float(d["params"]["carbon_budget"])
    M   = float(d["params"]["big_M_lane"])
    I0  = float(d["params"]["initial_inventory"])

    m.S = pyo.Param(m.I, m.P, m.T, initialize=S, default=0.0)
    m.D = pyo.Param(m.J, m.P, m.T, initialize=D, default=0.0)
    m.C = pyo.Param(m.I, m.J, m.P, initialize=C, default=0.0)
    m.pi = pyo.Param(m.J, m.P, initialize=pen, default=0.0)
    m.phi = pyo.Param(m.J, m.P, initialize=phi, default=0.0)
    m.h = pyo.Param(m.P, initialize=h)
    m.e = pyo.Param(m.I, m.J, initialize=emis, default=1.0)

    # ---- variables
    m.x = pyo.Var(m.I, m.J, m.P, m.T, within=pyo.NonNegativeReals)
    m.u = pyo.Var(m.J, m.P, m.T, within=pyo.NonNegativeReals)
    m.Inv = pyo.Var(m.I, m.P, m.T, within=pyo.NonNegativeReals)

    if relax_binary:
        m.y = pyo.Var(m.I, m.J, bounds=(0.0, 1.0), within=pyo.NonNegativeReals)
    else:
        m.y = pyo.Var(m.I, m.J, within=pyo.Binary)

    # ---- objective
    def obj_rule(m):
        transport = sum(m.C[i, j, p] * m.x[i, j, p, t]
                        for i in m.I for j in m.J for p in m.P for t in m.T)
        shortage  = sum(m.pi[j, p] * m.u[j, p, t]
                        for j in m.J for p in m.P for t in m.T)
        holding   = sum(m.h[p] * m.Inv[i, p, t]
                        for i in m.I for p in m.P for t in m.T)
        lane_fix  = F * sum(m.y[i, j] for i in m.I for j in m.J)
        return transport + shortage + holding + lane_fix
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # ---- (1) supply balance with inventory continuity
    T_list = list(m.T)
    def supply_balance_rule(m, i, p, t):
        idx_t = T_list.index(t)
        prev_inv = I0 if idx_t == 0 else m.Inv[i, p, T_list[idx_t - 1]]
        return (sum(m.x[i, j, p, t] for j in m.J) + m.Inv[i, p, t]
                == m.S[i, p, t] + prev_inv)
    m.supply_bal = pyo.Constraint(m.I, m.P, m.T, rule=supply_balance_rule)

    # ---- (2) demand balance
    def demand_balance_rule(m, j, p, t):
        return sum(m.x[i, j, p, t] for i in m.I) + m.u[j, p, t] == m.D[j, p, t]
    m.demand_bal = pyo.Constraint(m.J, m.P, m.T, rule=demand_balance_rule)

    # ---- (3) SLA floor (only if phi>0)
    def sla_rule(m, j, p, t):
        if m.phi[j, p] <= 1e-9 or m.D[j, p, t] <= 1e-9:
            return pyo.Constraint.Skip
        return sum(m.x[i, j, p, t] for i in m.I) >= m.phi[j, p] * m.D[j, p, t]
    m.sla = pyo.Constraint(m.J, m.P, m.T, rule=sla_rule)

    # ---- (4) lane linking
    def lane_link_rule(m, i, j):
        return sum(m.x[i, j, p, t] for p in m.P for t in m.T) <= M * m.y[i, j]
    m.lane_link = pyo.Constraint(m.I, m.J, rule=lane_link_rule)

    # ---- (5) lane cardinality
    def lane_card_rule(m, i):
        return sum(m.y[i, j] for j in m.J) <= K
    m.lane_card = pyo.Constraint(m.I, rule=lane_card_rule)

    # ---- (6) carbon budget
    def carbon_rule(m):
        return sum(m.e[i, j] * m.x[i, j, p, t]
                   for i in m.I for j in m.J for p in m.P for t in m.T) <= B
    m.carbon = pyo.Constraint(rule=carbon_rule)

    return m


# --------------------------------------------------------------------- solve
def solve(model, suffix_duals=False):
    solver = pyo.SolverFactory("gurobi")
    if not solver.available(exception_flag=False):
        raise RuntimeError("Gurobi is not available. Install gurobipy.")
    if suffix_duals:
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        model.rc   = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    result = solver.solve(model, tee=False)
    tc = result.solver.termination_condition
    assert tc == pyo.TerminationCondition.optimal, f"solver status: {tc}"
    return result


# --------------------------------------------------------------------- extract
def extract(model, has_binary):
    I = list(model.I); J = list(model.J); P = list(model.P); T = list(model.T)
    x = {(i, j, p, t): pyo.value(model.x[i, j, p, t])
         for i in I for j in J for p in P for t in T}
    u = {(j, p, t): pyo.value(model.u[j, p, t])
         for j in J for p in P for t in T}
    Inv = {(i, p, t): pyo.value(model.Inv[i, p, t])
           for i in I for p in P for t in T}
    y = {(i, j): pyo.value(model.y[i, j]) for i in I for j in J}

    cost_trans = sum(pyo.value(model.C[i, j, p]) * x[i, j, p, t]
                     for i in I for j in J for p in P for t in T)
    cost_short = sum(pyo.value(model.pi[j, p]) * u[j, p, t]
                     for j in J for p in P for t in T)
    cost_hold  = sum(pyo.value(model.h[p]) * Inv[i, p, t]
                     for i in I for p in P for t in T)
    lane_fix   = sum(y[i, j] for i in I for j in J) * 20.0  # = F

    result = {
        "objective": pyo.value(model.obj),
        "cost_transport": cost_trans,
        "cost_shortage":  cost_short,
        "cost_holding":   cost_hold,
        "cost_lane_fixed": lane_fix,
        "x": {f"{i}|{j}|{p}|{t}": round(v, 4) for (i, j, p, t), v in x.items() if v > 1e-6},
        "u": {f"{j}|{p}|{t}": round(v, 4) for (j, p, t), v in u.items() if v > 1e-6},
        "Inv": {f"{i}|{p}|{t}": round(v, 4) for (i, p, t), v in Inv.items() if v > 1e-6},
        "y": {f"{i}|{j}": round(v, 4) for (i, j), v in y.items()},
        "open_lanes": int(round(sum(1 for v in y.values() if v > 0.5))),
    }

    # shadow prices (only meaningful for LP relaxation)
    if hasattr(model, "dual"):
        supply_shadow = {}
        for i in I:
            for p in P:
                for t in T:
                    val = model.dual[model.supply_bal[i, p, t]]
                    supply_shadow[f"{i}|{p}|{t}"] = round(val, 4)
        result["dual_supply"] = supply_shadow
        carbon_dual = model.dual[model.carbon]
        result["dual_carbon"] = round(carbon_dual, 4)

    return result


# --------------------------------------------------------------------- main
def main():
    print("=" * 72)
    print("  Multi-period memory-chip allocation  —  Pyomo + Gurobi")
    print("=" * 72)

    d = load_data()
    I = d["suppliers"]["supplier_id"].tolist()
    P = d["products"]["product_id"].tolist()
    T = d["periods"]["period_id"].tolist()

    print(f"  Instance: {len(I)} fabs × {len(d['customers'])} customers × "
          f"{len(P)} products × {len(T)} periods")

    # ---- (P1) LP relaxation — for shadow prices
    print("\n--- Solving LP relaxation (P1)  (y continuous in [0,1]) ---")
    m_lp = build_model(d, relax_binary=True)
    solve(m_lp, suffix_duals=True)
    r_lp = extract(m_lp, has_binary=False)
    print(f"  z*_LP  = {r_lp['objective']:10.2f}   "
          f"(transport {r_lp['cost_transport']:.2f}, "
          f"shortage {r_lp['cost_shortage']:.2f}, "
          f"holding {r_lp['cost_holding']:.2f}, "
          f"lane {r_lp['cost_lane_fixed']:.2f})")

    # ---- (P2) full MILP
    print("\n--- Solving MILP (P2)  (y ∈ {0,1}) ---")
    m_mi = build_model(d, relax_binary=False)
    solve(m_mi, suffix_duals=False)
    r_mi = extract(m_mi, has_binary=True)
    print(f"  z*_MILP = {r_mi['objective']:10.2f}   "
          f"(transport {r_mi['cost_transport']:.2f}, "
          f"shortage {r_mi['cost_shortage']:.2f}, "
          f"holding {r_mi['cost_holding']:.2f}, "
          f"lane {r_mi['cost_lane_fixed']:.2f})")
    print(f"  open lanes : {r_mi['open_lanes']} / {len(I) * len(d['customers'])}")

    # ---- summarise LP shadow prices (aggregated over t)
    print("\n--- LP shadow prices on supply constraints ---")
    print("    (positive value = $ saved per +1 unit of that fab's "
          "capacity in that period)")
    print(f"    {'':10s} {'period':>6s}  " + "   ".join(f"{p:>7s}" for p in P))
    total_by_ip = {(i, p): 0.0 for i in I for p in P}
    for i in I:
        for ti, t in enumerate(T):
            row = [i if ti == 0 else ""]
            row.append(t)
            for p in P:
                v = r_lp["dual_supply"][f"{i}|{p}|{t}"]
                total_by_ip[(i, p)] += v
                row.append(f"{v:7.2f}")
            print("    {:10s} {:>6s}  {}".format(row[0], row[1], "   ".join(row[2:])))
        print()

    # ---- sensitivity: scale HBM3e supply at FAB_KR1 over range
    print("--- Sensitivity scan  (HBM3e at FAB_KR1) ---")
    scan_mult = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7]
    scan = []
    import copy
    for mult in scan_mult:
        d_s = copy.deepcopy(d)
        sup = d_s["supply"]
        mask = (sup["supplier_id"] == "FAB_KR1") & (sup["product_id"] == "HBM3e")
        sup.loc[mask, "supply_kwe"] = sup.loc[mask, "supply_kwe"] * mult
        m_s = build_model(d_s, relax_binary=True)
        solve(m_s)
        r_s = extract(m_s, has_binary=False)
        # AI hyperscaler HBM3e fill rate over horizon
        dem_ai = d_s["demand"].query("customer_id=='CUS_AI' and product_id=='HBM3e'")["demand_kwe"].sum()
        ship_ai = sum(v for k, v in r_s["x"].items()
                      if "|CUS_AI|HBM3e|" in k)
        fill = ship_ai / dem_ai if dem_ai > 0 else 1.0
        scan.append({"mult": mult, "z": r_s["objective"], "fill_ai_hbm": fill})
        print(f"  KR1 HBM3e × {mult:.2f}  →  z = {r_s['objective']:9.2f}   "
              f"AI-HBM3e fill = {100*fill:5.1f} %")

    # ---- dump results
    payload = {
        "instance": {
            "suppliers": I,
            "customers": d["customers"]["customer_id"].tolist(),
            "products":  P,
            "periods":   T,
        },
        "LP": r_lp,
        "MILP": r_mi,
        "gap_pct": 100.0 * (r_mi["objective"] - r_lp["objective"]) / r_lp["objective"],
        "sensitivity": scan,
        "total_shadow_by_fab_product": {
            f"{i}|{p}": round(v, 4) for (i, p), v in total_by_ip.items()
        },
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT / 'results.json'}.")
    print("=" * 72)


if __name__ == "__main__":
    main()
