"""
Generate the input dataset for the memory-allocation MILP.

Running this script writes 10 CSV files into ./data/ that together describe a
4-week, 3-fab, 5-customer, 3-product planning instance inspired by the
2024–2026 AI-memory supercycle.  The data are synthetic but the magnitudes
and relationships reflect publicly-reported industry structure:

  • HBM3e is structurally under-supplied (Fab-KR2 has zero HBM packaging).
  • DDR5 is moderately tight.
  • DDR4 is slack.
  • Shortage penalties are an order of magnitude higher for AI customers on
    HBM than for automotive customers on DDR4.
  • Automotive & industrial customers have contractual SLA floors on DDR4.
  • A CO2 budget caps total shipping emissions.

Seed is fixed (42) so that every run produces the same dataset.
Re-run to regenerate, or edit individual CSVs by hand to try variants.
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

# ------------------------------------------------------------------ sets ---
suppliers = pd.DataFrame(
    [
        ("FAB_KR1", "Fab-KR1 (Icheon)", "KR", True),
        ("FAB_KR2", "Fab-KR2 (Hwaseong, legacy node)", "KR", False),
        ("FAB_TW",  "Fab-TW  (Hsinchu)", "TW", True),
    ],
    columns=["supplier_id", "name", "country", "has_hbm_packaging"],
)

customers = pd.DataFrame(
    [
        ("CUS_AI",   "AI-Hyperscaler",     "AI",         "US"),
        ("CUS_SRV",  "Server OEM",         "Server",     "US"),
        ("CUS_PH",   "Smartphone OEM",     "Consumer",   "CN"),
        ("CUS_AUTO", "Automotive Tier-1",  "Automotive", "DE"),
        ("CUS_IND",  "Industrial OEM",     "Industrial", "JP"),
    ],
    columns=["customer_id", "name", "tier", "country"],
)

products = pd.DataFrame(
    [
        ("HBM3e", "HBM3e stack",  "Advanced", True),
        ("DDR5",  "DDR5 DRAM",    "Current",  False),
        ("DDR4",  "DDR4 DRAM",    "Legacy",   False),
    ],
    columns=["product_id", "name", "generation", "requires_advanced_packaging"],
)

T = 4
periods = pd.DataFrame(
    [(f"W{t+1}", f"Q1-2026 week {t+1}") for t in range(T)],
    columns=["period_id", "label"],
)

# ---------------------------------------------------------- supply S_ipt ---
#     HBM3e         DDR5          DDR4
base_S = {
    "FAB_KR1": [22, 80, 120],
    "FAB_KR2": [ 0, 95, 140],   # no HBM packaging
    "FAB_TW":  [18, 60, 110],
}
# Light week-over-week variation so multi-period matters.
supply_rows = []
for supplier_id, base in base_S.items():
    for pi, product_id in enumerate(products["product_id"]):
        for t, period_id in enumerate(periods["period_id"]):
            jitter = 1.0 + 0.05 * rng.standard_normal()
            val = max(0.0, base[pi] * jitter)
            # Ramp HBM3e slightly through the quarter as new stacks qualify
            if product_id == "HBM3e" and base[pi] > 0:
                val *= 1.0 + 0.03 * t
            supply_rows.append((supplier_id, product_id, period_id, round(val, 2)))
supply = pd.DataFrame(supply_rows,
                      columns=["supplier_id", "product_id", "period_id", "supply_kwe"])

# ---------------------------------------------------------- demand D_jpt ---
base_D = {
    # HBM3e  DDR5  DDR4
    "CUS_AI":   [55, 110,  0],
    "CUS_SRV":  [ 5,  95, 40],
    "CUS_PH":   [ 0,  60, 90],
    "CUS_AUTO": [ 0,   8, 95],
    "CUS_IND":  [ 0,   5, 70],
}
demand_rows = []
for customer_id, base in base_D.items():
    for pi, product_id in enumerate(products["product_id"]):
        for t, period_id in enumerate(periods["period_id"]):
            jitter = 1.0 + 0.07 * rng.standard_normal()
            val = max(0.0, base[pi] * jitter)
            # AI demand grows through the quarter (fine-tuning campaigns)
            if customer_id == "CUS_AI" and base[pi] > 0:
                val *= 1.0 + 0.04 * t
            demand_rows.append((customer_id, product_id, period_id, round(val, 2)))
demand = pd.DataFrame(demand_rows,
                      columns=["customer_id", "product_id", "period_id", "demand_kwe"])

# -------------------------------------------------- transport cost c_ijp ---
# $0.60–0.85 per unit base, plus a product-specific packaging/freight premium.
premium = {"HBM3e": 0.40, "DDR5": 0.12, "DDR4": 0.05}
trans_rows = []
for supplier_id in suppliers["supplier_id"]:
    for customer_id in customers["customer_id"]:
        for product_id in products["product_id"]:
            c = 0.60 + 0.25 * rng.random() + premium[product_id]
            trans_rows.append((supplier_id, customer_id, product_id, round(c, 3)))
transport_cost = pd.DataFrame(trans_rows,
    columns=["supplier_id", "customer_id", "product_id", "cost_per_unit"])

# ---------------------------------------------------- shortage penalty π ---
penalty = pd.DataFrame(
    [
        # ($/unit of unmet demand; reflects willingness-to-pay of each tier)
        ("CUS_AI",   "HBM3e", 18.00),
        ("CUS_AI",   "DDR5",   6.50),
        ("CUS_SRV",  "HBM3e", 12.00),
        ("CUS_SRV",  "DDR5",   4.80),
        ("CUS_SRV",  "DDR4",   1.80),
        ("CUS_PH",   "DDR5",   3.20),
        ("CUS_PH",   "DDR4",   1.40),
        ("CUS_AUTO", "DDR5",   2.00),
        ("CUS_AUTO", "DDR4",   2.60),   # automotive contracts → high DDR4 penalty
        ("CUS_IND",  "DDR5",   1.80),
        ("CUS_IND",  "DDR4",   2.20),
    ],
    columns=["customer_id", "product_id", "penalty_per_unit"],
)

# ------------------------- minimum SLA (fill-rate) per customer-product ---
# Strategic contractual floor: automotive DDR4 must be ≥85 %, industrial DDR4 ≥75 %
# Server DDR5 ≥70 %.  All others are 0 (best-effort).
sla = pd.DataFrame(
    [
        ("CUS_AUTO", "DDR4", 0.85),
        ("CUS_IND",  "DDR4", 0.75),
        ("CUS_SRV",  "DDR5", 0.70),
    ],
    columns=["customer_id", "product_id", "min_fill_rate"],
)

# -------------------------------------------------- inventory holding h_p ---
holding = pd.DataFrame(
    [
        ("HBM3e", 0.80),   # high-value → higher holding cost
        ("DDR5",  0.30),
        ("DDR4",  0.15),
    ],
    columns=["product_id", "cost_per_unit_period"],
)

# ------------------------------------------------ emissions kg CO2 / unit ---
# Intra-region is cheap; trans-pacific shipping is expensive.  We model it
# very simply as a function of the origin/destination country pair.
region_coef = {
    ("KR", "US"): 0.9,  ("TW", "US"): 0.9,
    ("KR", "CN"): 0.3,  ("TW", "CN"): 0.25,
    ("KR", "DE"): 1.1,  ("TW", "DE"): 1.2,
    ("KR", "JP"): 0.2,  ("TW", "JP"): 0.35,
}
emissions_rows = []
sup_country = dict(zip(suppliers["supplier_id"], suppliers["country"]))
cus_country = dict(zip(customers["customer_id"], customers["country"]))
for supplier_id in suppliers["supplier_id"]:
    for customer_id in customers["customer_id"]:
        key = (sup_country[supplier_id], cus_country[customer_id])
        e = region_coef.get(key, 1.0)
        emissions_rows.append((supplier_id, customer_id, round(e, 3)))
emissions = pd.DataFrame(emissions_rows,
    columns=["supplier_id", "customer_id", "kg_co2_per_unit"])

# --------------------------------------------------- global scalar params ---
params = pd.DataFrame(
    [
        ("T_periods",              T,      "number of planning periods"),
        ("lane_fixed_cost",       20.0,    "$k per opened supplier–customer lane over horizon"),
        ("lanes_per_fab",          3,      "K_i : max customers each fab can serve"),
        ("carbon_budget",        900.0,    "kg CO2 total (across horizon, all shipments)"),
        ("initial_inventory",      0.0,    "all products start at zero stock"),
        ("big_M_lane",          5000.0,    "valid big-M for lane linking constraint"),
    ],
    columns=["param_id", "value", "description"],
)

# ---------------------------------------------------------- dump all CSVs ---
suppliers.to_csv(DATA_DIR / "suppliers.csv", index=False)
customers.to_csv(DATA_DIR / "customers.csv", index=False)
products.to_csv(DATA_DIR / "products.csv", index=False)
periods.to_csv(DATA_DIR / "periods.csv", index=False)
supply.to_csv(DATA_DIR / "supply.csv", index=False)
demand.to_csv(DATA_DIR / "demand.csv", index=False)
transport_cost.to_csv(DATA_DIR / "transport_cost.csv", index=False)
penalty.to_csv(DATA_DIR / "shortage_penalty.csv", index=False)
sla.to_csv(DATA_DIR / "sla_min.csv", index=False)
holding.to_csv(DATA_DIR / "holding_cost.csv", index=False)
emissions.to_csv(DATA_DIR / "emissions.csv", index=False)
params.to_csv(DATA_DIR / "params.csv", index=False)

print("Wrote 12 CSVs to", DATA_DIR.resolve())
for p in sorted(DATA_DIR.iterdir()):
    print(f"  {p.name:25s}  {p.stat().st_size:>6d} bytes")
