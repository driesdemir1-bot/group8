from sqlalchemy import create_engine, text
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Hypothesis 1 (Idris)
# We test if budget relates to a few outcomes.

# NOTE: Password is hard-coded here. This is fine for a school task, but not for real projects.
engine = create_engine("postgresql+psycopg2://postgres:Idris123@localhost:5432/mydb")

def fetch_df(sql: str):
    # Small helper: run SQL and get a DataFrame
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn)

def corr_with_pvalue(x, y):
    # Drop rows with missing values in either column
    mask = x.notna() & y.notna()
    x2, y2 = x[mask], y[mask]
    n = len(x2)
    if n < 2:
        # Not enough data to calculate a correlation
        return {"n": n, "r": None, "p": None}
    # Pearson correlation and its p-value from SciPy
    r, p = pearsonr(x2, y2)
    return {"n": n, "r": float(r), "p": float(p)}

def fmt(res):
    # Pretty print of the result
    return f"N={res['n']}, r={res['r']:.3f}, p={res['p']:.4f}" if res["r"] is not None else f"N={res['n']} → no overlap"

# H1a: Budget vs Theatre Count
sql_h1a = """
SELECT 
  p.production_budget::numeric AS production_budget,   -- make sure it's numeric (not text/int)
  d.theatre_count::numeric     AS theatre_count        -- cast for consistent types
FROM public.performance p
JOIN public.distribution d USING (title, reldate)      -- join on same movie (title + date)
WHERE p.production_budget IS NOT NULL
  AND d.theatre_count IS NOT NULL;
"""
df_h1a = fetch_df(sql_h1a)
res_h1a = corr_with_pvalue(df_h1a["production_budget"], df_h1a["theatre_count"])

# H1b: Budget vs Opening Weekend Revenue
sql_h1b = """
SELECT 
  p.production_budget::numeric AS production_budget,
  COALESCE(d.opening_weekend_revenue, p.opening_weekend_revenue)::numeric AS opening_weekend_revenue
  -- COALESCE: use distribution value if it exists, otherwise use performance
FROM public.performance p
JOIN public.distribution d USING (title, reldate)
WHERE p.production_budget IS NOT NULL
  AND COALESCE(d.opening_weekend_revenue, p.opening_weekend_revenue) IS NOT NULL;
"""
df_h1b = fetch_df(sql_h1b)
res_h1b = corr_with_pvalue(df_h1b["production_budget"], df_h1b["opening_weekend_revenue"])

# H1c: Budget vs Avg Run per Theatre (now taken from distribution)
sql_h1c = """
SELECT 
  p.production_budget::numeric AS production_budget,
  d.avg_run_per_theatre::numeric AS avg_run_per_theatre  -- cast so Pandas sees numbers
FROM public.performance p
JOIN public.distribution d USING (title, reldate)
WHERE p.production_budget IS NOT NULL
  AND d.avg_run_per_theatre IS NOT NULL;
"""
df_h1c = fetch_df(sql_h1c)
res_h1c = corr_with_pvalue(df_h1c["production_budget"], df_h1c["avg_run_per_theatre"])

print("📊 Hypothesis 1 (clean tables):")
print(f"H1a: Budget ↔ Theatre Count           → {fmt(res_h1a)}")
print(f"H1b: Budget ↔ Opening Weekend Revenue → {fmt(res_h1b)}")
print(f"H1c: Budget ↔ Avg Run per Theatre     → {fmt(res_h1c)}")

# === Correlation matrix (Pearson; Spearman as extra) ===
sql_matrix = """
SELECT 
  p.production_budget::numeric                                           AS production_budget,
  d.theatre_count::numeric                                               AS theatre_count,
  COALESCE(d.opening_weekend_revenue, p.opening_weekend_revenue)::numeric AS opening_weekend_revenue,
  d.avg_run_per_theatre::numeric                                         AS avg_run_per_theatre
FROM public.performance p
JOIN public.distribution d USING (title, reldate)
WHERE p.production_budget IS NOT NULL
  AND (d.theatre_count IS NOT NULL 
       OR d.opening_weekend_revenue IS NOT NULL
       OR d.avg_run_per_theatre IS NOT NULL);
"""

df_all = fetch_df(sql_matrix)

# Make sure these columns are numeric (anything weird becomes NaN)
cols = ["production_budget", "theatre_count", "opening_weekend_revenue", "avg_run_per_theatre"]
df_all[cols] = df_all[cols].apply(pd.to_numeric, error_
