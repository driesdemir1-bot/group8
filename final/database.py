from sqlalchemy import create_engine, text
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

#code for the first hypothesis (idris)

engine = create_engine("postgresql+psycopg2://postgres:Idris123@localhost:5432/mydb")

def fetch_df(sql: str):
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn)

def corr_with_pvalue(x, y):
    mask = x.notna() & y.notna()
    x2, y2 = x[mask], y[mask]
    n = len(x2)
    if n < 2:
        return {"n": n, "r": None, "p": None}
    r, p = pearsonr(x2, y2)
    return {"n": n, "r": float(r), "p": float(p)}

def fmt(res):
    return f"N={res['n']}, r={res['r']:.3f}, p={res['p']:.4f}" if res["r"] is not None else f"N={res['n']} → no overlap"

# H1a: Budget vs Theatre Count
sql_h1a = """
SELECT 
  p.production_budget::numeric AS production_budget,
  d.theatre_count::numeric     AS theatre_count
FROM public.performance p
JOIN public.distribution d USING (title, reldate)
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
FROM public.performance p
JOIN public.distribution d USING (title, reldate)
WHERE p.production_budget IS NOT NULL
  AND COALESCE(d.opening_weekend_revenue, p.opening_weekend_revenue) IS NOT NULL;
"""
df_h1b = fetch_df(sql_h1b)
res_h1b = corr_with_pvalue(df_h1b["production_budget"], df_h1b["opening_weekend_revenue"])

# H1c: Budget vs Avg Run per Theatre (nu uit distribution)
sql_h1c = """
SELECT 
  p.production_budget::numeric AS production_budget,
  d.avg_run_per_theatre::numeric AS avg_run_per_theatre
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

# === Correlation matrix (Pearson, optioneel Spearman) ===
sql_matrix = """
SELECT 
  p.production_budget::numeric                                      AS production_budget,
  d.theatre_count::numeric                                          AS theatre_count,
  COALESCE(d.opening_weekend_revenue, p.opening_weekend_revenue)::numeric AS opening_weekend_revenue,
  d.avg_run_per_theatre::numeric                                    AS avg_run_per_theatre
FROM public.performance p
JOIN public.distribution d USING (title, reldate)
WHERE p.production_budget IS NOT NULL
  AND (d.theatre_count IS NOT NULL 
       OR d.opening_weekend_revenue IS NOT NULL
       OR d.avg_run_per_theatre IS NOT NULL);
"""

df_all = fetch_df(sql_matrix)

# Alleen relevante kolommen als numeriek (zou al numeric moeten zijn, maar for safety):
cols = ["production_budget", "theatre_count", "opening_weekend_revenue", "avg_run_per_theatre"]
df_all[cols] = df_all[cols].apply(pd.to_numeric, errors="coerce")

# Pearson (pairwise complete)
corr_pearson = df_all[cols].corr(method="pearson").round(3)
print("\n📐 Correlation matrix (Pearson):")
print(corr_pearson)

# Optioneel: Spearman (robuuster bij scheve verdelingen)
corr_spearman = df_all[cols].corr(method="spearman").round(3)
print("\n📐 Correlation matrix (Spearman):")
print(corr_spearman)

# Heatmap Pearson
plt.figure(figsize=(8,6))
sns.heatmap(corr_pearson, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlation Matrix (Pearson)", fontsize=14)
plt.tight_layout()
plt.show()