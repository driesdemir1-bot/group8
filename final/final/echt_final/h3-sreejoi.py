# h3_analysis.py
# Hypothesis (H3): Roy
# Positive user reviews sustain run length post-opening,
# whereas critic reviews influence initial theatre count and opening weekend revenue.

from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ---- DB connection ----
engine = create_engine("postgresql+psycopg2://tristanriethorst@localhost:5432/mydb")

def fetch_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})

def table_has_column(schema: str, table: str, column: str) -> bool:
    sql = """
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = :schema AND table_name = :table AND column_name = :column
    """
    df = fetch_df(sql, {"schema": schema, "table": table, "column": column})
    return not df.empty

# ---- 1) Extraction functions ----
def get_movies() -> pd.DataFrame:
    sql = """
    SELECT movie_id, url, title, reldate
    FROM public.movie
    WHERE url IS NOT NULL AND title IS NOT NULL AND reldate IS NOT NULL;
    """
    return fetch_df(sql)

def get_distribution() -> pd.DataFrame:
    """
    Pull distribution by movie_id. If 'avg_run_per_theatre' doesn't exist,
    compute a fallback as opening_weekend_revenue / theatre_count (avoids div by 0).
    """
    has_avg = table_has_column("public", "distribution", "avg_run_per_theatre")

    if has_avg:
        sql = """
        SELECT
          d.movie_id,
          d.theatre_count::numeric                       AS theatre_count,
          d.opening_weekend_revenue::numeric             AS owr_dist,
          d.avg_run_per_theatre::numeric                 AS avg_run_per_theatre
        FROM public.distribution d
        WHERE d.theatre_count IS NOT NULL
          AND d.opening_weekend_revenue IS NOT NULL;
        """
    else:
        sql = """
        SELECT
          d.movie_id,
          d.theatre_count::numeric                                        AS theatre_count,
          d.opening_weekend_revenue::numeric                              AS owr_dist,
          (d.opening_weekend_revenue::numeric / NULLIF(d.theatre_count::numeric, 0)) AS avg_run_per_theatre
        FROM public.distribution d
        WHERE d.theatre_count IS NOT NULL
          AND d.opening_weekend_revenue IS NOT NULL;
        """
    return fetch_df(sql)

def get_performance() -> pd.DataFrame:
    sql = """
    SELECT
      movie_id,
      production_budget::numeric      AS production_budget,
      worldwide_box_office::numeric   AS worldwide_box_office,
      opening_weekend_revenue::numeric AS owr_perf
    FROM public.performance
    WHERE production_budget IS NOT NULL
      AND worldwide_box_office IS NOT NULL
      AND opening_weekend_revenue IS NOT NULL;
    """
    return fetch_df(sql)

def get_expert_reviews_windowed(days_before:int=7, days_after:int=3) -> pd.DataFrame:
    """
    Aggregate critic scores per movie_id within [reldate - days_before, reldate + days_after].
    """
    sql = """
    SELECT
      er.movie_id,
      AVG(er.idvscore)::float AS criticscore_mean,
      COUNT(er.idvscore)::int AS critics_n
    FROM public.expert_reviews er
    JOIN public.movie m ON m.movie_id = er.movie_id
    WHERE er.idvscore IS NOT NULL
      AND er."date" BETWEEN (m.reldate - (:before || ' days')::interval)
                         AND (m.reldate + (:after  || ' days')::interval)
    GROUP BY er.movie_id;
    """
    return fetch_df(sql, {"before": days_before, "after": days_after})

def get_user_reviews_windowed(days_start_after:int=3, days_end_after:int=28) -> pd.DataFrame:
    """
    Aggregate user scores per movie_id within (reldate + start_after, reldate + end_after].
    """
    sql = """
    SELECT
      ur.movie_id,
      AVG(ur.idvscore)::float AS userscore_mean,
      COUNT(ur.idvscore)::int AS users_n
    FROM public.user_reviews ur
    JOIN public.movie m ON m.movie_id = ur.movie_id
    WHERE ur.idvscore IS NOT NULL
      AND ur.datep >  (m.reldate + (:start_after || ' days')::interval)
      AND ur.datep <= (m.reldate + (:end_after   || ' days')::interval)
    GROUP BY ur.movie_id;
    """
    return fetch_df(sql, {"start_after": days_start_after, "end_after": days_end_after})

# ---- 2) Build H3 dataset ----
def build_dataset_h3(
    critics_before:int=7, critics_after:int=3,
    users_start_after:int=3, users_end_after:int=28
) -> pd.DataFrame:
    m    = get_movies()
    d    = get_distribution()
    p    = get_performance()
    crit = get_expert_reviews_windowed(critics_before, critics_after)
    usr  = get_user_reviews_windowed(users_start_after, users_end_after)

    # Merge by movie_id across everything; keep movie metadata for readability
    df = (m
          .merge(d, on="movie_id", how="inner")
          .merge(p, on="movie_id", how="left")
          .merge(crit, on="movie_id", how="left")
          .merge(usr, on="movie_id", how="left"))

    # Prefer opening_weekend_revenue from distribution; fallback to performance if missing
    if "owr_dist" in df.columns and "owr_perf" in df.columns:
        df["opening_weekend_revenue"] = df["owr_dist"].fillna(df["owr_perf"])
    elif "owr_dist" in df.columns:
        df["opening_weekend_revenue"] = df["owr_dist"]
    else:
        df["opening_weekend_revenue"] = df["owr_perf"]

    # Keep non-null & positive where needed (for logs/ratios)
    for col in ["theatre_count", "opening_weekend_revenue", "production_budget",
                "avg_run_per_theatre", "worldwide_box_office"]:
        if col in df.columns:
            df = df[df[col].notna()]

    df = df[(df["theatre_count"] > 0) &
            (df["opening_weekend_revenue"] > 0) &
            (df["production_budget"] > 0) &
            (df["avg_run_per_theatre"] > 0) &
            (df["worldwide_box_office"] > 0)]

    # Sustaining proxy: multiplier after opening
    df["mult_after_opening"] = df["worldwide_box_office"] / df["opening_weekend_revenue"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mult_after_opening"])
    df = df[df["mult_after_opening"] > 0]

    # Log transforms (skewed)
    df["log_theatre_count"] = np.log(df["theatre_count"])
    df["log_opening_weekend_revenue"] = np.log(df["opening_weekend_revenue"])
    df["log_production_budget"] = np.log(df["production_budget"])
    df["log_avg_run_per_theatre"] = np.log(df["avg_run_per_theatre"]).replace([-np.inf, np.inf], np.nan)
    df["log_mult_after_opening"] = np.log(df["mult_after_opening"]).replace([-np.inf, np.inf], np.nan)

    return df

# ---- 3) Analysis helpers ----
def print_and_plot_corr_matrices(df: pd.DataFrame):
    # Opening block: critics with initial outcomes
    open_cols = ["criticscore_mean", "log_theatre_count", "log_opening_weekend_revenue", "log_production_budget"]
    df_open = df.dropna(subset=open_cols)
    corr_open = df_open[open_cols].corr()
    print("\n=== Correlation matrix (OPENING / Pearson) ===")
    print(corr_open)
    plot_corr_matrix(df_open, open_cols, "corr_opening.png")

    # Sustaining block: users with sustaining outcomes
    sustain_cols = ["userscore_mean", "log_avg_run_per_theatre", "log_mult_after_opening", "log_production_budget"]
    df_sus = df.dropna(subset=sustain_cols)
    corr_sustain = df_sus[sustain_cols].corr()
    print("\n=== Correlation matrix (SUSTAINING / Pearson) ===")
    print(corr_sustain)
    plot_corr_matrix(df_sus, sustain_cols, "corr_sustaining.png")

    # Which aligns most with critic score (opening block)
    if "criticscore_mean" in corr_open.columns:
        c = corr_open["criticscore_mean"].drop(labels=["criticscore_mean"])
        order = c.abs().sort_values(ascending=False)
        print("\n— Which variable aligns most with critic score (opening block)? (by |corr|)")
        for k, v in order.items():
            print(f"   {k}: {v:.3f}")
    print("\nSaved: corr_opening.png, corr_sustaining.png")

def partial_corr(df, x, y, z):
    """
    Pearson partial correlation of x and y, controlling for z (residual method).
    Returns (r, p, n).
    """
    d = df[[x, y, z]].dropna()
    if len(d) < 3:
        return np.nan, np.nan, len(d)
    Xz = sm.add_constant(d[[z]])
    rx = sm.OLS(d[x], Xz).fit().resid
    ry = sm.OLS(d[y], Xz).fit().resid
    r, p = pearsonr(rx, ry)
    return r, p, len(d)

def run_models_h3(df: pd.DataFrame):
    def ols(y, Xcols, data, label):
        d = data[[y] + Xcols].dropna()
        X = sm.add_constant(d[Xcols])
        model = sm.OLS(d[y], X).fit()
        print(f"\n=== OLS: {label} ===")
        print(model.summary())
        return model

    # Opening models: critics + budget
    ols("log_theatre_count", ["criticscore_mean", "log_production_budget"], df, "log(theatres) ~ critics + log(budget)")
    ols("log_opening_weekend_revenue", ["criticscore_mean", "log_production_budget"], df, "log(opening) ~ critics + log(budget)")

    # Sustaining models: users + budget
    ols("log_avg_run_per_theatre", ["userscore_mean", "log_production_budget"], df, "log(avg_run_per_theatre) ~ users + log(budget)")
    ols("log_mult_after_opening", ["userscore_mean", "log_production_budget"], df, "log(mult_after_opening) ~ users + log(budget)")

def plot_corr_matrix(df, cols, outfile="correlation_matrix.png"):
    corr = df[cols].corr()
    fig, ax = plt.subplots()
    cax = ax.matshow(corr, vmin=-1, vmax=1)  # default colormap, fixed scale
    fig.colorbar(cax)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="left")
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", va='center', ha='center')
    plt.title("Correlation matrix (Pearson)", pad=20)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

# ---- 4) Console summary of partials per hypothesis sides ----
def report_partials(df: pd.DataFrame):
    print("\n=== Partial correlations (budget-adjusted) ===")
    # Opening side (expect critics to matter):
    r, p, n = partial_corr(df, "criticscore_mean", "log_theatre_count", "log_production_budget")
    print(f"Critics ⟂ log(theatres) | log(budget): r={r:.3f}  p={p:.4g}  n={n}")
    r, p, n = partial_corr(df, "criticscore_mean", "log_opening_weekend_revenue", "log_production_budget")
    print(f"Critics ⟂ log(opening)  | log(budget): r={r:.3f}  p={p:.4g}  n={n}")

    # Sustaining side (expect users to matter):
    r, p, n = partial_corr(df, "userscore_mean", "log_avg_run_per_theatre", "log_production_budget")
    print(f"Users   ⟂ log(avg_run)  | log(budget): r={r:.3f}  p={p:.4g}  n={n}")
    r, p, n = partial_corr(df, "userscore_mean", "log_mult_after_opening", "log_production_budget")
    print(f"Users   ⟂ log(mult>open)| log(budget): r={r:.3f}  p={p:.4g}  n={n}")

# ---- 5) Main ----
if __name__ == "__main__":
    # Build dataset with default windows:
    # Critics: [reldate-7d, reldate+3d]
    # Users:   (reldate+3d,  reldate+28d]
    df = build_dataset_h3()

    print(f"Rows in base (post-join) dataset: {len(df)}")

    # Correlation matrices + heatmaps
    print_and_plot_corr_matrices(df)

    # Partial correlations (budget-adjusted)
    report_partials(df)

    # OLS models
    run_models_h3(df)

    print("\nDone. Saved heatmaps: corr_opening.png, corr_sustaining.png")
