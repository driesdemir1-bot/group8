# visualize_hypothesis.py - Tristan
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm
import matplotlib.pyplot as plt

engine = create_engine("postgresql+psycopg2://tristanriethorst@localhost:5432/mydb")

def fetch_df(sql: str):
    # Helper to run a SQL string and return a pandas DataFrame
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn)

# --- Build dataset (kept simple) ---
def build_dataset():
    # We make a small dataset for the plots.
    # The CTE "critics" averages expert review scores per movie URL.
    # AVG(...)::float forces a numeric type we can use in pandas.
    sql = """
    WITH critics AS (
        SELECT
            er.url,
            AVG(er.idvscore)::float AS criticscore_mean,   -- mean critic score per movie
            COUNT(er.idvscore)      AS critics_n           -- how many reviews were used
        FROM public.expert_reviews er
        WHERE er.idvscore IS NOT NULL
        GROUP BY er.url
    )
    SELECT
        m.url,
        m.title,
        m.reldate,
        d.theatre_count,
        d.opening_weekend_revenue,
        p.production_budget,
        c.criticscore_mean,
        c.critics_n
    FROM public.movie m
    JOIN public.distribution d
      ON m.title = d.title AND m.reldate = d.reldate      -- join on the same movie (natural key)
    LEFT JOIN public.performance p
      ON m.title = p.title AND m.reldate = p.reldate      -- LEFT JOIN in case some movies miss performance
    JOIN critics c
      ON m.url = c.url                                    -- attach the critic averages by URL
    WHERE d.theatre_count IS NOT NULL
      AND d.opening_weekend_revenue IS NOT NULL
      AND p.production_budget IS NOT NULL
      AND c.criticscore_mean IS NOT NULL;
    """
    df = fetch_df(sql)

    # Clean & transforms:
    # - remove non-positive values (log needs > 0)
    # - create log-versions of a few variables for nicer relationships
    for col in ["theatre_count", "opening_weekend_revenue", "production_budget"]:
        df = df[df[col] > 0]
        df[f"log_{col}"] = np.log(df[col])                -- natural log transform

    # Keep only the columns we need for the rest (keeps the DataFrame tidy)
    cols = [
        "url", "title", "reldate",
        "theatre_count", "opening_weekend_revenue",
        "production_budget", "criticscore_mean",
        "log_theatre_count", "log_opening_weekend_revenue", "log_production_budget"
    ]
    return df[cols].copy()

# --- Stats helpers ---
def partial_corr(df, x, y, z):
    """Pearson partial correlation of x and y controlling for z (residual method)."""
    # We regress x ~ z and y ~ z, then correlate the residuals.
    # This “controls for” z in a simple and readable way.
    Xz = sm.add_constant(df[z])          # add constant term for OLS
    rx = sm.OLS(df[x], Xz).fit().resid   # residuals of x after removing z
    ry = sm.OLS(df[y], Xz).fit().resid   # residuals of y after removing z
    r, p = pearsonr(rx, ry)              # correlation between the two residual series
    return r, p, rx, ry                  # return residuals too so we can plot them

def simple_reg_summary(df, y, xlist):
    # Small wrapper to fit an OLS model and return the summary object
    X = sm.add_constant(df[xlist])
    model = sm.OLS(df[y], X).fit()
    return model

# --- Visualization helpers (no seaborn, one figure per chart) ---
def scatter_with_trend(x, y, x_label, y_label, title, outfile):
    plt.figure()
    plt.scatter(x, y, alpha=0.7)         # simple scatter
    # Add a straight trend line using numpy.polyfit (degree=1)
    if len(x) >= 2:
        coeffs = np.polyfit(x, y, 1)     # [slope, intercept]
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ys = coeffs[0] * xs + coeffs[1]
        plt.plot(xs, ys)                 # draw the fitted line
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)        # save to file so we can include in the report
    plt.close()

def plot_corr_matrix(df, cols, outfile="correlation_matrix.png"):
    """Save a correlation-matrix heatmap (matplotlib-only) with annotated values."""
    # We calculate a Pearson correlation matrix with pandas, then draw it with matshow.
    corr = df[cols].corr()

    fig, ax = plt.subplots()
    cax = ax.matshow(corr, vmin=-1, vmax=1)  # fix color scale so -1..1 is consistent
    fig.colorbar(cax)

    # Axis labels (rotate x labels so they don’t overlap)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="left")
    ax.set_yticklabels(cols)

    # Write the numeric value into each cell
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", va='center', ha='center', color="black")

    plt.title("Correlation matrix (Pearson)", pad=20)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"✅ Correlation matrix saved to {outfile}")

def visualize(df):
    # 1) Correlation matrix (print to console)
    cols_for_corr = [
        "criticscore_mean",
        "log_theatre_count",
        "log_opening_weekend_revenue",
        "log_production_budget"
    ]
    corr = df[cols_for_corr].corr()   # quick check of pairwise relationships
    print("\n=== Correlation matrix (Pearson) ===")
    print(corr)

    # 1b) Save heatmap figure to a PNG
    plot_corr_matrix(df, cols_for_corr, outfile="correlation_matrix.png")

    # 2) Simple scatter + trendlines (helps to see direction and spread)
    scatter_with_trend(
        x=df["criticscore_mean"].values,
        y=df["log_theatre_count"].values,
        x_label="Critic score (mean)",
        y_label="log(Theatre count)",
        title="Critic score vs log(Theatre count)",
        outfile="scatter_critics_vs_log_theatres.png"
    )

    scatter_with_trend(
        x=df["criticscore_mean"].values,
        y=df["log_opening_weekend_revenue"].values,
        x_label="Critic score (mean)",
        y_label="log(Opening weekend revenue)",
        title="Critic score vs log(Opening weekend revenue)",
        outfile="scatter_critics_vs_log_opening.png"
    )

    # 3) Partial plots controlling for budget:
    # We check if the link remains after removing the effect of budget.
    r_theatres, p_theatres, rx_t, ry_t = partial_corr(
        df, "criticscore_mean", "log_theatre_count", "log_production_budget"
    )
    print(f"\nPartial corr (critics ⟂ theatres | budget): r={r_theatres:.3f}, p={p_theatres:.4f}")

    scatter_with_trend(
        x=rx_t,
        y=ry_t,
        x_label="Critic score residual (| log budget)",
        y_label="log(Theatre count) residual (| log budget)",
        title="Partial relationship: critics ↔ log theatres (| log budget)",
        outfile="partial_resid_critics_vs_log_theatres.png"
    )

    r_open, p_open, rx_o, ry_o = partial_corr(
        df, "criticscore_mean", "log_opening_weekend_revenue", "log_production_budget"
    )
    print(f"Partial corr (critics ⟂ opening | budget):  r={r_open:.3f}, p={p_open:.4f}")

    scatter_with_trend(
        x=rx_o,
        y=ry_o,
        x_label="Critic score residual (| log budget)",
        y_label="log(Opening weekend) residual (| log budget)",
        title="Partial relationship: critics ↔ log opening (| log budget)",
        outfile="partial_resid_critics_vs_log_opening.png"
    )

    # 4) Simple OLS regressions (text output)
    # Model 1: theatres explained by critics + budget
    m1 = simple_reg_summary(df, "log_theatre_count", ["criticscore_mean", "log_production_budget"])
    # Model 2: opening weekend explained by critics + budget
    m2 = simple_reg_summary(df, "log_opening_weekend_revenue", ["criticscore_mean", "log_production_budget"])

    print("\n=== OLS: log(Theatre count) ~ critics + log(budget) ===")
    print(m1.summary())

    print("\n=== OLS: log(Opening weekend) ~ critics + log(budget) ===")
    print(m2.summary())

if __name__ == "__main__":
    df = build_dataset()
    print(f"Rows in analysis dataset: {len(df)}")   # quick size check
    visualize(df)
    print("\nSaved figures:")
    print(" - correlation_matrix.png")
    print(" - scatter_critics_vs_log_theatres.png")
    print(" - scatter_critics_vs_log_opening.png")
    print(" - partial_resid_critics_vs_log_theatres.png")
    print(" - partial_resid_critics_vs_log_opening.png")
