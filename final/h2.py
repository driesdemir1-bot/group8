# visualize_hypothesis.py
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm
import matplotlib.pyplot as plt

# --- DB connection (use your working credentials here) ---
engine = create_engine("postgresql+psycopg2://tristanriethorst@localhost:5432/mydb")

def fetch_df(sql: str):
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn)

# --- Build dataset (kept simple) ---
def build_dataset():
    sql = """
    WITH critics AS (
        SELECT
            er.url,
            AVG(er.idvscore)::float AS criticscore_mean,
            COUNT(er.idvscore)      AS critics_n
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
      ON m.title = d.title AND m.reldate = d.reldate
    LEFT JOIN public.performance p
      ON m.title = p.title AND m.reldate = p.reldate
    JOIN critics c
      ON m.url = c.url
    WHERE d.theatre_count IS NOT NULL
      AND d.opening_weekend_revenue IS NOT NULL
      AND p.production_budget IS NOT NULL
      AND c.criticscore_mean IS NOT NULL;
    """
    df = fetch_df(sql)

    # Clean & transforms
    for col in ["theatre_count", "opening_weekend_revenue", "production_budget"]:
        df = df[df[col] > 0]
        df[f"log_{col}"] = np.log(df[col])

    # Keep only necessary columns for clarity
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
    Xz = sm.add_constant(df[z])
    rx = sm.OLS(df[x], Xz).fit().resid
    ry = sm.OLS(df[y], Xz).fit().resid
    r, p = pearsonr(rx, ry)
    return r, p, rx, ry  # also return residuals for plotting

def simple_reg_summary(df, y, xlist):
    X = sm.add_constant(df[xlist])
    model = sm.OLS(df[y], X).fit()
    return model

# --- Visualization helpers (no seaborn, one figure per chart) ---
def scatter_with_trend(x, y, x_label, y_label, title, outfile):
    plt.figure()
    plt.scatter(x, y, alpha=0.7)
    # simple linear fit line
    if len(x) >= 2:
        coeffs = np.polyfit(x, y, 1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ys = coeffs[0] * xs + coeffs[1]
        plt.plot(xs, ys)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def plot_corr_matrix(df, cols, outfile="correlation_matrix.png"):
    """Save a correlation-matrix heatmap (matplotlib-only) with annotated values."""
    corr = df[cols].corr()

    fig, ax = plt.subplots()
    cax = ax.matshow(corr, vmin=-1, vmax=1)  # default colormap; centered scale
    fig.colorbar(cax)

    # Tick labels
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="left")
    ax.set_yticklabels(cols)

    # Annotate each cell with the value
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
    corr = df[cols_for_corr].corr()
    print("\n=== Correlation matrix (Pearson) ===")
    print(corr)

    # 1b) Save heatmap
    plot_corr_matrix(df, cols_for_corr, outfile="correlation_matrix.png")

    # 2) Simple scatter + trendlines
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

    # 3) Partial (residual-vs-residual) plots controlling for budget
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

    # 4) Simple regressions for reference
    m1 = simple_reg_summary(df, "log_theatre_count", ["criticscore_mean", "log_production_budget"])
    m2 = simple_reg_summary(df, "log_opening_weekend_revenue", ["criticscore_mean", "log_production_budget"])

    print("\n=== OLS: log(Theatre count) ~ critics + log(budget) ===")
    print(m1.summary())

    print("\n=== OLS: log(Opening weekend) ~ critics + log(budget) ===")
    print(m2.summary())

if __name__ == "__main__":
    df = build_dataset()
    print(f"Rows in analysis dataset: {len(df)}")
    visualize(df)
    print("\nSaved figures:")
    print(" - correlation_matrix.png")
    print(" - scatter_critics_vs_log_theatres.png")
    print(" - scatter_critics_vs_log_opening.png")
    print(" - partial_resid_critics_vs_log_theatres.png")
    print(" - partial_resid_critics_vs_log_opening.png")
