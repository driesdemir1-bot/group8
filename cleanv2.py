#!/usr/bin/env python3
import re
import pandas as pd
import janitor  # requires: pip install pyjanitor

INPUT_XLSX = "sales.xlsx"
OUTPUT_CSV = "sales_cleanv2.csv"

def clean_values(df):
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].apply(
                lambda v: (
                    str(v).replace("\r", " ").replace("\n", " ").strip()
                    .replace('"', "")        # remove double quotes
                    .lstrip("|").strip()     # remove leading |
                ) if isinstance(v, str) else v
            )
            # remove empty strings -> NULL
            df[col] = df[col].replace({"": pd.NA})

    # --- fix for title column: remove consecutive dots like "..", "...", etc. ---
    if "title" in df.columns:
        # remove all runs of 2+ dots
        df["title"] = df["title"].astype(str).str.replace(r"\.{2,}", "", regex=True).str.strip()
        df["title"] = df["title"].replace({"": pd.NA})
        # If you prefer to collapse to a single dot instead, use:
        # df["title"] = df["title"].astype(str).str.replace(r"\.{2,}", ".", regex=True).str.strip()

    return df

def main():
    df = pd.read_excel(INPUT_XLSX)

    # janitor: normalize headers and drop fully empty rows/columns
    df = (
        df
        .clean_names()   # lowercase, snake_case, safe chars
        .remove_empty()  # drop completely empty rows/cols
    )

    df = clean_values(df)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8", sep=",", na_rep="NULL")
    print(f"✅ Cleaned file saved as: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
