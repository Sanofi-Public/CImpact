from causalimpact import CausalImpact
import pandas as pd
import numpy as np
import datetime as dt


def next_period(date, frequency="monthly"):
    """Calculate next period after cutover"""
    if frequency == "monthly":
        if date.month == 12:
            return dt.date(date.year + 1, 1, 1)
        else:
            return dt.date(date.year, date.month + 1, 1)
    else:
        return date + dt.timedelta(days=7)


def provide_analyse(scenario, df, model_args=None):
    print()
    print()

    # getting data
    brands = scenario["brands"].copy()
    brands.remove(scenario["target_brand"])
    new_columns = [scenario["target_brand"]] + brands

    data = df[df["COUNTRY"] == scenario["country"]]
    data = data[["SALES_DT", "LOCAL_BRAND_NAME", "SUM(QUANTITY_SOLD)"]]
    data["SALES_DT"] = pd.to_datetime(data["SALES_DT"])
    data["SUM(QUANTITY_SOLD)"] = data["SUM(QUANTITY_SOLD)"].astype(np.int64)
    data = data.pivot(
        index="SALES_DT", columns="LOCAL_BRAND_NAME", values="SUM(QUANTITY_SOLD)"
    )
    data = data.reindex(columns=new_columns)

    # set boundaries of analysis
    cutover_date = dt.date.fromisoformat(scenario["cutover"])
    after_cutover_date = next_period(cutover_date, scenario["frequency"])

    if "min_date" in scenario:
        # min_date = scenario['min_date']
        min_date = df[df["SALES_DT"] > dt.date.fromisoformat(scenario["min_date"])][
            "SALES_DT"
        ].min()
    else:
        min_date = data.index.min().to_pydatetime().date()

    if "max_date" in scenario:
        max_date = df[df["SALES_DT"] < dt.date.fromisoformat(scenario["max_date"])][
            "SALES_DT"
        ].max()
    else:
        max_date = data.index.max().to_pydatetime().date()

    pre_period = [str(min_date), scenario["cutover"]]
    post_period = [str(after_cutover_date), str(max_date)]

    print(pre_period, post_period)
    print()

    # calculate the analysis
    impact = CausalImpact(data, pre_period, post_period, model_args=model_args)

    return impact
