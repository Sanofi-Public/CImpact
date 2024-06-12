import pandas as pd
import src.turing_causal_impact.snowflake_api as snowflake_api
import os
import datetime as dt


def get_kpis_UC14(scenarios):
    """Get all kpis (adoption and adhrence) data for a given (group of ) brand(s) for  given markets
    scenario should be a list of dictionary like
     [{'country':"BRAZIL", brands = ["%PURAN%"]}]"""
    exceptions = ["US"]

    amer = [
        scenar["country"] for scenar in scenarios if scenar["country"] in exceptions
    ]

    # check if we need amer connection
    need_amer = len(amer) > 0
    # check if we need emea connection
    need_emea = len(amer) < len(scenarios)
    print("amer", need_amer, "emea", need_emea)

    if need_amer:
        connection_amer = snowflake_api.Connection("turing_amer_prod")
    if need_emea:
        connection_emea = snowflake_api.Connection("turing_emea_prod")

    # open the query
    if need_emea:
        with open(
            os.path.join(os.getcwd(), "src/turing_causal_impact/sql/kpis-emea.sql")
        ) as f:
            query_emea = f.read()
    # if need_amer:
    #     with open(
    #         os.path.join(os.getcwd(), "src/turing_causal_impact/sql/sales-amer-external.sql")
    #     ) as f:
    #         query_amer = f.read()

    for scenar in scenarios:
        # return df
        df = pd.DataFrame()
        country, brand = scenar["country"], scenar["target_brand"]

        if country in exceptions:
            query = query_amer
        else:
            # query = query.replace('country', country).replace('brand_sales', brand).replace('since_sales', min_date)
            query = query_emea

        print(brand)
        
        print(f"Query for {brand}")
        query_t = query.replace("%COUNTRY%", country).replace("%BRAND%", brand)
        # print(query_t)

        # get data
        if country in exceptions:
            df = connection_amer.fetch(query_t)
        else:
            df = connection_emea.fetch(query_t)

        
        # df.to_parquet(f"data/{scenar['name']}-{dt.date.today()}.parquet")
        df.to_parquet(f"data/{scenar['name']}-KPI.parquet")
