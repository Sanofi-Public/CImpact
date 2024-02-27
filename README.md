# Turing sales data


Generic package to query sales data from either Turing EMEA or AMER
Perform causal inference analysis on this data set


## To start with the repo:

* Tested for Python 3.9 & 3.11
* Install dependencies first `make install`
* Follow the steps




## Example of scenarios

A json file to be put into `.scenarios/` folder

`
{
    "name": "us-toujeo-MONARCH",
    "country": "US",
    "brands": [ "TOUJEO", "LANTUS"],
    "target_brand": "TOUJEO",
    "metric": "SUM(QUANTITY_SOLD)",
    "cutover": "2023-07-07",
    "min_date": "2023-01-01",
    "max_date": "2023-09-30",
    "frequency": "weekly"
}
   `

* name :  free format string to identify your scenario
* country : Name of the country where we need to run the analysis (e.g US, FRANCE, ...)
* brands: All the brands to be used for the analysis
* target_brand: the brand we want to run the analysis
* metric: the column of the data we need to use 
* cutover:
* min_date:
* max_date:
* frequency: weekly (US only), monthly (RoW)