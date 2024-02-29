# Turing sales data


Generic package to query sales data from either Turing EMEA or AMER
Perform causal inference analysis on this data set


## To start with the repo:

* Tested for Python 3.9 & 3.11
* Install dependencies first `make install`
* Follow the steps


## List of current listed scenarios

[List of scenarios](./docs/scenarios.MD)

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
* brands: All the brands to be used for the analysis (these brands will set the aseline for projected analysis after the cutover time)
* target_brand: the brand we want to run the analysis
* metric: The column of the data we need to use (e.g. sales quantity)
* cutover: The time at which intervention happened
* min_date: The minimum date for the analysis (this falls before the cutover date)
* max_date: The maximum date for the analysis (this falls after the cutover date)
* frequency: weekly (US only), monthly (rest of the world) - this depends on the granularity of the target data (such as sales)

## Updating codes

### Adding or updating a new scenario file

* Create your scenario in the right folder under `scenarios/` 
* Update `docs/scenarios.MD`file
* commit to main and push to master

### Update in code structure

TBD
