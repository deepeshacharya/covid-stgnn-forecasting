# Data Dictionary — CDC Community Profile Report

## Source
CDC Community Profile Report (HHS)
URL: https://healthdata.gov/Health/COVID-19-Community-Profile-Report/gqxm-d9w9

## Coverage
- Geography : US counties (FIPS level), 3,221 unique counties
- Date range: February 2021 to May 2023
- Frequency : Near-daily snapshots (some gaps exist)
- Files     : 232 CSV files in data/raw/

## Key Columns

| Normalized name                        | Description                                      |
|----------------------------------------|--------------------------------------------------|
| fips                                   | 5-digit county FIPS code (zero-padded string)    |
| county                                 | County name                                      |
| state                                  | 2-letter state abbreviation (uppercase)          |
| date                                   | Report date                                      |
| casesper100klast7days                  | PRIMARY TARGET: 7-day rolling case rate /100k    |
| caseslast7days                         | Total new cases last 7 days                      |
| totalcases                             | Cumulative total cases                           |
| deathslast7days                        | New deaths last 7 days                           |
| deathsper100klast7days                 | 7-day death rate per 100k                        |
| totaldeaths                            | Cumulative total deaths                          |
| testpositivityratelast7days            | 7-day test positivity rate                       |
| confirmedcovidhosplast7days            | Confirmed COVID hospitalizations last 7 days     |
| confirmedcovidhospper100bedslast7days  | Confirmed COVID hosp per 100 beds                |
| suspectedcovidhosplast7days            | Suspected COVID hospitalizations last 7 days     |
| suspectedcovidhospper100bedslast7days  | Suspected COVID hosp per 100 beds                |
| pctinpatientbedsusedavglast7days       | Pct inpatient beds occupied (all), 7-day avg     |
| pctinpatientbedsusedcovidavglast7days  | Pct inpatient beds occupied by COVID, 7-day avg  |
| pcticubedsusedavglast7days             | Pct ICU beds occupied (all), 7-day avg           |
| pcticubedsusedcovidavglast7days        | Pct ICU beds occupied by COVID, 7-day avg        |

## County Adjacency
Source : US Census Bureau County Adjacency File (2023)
URL    : https://www2.census.gov/geo/docs/reference/county_adjacency.txt
Pairs after filtering: 2,794 total | 2,157 intra-state | 637 inter-state

## Important Notes
1. The target casesper100klast7days is a 7-day rolling average.
   Consecutive values share 6/7 days of raw data (autocorrelation ~0.97).
2. Isolated states (PR, AK, HI, GU, VI, MP, AS) have limited adjacency
   and are excluded from inter-state spillover analysis.
3. Rows with county = Unallocated are removed during preprocessing.
4. Duplicate date files (e.g. 2021-03-12.csv and 2021-03-12(2).csv)
   are deduplicated keeping the last record per (date, fips).
