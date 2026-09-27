from sfc_tfsig.data.wrds import fetch_table

df = fetch_table(
    "crsp.msf",
    columns="permno, date, prc",
    where_clause="date >= '2020-01-01'",
    limit=5,
    cache_name="wrds_test",
)

print(df.head())
