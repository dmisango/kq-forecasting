# # load_jan_actuals.py  — run once
# import sqlite3, pandas as pd

# DB_PATH = "forecasting.db"   # adjust path

# # Your raw Jan 2021 prices — replace with your actual data
# jan_data = pd.DataFrame({
#     "route"          : "NBO-MBA",
#     "departure_date" : pd.date_range("2021-01-01", "2021-01-31", freq="D"),
#     "actual_price"   : [39.47, 50.06, 55.32, 46.03, 36.24, 31.14, 30.79, 37.40, 36.16, 39.82, 36.26, 32.90, 34.41, 33.13, 33.57, 33.53, 33.49, 30.56, 31.99, 33.39, 30.97, 36.65, 32.08, 36.02, 36.96, 40.04, 36.95, 41.48, 39.51, 50.11, 47.41],   # your 31 raw prices here
#     "booking_window" : None,                    # set if known, else NULL is fine
#     "departure_month": range(1, 32),            # day-of-month proxy, or set to 1
# })

# con = sqlite3.connect(DB_PATH)
# jan_data.to_sql("actuals", con, if_exists="append", index=False)
# con.close()
# print("Inserted", len(jan_data), "Jan 2021 actuals.")

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

DB_PATH = "forecasting.db"

jan_prices = [39.47, 50.06, 55.32, 46.03, 36.24, 31.14, 30.79, 37.40, 36.16, 39.82,
              36.26, 32.90, 34.41, 33.13, 33.57, 33.53, 33.49, 30.56, 31.99, 33.39,
              30.97, 36.65, 32.08, 36.02, 36.96, 40.04, 36.95, 41.48, 39.51, 50.11,
              47.41]   # ← no trailing comma here

dates = pd.date_range("2021-01-01", periods=len(jan_prices), freq="D")

jan_data = pd.DataFrame({
    "recorded_at"    : datetime.now().isoformat(),
    "route"          : "NBO-MBA",
    "departure_date" : dates.strftime("%Y-%m-%d"),
    "actual_price"   : [float(p) for p in jan_prices],
    "booking_window" : np.nan,
    "departure_month": [int(d.month) for d in dates],
})

con = sqlite3.connect(DB_PATH)

existing = con.execute(
    "SELECT COUNT(*) FROM actuals WHERE route = 'NBO-MBA' "
    "AND departure_date BETWEEN '2021-01-01' AND '2021-01-31'"
).fetchone()[0]

if existing > 0:
    print(f"⚠️  {existing} Jan 2021 records already exist — skipping insert.")
else:
    jan_data.to_sql("actuals", con, if_exists="append", index=False)
    print(f"✅ Inserted {len(jan_data)} Jan 2021 actuals into forecasting.db")

check = pd.read_sql(
    "SELECT departure_date, actual_price FROM actuals "
    "WHERE route = 'NBO-MBA' "
    "AND departure_date BETWEEN '2021-01-01' AND '2021-01-31' "
    "ORDER BY departure_date",
    con
)
print("\nJan 2021 records now in DB:")
print(check.to_string(index=False))

con.close()
