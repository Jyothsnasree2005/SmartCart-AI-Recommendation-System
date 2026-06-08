import pandas as pd
from database import engine

df = pd.read_sql(
    "SELECT COUNT(*) as total FROM customers",
    engine
)

print(df)