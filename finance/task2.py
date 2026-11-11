import pandas as pd
import numpy as np

df = pd.read_csv('sales_data.csv')

corr = df.select_dtypes(include=np.number).corr()
print(corr)

sales = df.groupby('Region').agg({'Sales_Amount': 'sum'})
print(sales)

max_sales = df.groupby('Sales_Rep').agg({'Sales_Amount': 'max'})
print(max_sales)

mean_unit_price = df.groupby('Product_Category').agg({'Unit_Price': 'mean'})
print(mean_unit_price)