import pandas as pd
import plotly.express as px

data=pd.read_csv('test.csv')
# print(data)

df=px.data.tips()
print(df.head())

import plotly.express as px

long_df = px.data.medals_long()
print(long_df)
fig = px.bar(long_df, x="nation", y="count", color="medal", title="Long-Form Input")
fig.show()