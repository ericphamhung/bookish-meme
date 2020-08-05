df1 = pd.read_json("train_spider.json")
df2 = df1[(~df1['query'].str.contains('JOIN',regex=False))]
df2 = df2[(~df2['query'].str.contains('UNION',regex=False))]
df2 = df2[(~df2['query'].str.contains('INTERSECT',regex=False))]
df2 = df2[(~df2['query'].str.contains('EXCEPT',regex=False))]

df2.shape
