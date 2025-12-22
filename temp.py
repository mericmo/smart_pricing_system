import pandas as pd


hsr_df = pd.read_csv("raw/historical_transactions.csv", encoding='utf-8', parse_dates=['日期', '交易时间'],
                     dtype={"商品编码": str, "门店编码": str, "流水单号": str, "会员id": str})

print(hsr_df.columns)
print(len(hsr_df))
df = hsr_df.copy()
# df = df[(df['商品编码'] == '8006144') ]# (df['门店编码'] == '205625') & & (df['销售数量'] > 0)& (df["销售金额"] > 0) & (df["售价"] > 0) #8006144 4701098
df = df[(df['门店编码'] == '205625') & (df['商品编码'] == '8006144') & (df['销售数量'] > 0) ]
print("过滤后的记录数：", len(df))
