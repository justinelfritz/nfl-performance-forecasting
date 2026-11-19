import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df =  pd.read_csv("raw_data_csv.csv",index_col=False)
df["WinPct"] = 100.0*df["Wins"]/df["Games"]
print(df.dropna().head(50))

df_1 = df[['WinPct','OffTotalYPG','OffPPG','DefTotalYPG','DefPPG','TurnoverDiff']].dropna()
df_1[['DefTotalYPG','DefPPG']] = -df_1[['DefTotalYPG','DefPPG']]

data_1 = df_1.values
scaler = StandardScaler()

data_sc = scaler.fit_transform(data_1)
df_sc = pd.DataFrame(data_sc, columns=df_1.columns)
#print(df_sc.head(50))

#df_sc["SumScore"] = df_sc.sum(axis=1,numeric_only=True)
score_weights = [0.8, 0.5, 0.7, 0.5, 0.7, 0.3]
weights_sum = np.sum(score_weights)
weighted_score = (df_sc * score_weights).sum(axis=1)/weights_sum
df_sc["SumScore"] = weighted_score
df_sc["Team"] = df["Team"].dropna()
print(df_sc.head(50))
df_out = df_sc.sort_values(by="SumScore")
plt.bar(df_out.Team, df_out.SumScore)
plt.xticks(rotation=90)
plt.show()