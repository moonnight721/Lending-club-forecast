import pandas as pd
from sklearn.model_selection import train_test_split

acc_df = pd.read_csv("datasets\\accepted_2007_to_2018Q4.csv",low_memory=False)
df = acc_df.copy()
missing_count = df['issue_d'].isnull().sum()
missing_ratio = df['issue_d'].isnull().mean() * 100

print(f"issue_d 缺失值數量: {missing_count}")
print(f"issue_d 缺失值比例: {missing_ratio:.4f}%")

before_len=len(df)
df = df.dropna(subset=['issue_d'])
after_len=len(df)
print(before_len,after_len)

df['issue_d'] = pd.to_datetime(df['issue_d'],format='%b-%Y')
df = df.sort_values('issue_d').reset_index(drop=True)
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print(train_df.shape,test_df.shape)

train_out_path = 'datasets\\train.csv'
test_ot_path = 'datasets\\test.csv'

train_df.to_csv(train_out_path, index=False)
print("存好了：", train_out_path)
test_df.to_csv(test_ot_path, index=False)
print("存好了：", test_ot_path)
