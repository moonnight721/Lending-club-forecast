import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 讀取原始大檔
acc_df = pd.read_csv("datasets\\accepted_2007_to_2018Q4.csv", low_memory=False)
df = acc_df.copy()

# 2. 處理 issue_d 缺失值 (維持您原本的邏輯)
missing_count = df['issue_d'].isnull().sum()
missing_ratio = df['issue_d'].isnull().mean() * 100
print(f"issue_d 缺失值數量: {missing_count}")
print(f"issue_d 缺失值比例: {missing_ratio:.4f}%")

before_len = len(df)
df = df.dropna(subset=['issue_d'])
after_len = len(df)
print(f"移除 issue_d 缺失後筆數: {before_len} -> {after_len}")

target_col = 'loan_status'
print("\n正在移除 'Current'")
before_filter_len = len(df)
df = df[df[target_col] != 'Current']


df = df.reset_index(drop=True)

after_filter_len = len(df)
print(f"移除 Current 後筆數: {before_filter_len} -> {after_filter_len}")
print(f"剩餘的類別分佈:\n{df[target_col].value_counts()}")


train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    stratify=df[target_col], 
    random_state=42
)

print(f"\n訓練集形狀: {train_df.shape}")
print(f"測試集形狀: {test_df.shape}")


train_out_path = 'datasets\\train.csv'
test_ot_path = 'datasets\\test.csv'

train_df.to_csv(train_out_path, index=False)
print("存好了：", train_out_path)
test_df.to_csv(test_ot_path, index=False)
print("存好了：", test_ot_path)