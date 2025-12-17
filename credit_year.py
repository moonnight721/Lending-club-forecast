import pandas as pd
train_df = pd.read_csv('datasets\\train.csv', low_memory=False)
test_df =pd.read_csv('datasets\\test.csv', low_memory=False)

train_df['earliest_cr_line'] = pd.to_datetime(train_df['earliest_cr_line'], format='%b-%Y')
train_df['issue_d'] = pd.to_datetime(train_df['issue_d'], format='%b-%Y')
test_df['earliest_cr_line'] = pd.to_datetime(test_df['earliest_cr_line'], format='%b-%Y')
test_df['issue_d'] = pd.to_datetime(test_df['issue_d'], format='%b-%Y')

train_df['credit_hist_years'] = (train_df['issue_d'] - train_df['earliest_cr_line']).dt.days / 365
test_df['credit_hist_years'] = (test_df['issue_d'] - test_df['earliest_cr_line']).dt.days / 365


print(test_df[['issue_d', 'earliest_cr_line', 'credit_hist_years']].head())


train_df.drop(columns=['earliest_cr_line', 'issue_d'], inplace=True)
test_df.drop(columns=['earliest_cr_line', 'issue_d'], inplace=True)

train_df.to_csv('datasets\\train.csv', index=False)
test_df.to_csv('datasets\\test.csv', index=False)
