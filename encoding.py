import preprocessor 
import pandas as pd
from sklearn.model_selection import KFold

train_df = pd.read_csv('datasets\\train.csv',low_memory=False)
test_df = pd.read_csv('datasets\\test.csv',low_memory=False)

missing_info = preprocessor.check_missing(train_df)
category = preprocessor.show_categories(train_df)

target_col = 'loan_status'
mapping_dict = {
    "grade":{'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7},
    "term":{' 36 months':36,' 60 months':60},
    "loan_status":{"Fully Paid":0,'In Grace Period':0,'Does not meet the credit policy. Status:Fully Paid':0,
                   'Late (16-30 days)':0,
                   "Charged Off":1,'Default':1,'Does not meet the credit policy. Status:Charged Off':1,'Late (31-120 days)':1
                   }
}
train_df,test_df = preprocessor.ordinal_encoed(train_df,test_df, mapping_dict)
print(train_df.shape, test_df.shape)
train_df.to_csv('datasets\\train.csv',index=False)
test_df.to_csv('datasets\\test.csv',index=False)

print(train_df.select_dtypes(include=['object']))