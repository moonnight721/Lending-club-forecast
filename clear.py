import pandas as pd
import preprocessor

train_df = pd.read_csv('datasets\\train.csv',low_memory=False)
test_df = pd.read_csv('datasets\\test.csv',low_memory=False)

missing_info = preprocessor.check_missing(train_df)
train_df, test_df = preprocessor.drop_high_missing(train_df, test_df, threshold=0.6)
numeric_cols = [
    "loan_amnt", "funded_amnt", "funded_amnt_inv", "int_rate", "installment",
    "annual_inc", "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "inq_last_12m", "open_acc", "total_acc", "pub_rec",
    "revol_bal", "revol_util", "collections_12_mths_ex_med",
    "pub_rec_bankruptcies", "tax_liens", "mort_acc",
    "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m",
    "mths_since_rcnt_il", "open_rv_12m", "open_rv_24m", "max_bal_bc",
    "total_cu_tl", "acc_open_past_24mths", "bc_util", "pct_tl_nvr_dlq",
    "percent_bc_gt_75", "tot_hi_cred_lim", "total_bc_limit",
    "total_il_high_credit_limit","credit_hist_years"
]
object_cols = [
    "term",
    "grade",
    "home_ownership",
    "verification_status",
    "loan_status"    
]

cols_to_keep = numeric_cols + object_cols

train_keep, test_keep = preprocessor.select_features(train_df, test_df,cols_to_keep)

train_keep.to_csv('datasets\\train.csv', index=False)
test_keep.to_csv('datasets\\test.csv', index=False)