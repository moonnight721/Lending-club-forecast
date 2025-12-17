import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import preprocessor as pp
import model_engine as me

TRAIN_PATH = 'datasets\\train.csv' 
TEST_PATH = 'datasets\\test.csv'    
TARGET_COL = 'loan_status'

COLS_Target = ['home_ownership', 'verification_status'] 

COLS_FILL_0 = [
    'open_act_il', 'open_il_12m', 'open_il_24m', 'open_rv_12m', 'open_rv_24m',
    'inq_last_12m', 'inq_last_6mths', 'total_cu_tl', 'max_bal_bc', 'open_acc_6m',
    'bc_util', 'percent_bc_gt_75', 'revol_util',
    'tot_hi_cred_lim', 'total_il_high_credit_limit', 'total_bc_limit',
    'mort_acc', 'acc_open_past_24mths',
    'pub_rec', 'pub_rec_bankruptcies', 'tax_liens', 'collections_12_mths_ex_med',
    'delinq_2yrs', 'total_acc', 'open_acc','credit_hist_years'
]
COLS_SPECIAL = ['mths_since_rcnt_il']
COLS_MEDIAN = ['annual_inc', 'dti', 'pct_tl_nvr_dlq']

N_FOLDS = 5
SEED = 42

def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    if TARGET_COL in test_df.columns:
        y_test_global = test_df[TARGET_COL]
    else:
        y_test_global = None 

    y_train_all = train_df[TARGET_COL]
    X_train_all = train_df.drop(columns=[TARGET_COL])
    X_test_all = test_df.drop(columns=[TARGET_COL], errors='ignore')

    print(" -> 執行基礎補 0 (Global Fill 0)...")
    X_train_filled_0 = pp.process_global(X_train_all, cols_fill_0=COLS_FILL_0)
    X_test_filled_0 = pp.process_global(X_test_all, cols_fill_0=COLS_FILL_0)

    all_metrics_logs = [] 
    test_predictions_sum = {} 
    all_importances = [] # [NEW] 用於收集特徵重要性

    print(f"\n{N_FOLDS}-Fold Cross Validation (Target Encoding Inside Loop) ===")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_filled_0, y_train_all)):
        print(f"\n--- [Fold {fold}] ---")
        
        X_tr_fold_raw = X_train_filled_0.iloc[train_idx]
        y_tr_fold = y_train_all.iloc[train_idx]
        X_val_fold_raw = X_train_filled_0.iloc[val_idx]
        y_val_fold = y_train_all.iloc[val_idx]
        
        X_te_fold_raw = X_test_filled_0.copy()
        
        print("   -> 執行 Target Encoding...")
        X_tr_enc, X_val_enc, X_te_enc = pp.encode_features(
            X_tr_fold_raw, y_tr_fold,  
            X_val_fold_raw,           
            X_te_fold_raw,              
            target_cols=COLS_Target
        )
        
        # A. Native Data (For XGB/LGBM)
        X_tr_nat, X_val_nat, X_te_nat = pp.get_native_tree_data(
            X_tr_enc, X_val_enc, X_te_enc
        )
        
        # B. Filled Data (For RF / SMOTE)
        X_tr_filled, X_val_filled, X_te_filled = pp.get_filled_data(
            X_tr_enc, X_val_enc, X_te_enc,
            cols_special=COLS_SPECIAL, cols_median=COLS_MEDIAN
        )
        
        # C. Linear Data (For Logistic/SVM)
        X_tr_lin, X_val_lin, X_te_lin = pp.get_linear_data(
            X_tr_enc, X_val_enc, X_te_enc,
            cols_special=COLS_SPECIAL, cols_median=COLS_MEDIAN
        )

        # 1. 內建權重樹模型 (Weighted Tree)
        logs_w, preds_w, imps_w = me.train_weighted_tree_models(
            fold, X_tr_nat, y_tr_fold, X_val_nat, y_val_fold, X_te_nat, y_test_global, 
            X_tr_filled, X_val_filled, X_te_filled
        )
        
        all_metrics_logs.extend(logs_w)
        all_importances.extend(imps_w) 
        for model_name, preds in preds_w.items():
            if model_name not in test_predictions_sum:
                test_predictions_sum[model_name] = np.zeros(len(preds))
            test_predictions_sum[model_name] += preds

        # 2. SMOTE 樹模型 (SMOTE Tree)
        logs_s, preds_s, imps_s = me.train_smote_tree_models(
            fold, X_tr_filled, y_tr_fold, X_val_filled, y_val_fold, X_te_filled, y_test_global
        )
        
        all_metrics_logs.extend(logs_s)
        all_importances.extend(imps_s) 
        for model_name, preds in preds_s.items():
            if model_name not in test_predictions_sum:
                test_predictions_sum[model_name] = np.zeros(len(preds))
            test_predictions_sum[model_name] += preds

        # 3. 線性模型 (Linear Models) - Weighted & SMOTE
        logs_l, preds_l, imps_l = me.train_weighted_linear_models(
            fold, X_tr_lin, y_tr_fold, X_val_lin, y_val_fold, X_te_lin, y_test_global
        )       
        all_metrics_logs.extend(logs_l)
        all_importances.extend(imps_l) 
        for model_name, preds in preds_l.items():
            if model_name not in test_predictions_sum:
                test_predictions_sum[model_name] = np.zeros(len(preds))
            test_predictions_sum[model_name] += preds

        logs_ls, preds_ls, imps_ls = me.train_smote_linear_models(
            fold, X_tr_lin, y_tr_fold, X_val_lin, y_val_fold, X_te_lin, y_test_global
        )
        all_metrics_logs.extend(logs_ls)
        all_importances.extend(imps_ls) 
        for model_name, preds in preds_ls.items():
            if model_name not in test_predictions_sum:
                test_predictions_sum[model_name] = np.zeros(len(preds))
            test_predictions_sum[model_name] += preds

    print("\n訓練完成")
    

    metrics_df = pd.DataFrame(all_metrics_logs)
    metrics_df.to_csv("training_metrics_report.csv", index=False, encoding='utf-8-sig')
    
    time_summary = metrics_df.groupby('Model')['Duration_Sec'].sum().sort_values(ascending=False)
    for model_name, total_sec in time_summary.items():
        mins, secs = divmod(total_sec, 60)
        print(f"{model_name:<15} : {int(mins)}分 {int(secs)}秒 ({total_sec:.2f} s)")

    print("\n模型平均表現 (依 Test AUC 排序):")
    summary = metrics_df.groupby(['Strategy', 'Model'])[['Val_AUC', 'Test_AUC', 'Test_F1','Duration_Sec']].mean()
    print(summary.sort_values(by='Test_AUC', ascending=False))


    if all_importances:
        print("\n正在計算平均特徵重要性...")
        full_imp_df = pd.concat(all_importances, axis=0)
        

        avg_imp_df = full_imp_df.groupby(['Model', 'Feature'])['Importance'].mean().reset_index()
        

        avg_imp_df = avg_imp_df.sort_values(['Model', 'Importance'], ascending=[True, False])
        
        avg_imp_df.to_csv("feature_importance_report.csv", index=False, encoding='utf-8-sig')
        print("特徵重要性已儲存: feature_importance_report.csv")
    else:
        print("\n[Warning] 未收集到任何特徵重要性資料。")


    final_submission = pd.DataFrame()
    if 'id' in test_df.columns:
        final_submission['id'] = test_df['id']
    
    for model_name, sum_preds in test_predictions_sum.items():
        avg_preds = sum_preds / N_FOLDS
        final_submission[f'{model_name}_prob'] = avg_preds
        
    final_submission.to_csv("final_test_predictions.csv", index=False)
    print(f"\n所有結果已儲存：\n1. 指標報告: training_metrics_report.csv\n2. 特徵重要性: feature_importance_report.csv\n3. 預測結果: final_test_predictions.csv")

if __name__ == "__main__":
    main()