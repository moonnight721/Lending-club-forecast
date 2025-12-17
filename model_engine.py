import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def _compute_metrics(y_true, y_probs, threshold=0.5):
    try:
        y_preds = (y_probs >= threshold).astype(int)
        return {
            'AUC': roc_auc_score(y_true, y_probs),
            'F1': f1_score(y_true, y_preds),
            'Acc': accuracy_score(y_true, y_preds),
            'Prec': precision_score(y_true, y_preds, zero_division=0),
            'Rec': recall_score(y_true, y_preds, zero_division=0)
        }
    except:
        return {'AUC': 0, 'F1': 0, 'Acc': 0, 'Prec': 0, 'Rec': 0}

def _get_feature_importance(model, model_name, fold, feature_names):

    imp_df = pd.DataFrame()
    
    try:
        # 1. 樹模型 (XGB, LGBM, RF) - 這裡是 MDP (Mean Decrease Impurity)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
        # 2. 線性模型 (LogisticRegression, LinearSVC raw)
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0]) 
            
        # 3. CalibratedClassifierCV (特殊處理)
        elif hasattr(model, 'calibrated_classifiers_'):

            try:
                base_coefs = [clf.base_estimator.coef_[0] for clf in model.calibrated_classifiers_]
                importances = np.mean(np.abs(base_coefs), axis=0)
            except:
                return pd.DataFrame() # 提取失敗，回傳空
        else:
            return pd.DataFrame() # 無法提取


        if len(importances) == len(feature_names):
            imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances,
                'Model': model_name,
                'Fold': fold
            })
    except Exception as e:
        print(f"   [Info] 無法提取 {model_name} 的特徵重要性: {e}")
        
    return imp_df

def _train_and_log(fold, model, model_name, strategy, X_train, y_train, X_val, y_val, X_test, y_test):
    start_time = time.time()
    feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'f{i}' for i in range(X_train.shape[1])]

    model.fit(X_train, y_train)
    
    try:
        val_probs = model.predict_proba(X_val)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"  [Error] {model_name} Predict Failed: {e}")
        val_probs = np.zeros(len(X_val))
        test_probs = np.zeros(len(X_test))

    val_metrics = _compute_metrics(y_val, val_probs)
    test_metrics = _compute_metrics(y_test, test_probs)

    duration = time.time() - start_time
    
    print(f"  [{strategy}] {model_name} | Fold {fold} | Val AUC: {val_metrics['AUC']:.4f} | Test AUC: {test_metrics['AUC']:.4f} | Time: {duration:.2f}s")

    log_entry = {
        'Fold': fold,
        'Model': model_name,
        'Strategy': strategy,
        'Duration_Sec': duration,
        'Val_AUC': val_metrics['AUC'],
        'Val_F1': val_metrics['F1'],
        'Val_Acc': val_metrics['Acc'],
        'Val_Prec': val_metrics['Prec'],
        'Val_Rec': val_metrics['Rec'],
        'Test_AUC': test_metrics['AUC'],
        'Test_F1': test_metrics['F1'],
        'Test_Acc': test_metrics['Acc'],
        'Test_Prec': test_metrics['Prec'],
        'Test_Rec': test_metrics['Rec']
    }
    
    imp_df = _get_feature_importance(model, model_name, fold, feature_names)
    
    return log_entry, test_probs, imp_df

# 內建權重樹模型 (Weighted Tree)

def train_weighted_tree_models(fold, X_train, y_train, X_val, y_val, X_test, y_test, X_train_rf, X_val_rf, X_test_rf):
    logs = []
    test_preds_dict = {}
    importances_list = [] 

    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    models_config = [
        ('XGBoost_W', XGBClassifier(
            scale_pos_weight=scale_weight,
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            eval_metric='logloss', device='cuda', tree_method='hist',
            random_state=42, early_stopping_rounds=50
        )),
        ('LightGBM_W', LGBMClassifier(
            scale_pos_weight=scale_weight,
            n_estimators=1000, learning_rate=0.05,
            verbose=-1, random_state=42
        )),
        ('RandomForest_W', RandomForestClassifier(
            class_weight='balanced',
            n_estimators=200, n_jobs=-1, random_state=42
        ))
    ]

    for name, model in models_config:

        if 'RandomForest_W' in name:
            cur_X_tr, cur_X_val, cur_X_test = X_train_rf, X_val_rf, X_test_rf
        else:
            cur_X_tr, cur_X_val, cur_X_test = X_train, X_val, X_test

        feature_names = cur_X_tr.columns if hasattr(cur_X_tr, 'columns') else [f'f{i}' for i in range(cur_X_tr.shape[1])]


        if 'XGB' in name or 'Light' in name:
             try:
                start_time = time.time()
                model.fit(cur_X_tr, y_train, eval_set=[(cur_X_val, y_val)])
                
                val_probs = model.predict_proba(cur_X_val)[:, 1]
                test_probs = model.predict_proba(cur_X_test)[:, 1]
                
                val_metrics = _compute_metrics(y_val, val_probs)
                test_metrics = _compute_metrics(y_test, test_probs)
                
                duration = time.time() - start_time
                print(f"  [Weighted] {name} | Fold {fold} | Val AUC: {val_metrics['AUC']:.4f} | Test AUC: {test_metrics['AUC']:.4f} |Time: {duration:.2f}s")
            
                log_entry = {
                    'Fold': fold, 'Model': name, 'Strategy': 'Weighted', 'Duration_Sec': duration,
                    'Val_AUC': val_metrics['AUC'], 'Val_F1': val_metrics['F1'], 
                    'Val_Acc': val_metrics['Acc'], 'Val_Prec': val_metrics['Prec'], 'Val_Rec': val_metrics['Rec'],
                    'Test_AUC': test_metrics['AUC'], 'Test_F1': test_metrics['F1'], 
                    'Test_Acc': test_metrics['Acc'], 'Test_Prec': test_metrics['Prec'], 'Test_Rec': test_metrics['Rec']
                }
                logs.append(log_entry)
                test_preds_dict[name] = test_probs
                
                imp_df = _get_feature_importance(model, name, fold, feature_names)
                if not imp_df.empty:
                    importances_list.append(imp_df)
                
                continue
             except Exception as e:
                 print(f"Tree fitting error ({name}): {e}")
                 pass 


        log, pred, imp_df = _train_and_log(fold, model, name, 'Weighted', cur_X_tr, y_train, cur_X_val, y_val, cur_X_test, y_test)
        logs.append(log)
        test_preds_dict[name] = pred
        if not imp_df.empty:
            importances_list.append(imp_df)
        
    return logs, test_preds_dict, importances_list

# SMOTE + 樹模型 (SMOTE Tree)

def train_smote_tree_models(fold, X_train, y_train, X_val, y_val, X_test, y_test):
    logs = []
    test_preds_dict = {}
    importances_list = []

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    models_config = [
        ('XGBoost_S', XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
             eval_metric='logloss', random_state=42, device='cuda', tree_method='hist'
        )),
        ('LightGBM_S', LGBMClassifier(
            n_estimators=1000, learning_rate=0.05,
            verbose=-1, random_state=42
        )),
        ('RandomForest_S', RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=42
        ))
    ]
    for name, model in models_config:
        log, pred, imp_df = _train_and_log(fold, model, name, 'SMOTE', X_train_res, y_train_res, X_val, y_val, X_test, y_test)
        logs.append(log)
        test_preds_dict[name] = pred
        if not imp_df.empty:
            importances_list.append(imp_df)

    return logs, test_preds_dict, importances_list

# 內建權重線性模型 (Weighted Linear)

def train_weighted_linear_models(fold, X_train, y_train, X_val, y_val, X_test, y_test):
    logs = []
    test_preds_dict = {}
    importances_list = []

    l_svc = LinearSVC(class_weight='balanced', dual=False, random_state=42, max_iter=2000)
    calibrated_l_svc = CalibratedClassifierCV(l_svc) 

    models_config = [
        ('LogReg_W', LogisticRegression(
            class_weight='balanced', max_iter=1000, C=0.1, n_jobs=-1, random_state=42
        )),
        ('LinearSVC_W', calibrated_l_svc),
    ]

    for name, model in models_config:
        log, pred, imp_df = _train_and_log(fold, model, name, 'Weight_Linear', X_train, y_train, X_val, y_val, X_test, y_test)
        logs.append(log)
        test_preds_dict[name] = pred
        if not imp_df.empty:
            importances_list.append(imp_df)

    return logs, test_preds_dict, importances_list


# SMOTE + 線性模型 (SMOTE Linear)
def train_smote_linear_models(fold, X_train, y_train, X_val, y_val, X_test, y_test):
    logs= []
    test_preds_dict = {}
    importances_list = []

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    l_svc = LinearSVC(dual=False, random_state=42, max_iter=2000)
    calibrated_l_svc = CalibratedClassifierCV(l_svc) 

    models_config = [
        ('LogReg_S', LogisticRegression(
            max_iter=1000, C=0.1, n_jobs=-1, random_state=42
        )),
        ('LinearSVC_S', calibrated_l_svc),
    ]

    for name, model in models_config:
        log, pred, imp_df = _train_and_log(fold, model, name, 'SMOTE_Linear', X_train_res, y_train_res, X_val, y_val, X_test, y_test)
        logs.append(log)
        test_preds_dict[name] = pred
        if not imp_df.empty:
            importances_list.append(imp_df)

    return logs, test_preds_dict, importances_list