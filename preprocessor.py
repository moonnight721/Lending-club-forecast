import pandas as pd
from sklearn.preprocessing import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def check_missing(df,threshold=0.0, top_n = None)-> pd.DataFrame:
    missing_rate = df.isna().mean()
    missing_cnt = df.isna().sum()

    info = pd.DataFrame({
        'missing_count': missing_cnt,
        'missing_rate': missing_rate
    })
    info  = info[info['missing_count'] > 0]
    if threshold > 0.0:
        info = info[info['missing_rate'] >= threshold]
    info = info.sort_values(by='missing_rate',ascending=False)    
    if top_n is not None:
        info = info.head(top_n)

   
    info["missing_rate(%)"] = (info["missing_rate"] * 100).round(2)
    print("\n各欄位缺失資訊：由高到低")
    print(info[['missing_count','missing_rate(%)']].to_string())
    return info

def drop_high_missing (train_df,test_df,threshold=0.5):
    missing_rate =train_df.isnull().mean()
    drop_cols = missing_rate[missing_rate > threshold].index.tolist()
    print("刪除高缺失欄位：",drop_cols)
    train_df =train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols)
    print("刪除後形狀：",train_df.shape,test_df.shape)
    return train_df,test_df

def drop_columns(train_df, test_df, cols_maps: list) :
    for col in cols_maps:
        if col in train_df.columns and col in test_df.columns:
            print(f"刪除的有{col}")
            train_df = train_df.drop(columns=col, errors="ignore")
            test_df  = test_df.drop(columns=col, errors="ignore")
    return train_df, test_df

def show_categories(df) -> None:
    cat_log = df.select_dtypes(include=['object']).columns
    for col in cat_log:
        print(f"{col}: {df[col].unique()}")

def select_features(train_df,test_df,col_list:list):
    missing_train = [c for c in col_list if c not in train_df.columns]
    missing_test = [c for c in col_list if c not in test_df.columns]

    if missing_train:
        print("train缺少欄位:",missing_train)
    if missing_test:
        print("test缺少欄位:",missing_test)
    
    train_selected = train_df[col_list].copy()
    test_selected = test_df[col_list].copy()
    print("選取後形狀:",train_selected.shape,test_selected.shape)
    return train_selected,test_selected

def ordinal_encoed(train_df,test_df,mapping_dict: dict):
    for col, mapping in mapping_dict.items():
            if col in train_df.columns :
                print(f"train正在轉換欄位: {col}")
                train_df[col] = train_df[col].map(mapping)
                train_nan_count = train_df[col].isnull().sum()
                if train_nan_count > 0 :
                    print(f"waring: {col} 有 {train_nan_count}  筆資料對應不到字典，已變成 NaN")

            if col in test_df.columns:
                print(f"test正在轉換欄位: {col}")
                test_df[col] = test_df[col].map(mapping)
                test_nan_count = test_df[col].isnull().sum()
                if test_nan_count > 0:
                    print(f"waring: {col} 有 {test_nan_count}  筆資料對應不到字典，已變成 NaN")

    return train_df,test_df

def encode_features(X_train, y_train, X_val, X_test, target_cols):
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('target_enc', TargetEncoder(target_type='auto', smooth='auto', random_state=42)) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_pipeline, target_cols)
        ],
        remainder='passthrough',      
        verbose_feature_names_out=False 
    )
    
    preprocessor.set_output(transform="pandas")

    print("正在處理訓練集 (Fitting & Transforming)...")
    train_encoded = preprocessor.fit_transform(X_train, y_train)
    
    print("正在處理驗證、測試集 (Transforming)...")
    val_encoded = preprocessor.transform(X_val)
    test_encoded = preprocessor.transform(X_test)

    return train_encoded, val_encoded,test_encoded

def process_global(df, cols_fill_0=None):
    df_out = df.copy()
    if cols_fill_0:
        imputer_0 = SimpleImputer(strategy='constant', fill_value=0).set_output(transform="pandas")
        valid_cols = [c for c in cols_fill_0 if c in df_out.columns]
        if valid_cols:
            df_out[valid_cols] = imputer_0.fit_transform(df_out[valid_cols])
    return df_out


def get_native_tree_data(df_train, df_val, df_test):
    return df_train.copy(), df_val.copy(), df_test.copy()

def get_filled_data(df_train, df_val, df_test, cols_special=None, cols_median=None):
    X_train = df_train.copy()
    X_val = df_val.copy()
    X_test = df_test.copy()
    if cols_special:
        imp_999 = SimpleImputer(strategy='constant', fill_value=999).set_output(transform="pandas")
        valid_cols = [c for c in cols_special if c in X_train.columns]
        if valid_cols:
            X_train[valid_cols] = imp_999.fit_transform(X_train[valid_cols])
            X_val[valid_cols] = imp_999.transform(X_val[valid_cols])
            X_test[valid_cols] = imp_999.transform(X_test[valid_cols])

    imp_med = SimpleImputer(strategy='median').set_output(transform="pandas")

    if cols_median:
        target_cols = [c for c in cols_median if c in X_train.columns]
    else:
        target_cols = [c for c in X_train.columns if X_train[c].isnull().any()]
    if target_cols:
        X_train[target_cols] = imp_med.fit_transform(X_train[target_cols])
        X_val[target_cols] = imp_med.transform(X_val[target_cols])
        X_test[target_cols] = imp_med.transform(X_test[target_cols])

def get_filled_data(df_train, df_val, df_test, cols_special, cols_median):
    X_train = df_train.copy()
    X_val = df_val.copy()
    X_test = df_test.copy()

    if cols_special:
        imp_999 = SimpleImputer(strategy='constant', fill_value=999).set_output(transform="pandas")
        valid = [c for c in cols_special if c in X_train.columns]
        if valid:
            X_train[valid] = imp_999.fit_transform(X_train[valid])
            X_val[valid] = imp_999.transform(X_val[valid])
            X_test[valid] = imp_999.transform(X_test[valid])

    imp_med = SimpleImputer(strategy='median').set_output(transform="pandas")
    if cols_median:
        valid = [c for c in cols_median if c in X_train.columns]
        if valid:
            X_train[valid] = imp_med.fit_transform(X_train[valid])
            X_val[valid] = imp_med.transform(X_val[valid])
            X_test[valid] = imp_med.transform(X_test[valid])

    remaining_nan = [c for c in X_train.columns if X_train[c].isnull().any()]
    if remaining_nan:
        print(f"{len(remaining_nan)} 沒補到")
        print(f"   (欄位: {remaining_nan[:3]}...)")

    return X_train, X_val, X_test

def get_linear_data(df_train, df_val, df_test, cols_special=None, cols_median=None):

    X_train = df_train.copy()
    X_val = df_val.copy()
    X_test = df_test.copy()

    if cols_special:
        valid_cols = [c for c in cols_special if c in X_train.columns]
        for col in valid_cols:
            tr_max = X_train[col].max()
            X_train[col] = X_train[col].fillna(tr_max)
            X_val[col] = X_val[col].fillna(tr_max)
            X_test[col] = X_test[col].fillna(tr_max)

    imp_med = SimpleImputer(strategy='median').set_output(transform="pandas")
    if cols_median:
        target_cols = [c for c in cols_median if c in X_train.columns]
    else:
        target_cols = [c for c in X_train.columns if X_train[c].isnull().any()]

    if target_cols:
        X_train[target_cols] = imp_med.fit_transform(X_train[target_cols])
        X_val[target_cols] = imp_med.transform(X_val[target_cols])
        X_test[target_cols] = imp_med.transform(X_test[target_cols])
        
    scaler = StandardScaler()
    cols = X_train.columns
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=cols, index=X_train.index)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=cols, index=X_val.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=cols, index=X_test.index)

    return X_train, X_val, X_test
