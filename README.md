# Lending Club Credit Risk Prediction

使用 Kaggle **Lending Club** 資料集進行貸款違約風險預測與資料分析的專案。  
這個 repo 主要作為機器學習實作與資料前處理練習，也可做為往後延伸模型實驗的基礎。

---

## 目標說明 (Objectives)

本專案的主要目標為：

1. **建立二元分類模型**  
   - 預測貸款是否可能「違約 / Charged Off」或「正常還款 / Fully Paid」  
   - 以 `loan_status` 為主要目標欄位（會先重新編碼成 0/1）

2. **練習完整的機器學習流程**  
   - 從原始 Kaggle 資料匯入、清理、特徵工程  
   - 切分訓練 / 測試資料，並處理類別不平衡問題  
   - 建立多種模型並比較效能

3. **建立一個可重現的實驗框架**  
   - 方便之後加入更多特徵 / 不同模型（如 Logistic Regression、Tree-based models、XGBoost 等）  
   - 日後可延伸到模型調參（Grid Search / Random Search / Optuna 等）

---

## 資料集來源 (Dataset)

- 來源：Kaggle – **Lending Club Loan Data**  
- 連結：<https://www.kaggle.com/datasets/wordsforthewise/lending-club>  
- 內容簡介：
  - 美國 P2P 借貸平台 **Lending Club** 的歷史貸款資料  
  - 每筆貸款包含申請人特徵（收入、職業、信用分數等）、貸款條件以及實際還款狀態  
  - 適合作為「信用風險評估 / 預測違約」的練習資料集

> 請先到 Kaggle 下載資料，並依照本機路徑放置於專案資料夾中（例如 `data/`）。

---

## 專案內容概要 (Project Overview)

本專案預計涵蓋以下步驟（可依實作進度調整）：

1. **資料前處理**
   - 移除無用或高度識別性的欄位（例如：`id`, `member_id` 等）
   - 處理遺失值（drop / 填補 / 移除高缺失欄位）
   - 選擇需要分析的 `loan_status` 類別，例如只保留：
     - `Fully Paid`
     - `Charged Off`
   - 將 `loan_status` 映射為二元標籤（如：Fully Paid = 0, Charged Off = 1）

2. **特徵工程**
   - 類別型變數編碼（Label Encoding / One-Hot Encoding）
   - 數值型變數標準化或正規化（如 StandardScaler / MinMaxScaler）
   - 視需求移除高度共線或無意義的特徵

3. **資料切分與樣本不平衡處理**
   - 將資料分成：
     - 訓練集 (train)
     - 測試集 (test)
   - 在訓練集上進行：
     - **Oversampling**（例如使用 SMOTE）
     - 或 **Undersampling / 結合 Pipeline**

4. **模型訓練與比較**
   - 基準模型（Baseline）：
     - Logistic Regression
   - 進階模型（可依實作情況調整）：
     - Decision Tree
     - Random Forest
     - Gradient Boosting / XGBoost
   - 使用交叉驗證 (Cross-validation) 比較模型表現

5. **效能評估**
   - 常用指標：
     - Accuracy
     - Precision / Recall / F1-score
     - ROC-AUC
     - Confusion Matrix
   - 視覺化：
     - ROC Curve
     - Precision-Recall Curve
     - Feature Importance（若使用樹模型）

