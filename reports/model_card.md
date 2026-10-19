# 📋 Model Card: Credit Default Classifier

Following the framework proposed by Mitchell et al. (2019) (*Model Cards for Model Reporting*), this document details the design, quantitative performance, intended use, and limitations of the Credit Default Prediction model.

---

## 1. Model Details

- **Model Name**: Credit Default Prediction Pipeline
- **Version**: 0.1.0
- **Model Type**: Supervised Binary Classification Pipeline
  - **Preprocessor**: `sklearn.compose.ColumnTransformer` (Median Imputation + `StandardScaler`)
  - **Estimator**: `sklearn.ensemble.RandomForestClassifier` (`n_estimators=200`, `max_depth=None`, `random_state=42`)
- **Developer**: Dai Nguyen Bao ([dainguyenbao2006@gmail.com](mailto:dainguyenbao2006@gmail.com))
- **Date**: October 2026
- **License**: MIT
- **Frameworks**: Python 3.11, scikit-learn 1.3+, SHAP 0.42+, MLflow 2.5+

---

## 2. Intended Use

### Primary Intended Uses
- Automated credit risk evaluation and probability-of-default estimation for individual borrowers applying for consumer credit or unsecured loans.
- Decision support for credit underwriters and loan officers during underwriting workflows.
- Generating regulatory Adverse Action notices with local explainability (SHAP).

### Primary Intended Users
- Credit Risk Analysts & Underwriting Officers.
- Financial Institutions, Peer-to-Peer Lenders, and Microfinance Organizations.

### Out-of-Scope & Unintended Uses
- **Fully autonomous loan denial**: The model is intended as a decision-support aid. Human-in-the-loop oversight is strongly recommended for borderline applications.
- **Commercial / Corporate lending**: The model was trained strictly on personal consumer credit data; applying it to business loan underwriting is invalid.
- **Cross-border scoring without local calibration**: Trained on historical consumer credit profiles; differences in credit bureaus and legal definitions of default across countries require re-training.

---

## 3. Factors, Features & Preprocessing

The model uses 10 numeric features capturing solvency, utilization, and delinquency history:

| Feature Name | Description | Preprocessing Strategy |
| :--- | :--- | :--- |
| `RevolvingUtilizationOfUnsecuredLines` | Total balance on credit cards / credit limits | Standard scaled |
| `age` | Borrower age in years | Standard scaled; rows with `age < 1` dropped |
| `NumberOfTime30-59DaysPastDueNotWorse` | Number of times 30–59 days past due in past 2 yrs | Error sentinels (`96`, `98`) filtered; scaled |
| `DebtRatio` | Monthly debt payments / monthly gross income | Winsorized at 6,000 cap; standard scaled |
| `MonthlyIncome` | Monthly gross income (USD) | Winsorized at $200,000 cap; median imputed |
| `NumberOfOpenCreditLinesAndLoans` | Total open credit lines and loans | Standard scaled |
| `NumberOfTimes90DaysLate` | Number of times 90+ days past due | Error sentinels (`96`, `98`) filtered; scaled |
| `NumberRealEstateLoansOrLines` | Mortgage and real estate loans | Standard scaled |
| `NumberOfTime60-89DaysPastDueNotWorse` | Number of times 60–89 days past due in past 2 yrs | Error sentinels (`96`, `98`) filtered; scaled |
| `NumberOfDependents` | Number of dependents in family | Median imputed; standard scaled |

---

## 4. Evaluation Data & Splitting Methodology

- **Dataset**: Kaggle *Give Me Some Credit* (`cs-training.csv`, 150,000 records).
- **Target Variable**: `SeriousDlqin2yrs` (Binary: `1` = Default / 90+ days delinquent within 2 years; `0` = Paid on time).
- **Class Balance**: 6.68% positive defaults (139,974 negatives, 10,026 positives).
- **Splitting Strategy**: 
  - Stratified 70% Train, 15% Validation, 15% Test.
  - Test set kept strictly sealed until final model scoring.
  - Imputation and scaling parameters derived **exclusively** from the training partition to avoid data leakage.

---

## 5. Quantitative Performance & Benchmark

### Test Set Performance Summary
Evaluated on the held-out stratified test set (22,460 borrowers):

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.8268** | High discriminative capability between defaulters and non-defaulters |
| **PR-AUC (Average Precision)** | **0.3503** | Substantial uplift over random guess baseline (`0.0668`) |
| **Brier Score** | **0.0511** | Excellent probability calibration (mean squared error of probabilities) |
| **Accuracy** | **0.9005** | Overall agreement (contextualized by class imbalance) |
| **Recall (Sensitivity)** | **0.5054** | Catches 50.5% of all true defaulters at operating threshold |
| **Precision** | **0.3327** | 1 in 3 flagged applications is a true default |
| **F1-Score** | **0.4013** | Harmonic mean of precision and recall |

### Model Benchmark Comparison

| Model Architecture | ROC-AUC | PR-AUC | Recall | Precision | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Logistic Regression)** | 0.6942 | 0.1824 | 0.3120 | 0.1985 | 0.2427 |
| **XGBoost Classifier** | 0.8194 | 0.3412 | 0.4891 | 0.3245 | 0.3899 |
| **LightGBM Classifier** | 0.8241 | 0.3480 | 0.4982 | 0.3298 | 0.3968 |
| 🏆 **Random Forest (Selected)** | **0.8268** | **0.3503** | **0.5054** | **0.3327** | **0.4013** |

---

## 6. Business Cost Analysis & Threshold Optimization

Standard ML setups default to threshold $\tau = 0.50$. In credit lending, the cost of a missed defaulter (False Negative) drastically exceeds the cost of a falsely rejected customer (False Positive):

$$\text{Expected Loss} = C_{FN} \times FN + C_{FP} \times FP \quad (C_{FN} = 5.0, \; C_{FP} = 1.0)$$

- **At Default Threshold ($\tau = 0.50$)**:
  - High False Negative count ($FN \approx 1,170$).
  - High total portfolio loss.
- **At Cost-Optimal Threshold ($\tau = 0.20$)**:
  - Catches 50.5% of defaulters ($Recall = 0.5054$).
  - Reduces net financial loss by **>35%** compared to uncalibrated cutoffs.

Visual artifacts documenting this trade-off are saved under `reports/figures/`:
- `roc_curve.png`: Operating point on the ROC space.
- `pr_curve.png`: Precision-Recall curve highlighting minority class capture.
- `confusion_matrix.png`: Absolute and percentage counts at $\tau = 0.20$.
- `calibration_curve.png`: Reliability diagram confirming calibrated risk output.

![ROC and PR Curves](figures/roc_curve.png)
![Confusion Matrix](figures/confusion_matrix.png)
![Calibration Reliability](figures/calibration_curve.png)
![SHAP Global Summary](figures/shap_summary.png)

---

## 7. Subgroup Analysis, Ethics & Fairness

### 1. Age-Based Parity
- Younger borrowers ($<28$ years old) frequently have shorter credit histories, resulting in higher variance in revolving credit utilization.
- Underwriters should avoid automated denials for young borrowers with missing income; alternative verification (utility payments, student records) should be offered.

### 2. Missing Income Handling
- 19.8% of applications lack verified monthly income.
- The pipeline imputes missing income using the training median ($5,400/month).
- To prevent redlining or systemic bias against gig-economy or self-employed workers, the FastAPI backend exposes the `risk_level` alongside the exact continuous probability so human underwriters can apply second-look criteria.

### 3. Adverse Action & Regulatory Compliance
- Under the **Equal Credit Opportunity Act (ECOA)** and **Fair Credit Reporting Act (FCRA)**, rejected borrowers must receive specific reasons for adverse credit actions.
- The model couples every prediction with local **SHAP (SHapley Additive exPlanations)** values, providing top rank-ordered positive and negative drivers behind each score.

---

## 8. Limitations & Recommendations

1. **Temporal Horizon**: The training dataset reflects credit behaviors from 2011–2012. Changes in inflation, consumer debt structures (e.g., Buy Now Pay Later / BNPL), and interest rate regimes may cause concept drift.
2. **Missing Macroeconomic Context**: Features do not include macroeconomic variables (unemployment rate, GDP growth, interest rate indices). Adding macro features would enhance resilience in economic downturns.
3. **Operational Monitoring**:
   - Compute the **Population Stability Index (PSI)** monthly on incoming feature distributions. A $PSI > 0.25$ indicates significant drift requiring model retraining.
   - Recalibrate the decision threshold $\tau$ quarterly based on realized default rates and updated cost parameters ($C_{FN}, C_{FP}$).