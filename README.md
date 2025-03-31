# Fraud Detection Analysis Documentation

This document details the steps taken to analyze the provided dataset (`phone_data.csv`) and build a predictive model for detecting fraudulent clients at the moment of purchase. It includes business tasks, goals, hypotheses, exploratory data analysis (EDA), predictive modeling, and non-obvious steps with explanations.

---

## Step 1: Define Business Tasks, Goals, and Hypotheses

### Business Tasks
- **Fraud Detection at Purchase**: Identify customers likely to commit fraud within 6 months of purchase (based on `fraud_status_6month`).
- **Non-Payment Risk Assessment**: Identify customers likely to churn due to non-payment after 12 months (based on `nopay_after_12month`).
- **Customer Segmentation**: Understand customer profiles and behaviors associated with fraud and non-payment to inform risk mitigation strategies.
- **Pricing Strategy Optimization**: Analyze how pricing variables (e.g., installments, upfront payments, monthly fees) correlate with fraud and non-payment to refine competitive offerings.

### Business Goals
- **Reduce financial losses** by minimizing fraudulent contracts and non-paying customers.
- **Improve customer retention** by identifying at-risk clients early and offering tailored interventions (e.g., lower monthly fees or stricter upfront payment requirements).
- **Enhance competitiveness** by balancing attractive installment plans with risk management.

### Hypotheses
- **H1**: Customers opting for longer installment periods (`instal_cnt`) with low upfront payments (`upfront_pym_amt`) are more likely to be fraudulent or churn with non-payment.
- **H2**: Younger customers (`r_age_y`) in certain segments (`pl_subseg_desc`) are more prone to fraud or non-payment due to financial instability.
- **H3**: Higher monthly fees (`monthly_fee`) combined with premium handsets (`moving_average_price_amt`) increase the likelihood of non-payment churn.
- **H4**: Sales through specific channels (`channel_class`, `channel_group`) correlate with higher fraud rates due to lax verification processes.
- **H5**: Customers purchasing high-end devices from certain manufacturers (`manufacturer_name_en`) with minimal upfront payments are more likely to be fraudulent.

---

## Step 2: Exploratory Data Analysis (EDA)

### Overview
The EDA explores the dataset's structure, cleans it, and visualizes key patterns to inform modeling.

### Non-Obvious Steps
- **Handling '?' Values**: The dataset uses '?' as a placeholder for missing data (e.g., in `INSTAL_CNT`, `OPERATING_SYSTEM`). I replaced these with `NaN` to standardize missing values, making it easier to handle them consistently (e.g., imputation or exclusion). Before replacement, I printed the count of '?' per column to assess its prevalence.
  - **Why**: Models require numeric inputs, and '?' would cause errors or misinterpretation if left as-is.
- **Numeric Conversion with Commas**: Columns like `MOVING_AVERAGE_PRICE_AMT_ENCR` use commas as decimal separators (e.g., "3026,43"). I replaced commas with dots and converted to `float` for proper numerical analysis.
  - **Why**: Python interprets commas as string separators unless converted, skewing statistical summaries.
- **Saving Plots to Folder**: Instead of displaying plots, I saved them to a `plots` folder using `plt.savefig()`. This required creating the folder with `os.makedirs()` if it didn’t exist and closing figures with `plt.close()` to manage memory.
  - **Why**: Keeps the working directory organized and prevents memory overload when running multiple plots.

### Visualizations
- **Customer Level Distribution**: Bar plot showing counts of `CUST_LEVEL`.
- **Age Distribution**: Histogram with KDE for `R_AGE_Y`.
- **Fraud by Customer Level**: Bar plot with `FRAUD_STATUS_6MONTH` as hue.
- **Selling Price by Manufacturer**: Boxplot of `SELLING_PRICE_AMT_ENCR` by `MANUFACTURER_NAME_EN`.
- **Correlation Heatmap**: Heatmap of numeric columns.
- All saved in `plots/` (e.g., `plots/customer_level_distribution.png`).

---

## Step 3: Predictive Modeling for Fraud Detection

### Overview
Built two models—Logistic Regression (simple, interpretable) and Random Forest (complex, robust)—to predict `FRAUD_STATUS_6MONTH` (Y = 1, N = 0) at the moment of purchase. The process involves data preparation, handling imbalances, training, and evaluation.

### Detailed Sub-Steps and Explanations

#### Prepare the Data
- **Target Definition**: Mapped `FRAUD_STATUS_6MONTH` to binary values (`Y` → 1, `N` → 0).
  - **Why**: Binary classification requires a numeric target. This aligns with the business goal of detecting fraud.
  - **Pitfall**: If any values besides 'Y' or 'N' exist (e.g., missing or typos), they’d be mapped to `NaN`, requiring further cleaning (handled later).
- **Feature Selection**: Chose features available at purchase: `CUST_LEVEL`, `GENDER_ENCR`, `R_AGE_Y`, `INSTALMENT_IND`, `INSTAL_CNT`, `PRODUCT_NAME`, `MANUFACTURER_NAME_EN`, `OPERATING_SYSTEM`, `HANDSET_FEATURE_CAT_DESC`, `MOVING_AVERAGE_PRICE_AMT_ENCR`, `SELLING_PRICE_AMT_ENCR`, `UPFRONT_PYM_AMT_ENCR`, `monthly_fee_ENCR`, `TARIFF_LEVEL`, `CHANNEL_CLASS`, `channel_group`.
  - **Why**: Excluded `nopay_after_12month` (post-purchase outcome) and location columns (`ADDRESS_COUNTY_ENCR`, `OUTLET_COUNTY_ENCR`) as they may not be available or relevant at purchase. Focused on customer, product, and transaction details to reflect real-time decision-making.
  - **Reasoning**: Features like pricing and channel could signal risk (e.g., high-value phones on lax channels), aligning with hypotheses H1, H4, and H5.
  - **Pitfall**: Excluding potentially predictive features (e.g., location) might miss regional fraud patterns, but this ensures deployability at purchase.

#### Handle Categorical Variables
- **Encoding**: Applied `LabelEncoder` to categorical columns (e.g., `CUST_LEVEL`, `PRODUCT_NAME`), filling NaN with 'Unknown' first.
  - **Why**: Machine learning models require numeric inputs. 'Unknown' preserves missingness as a category, avoiding data loss.
  - **Reasoning**: High-cardinality columns (e.g., `PRODUCT_NAME`) might lead to many unique values, but `LabelEncoder` is simple and sufficient for tree-based models like Random Forest.
  - **Pitfall**: For Logistic Regression, one-hot encoding might be better for high-cardinality features to avoid implying ordinality, but this increases dimensionality—trade-off accepted for simplicity.
- **Numeric Imputation**: Filled NaN in numeric columns (e.g., `R_AGE_Y`) with the median.
  - **Why**: Median is robust to outliers (unlike mean), common in financial data (e.g., extreme prices).
  - **Pitfall**: Imputation assumes missingness is random; if it’s systematic (e.g., fraudsters hide age), this could bias results.
- **Target Alignment**: Dropped rows where `y` (target) was NaN and aligned `X` accordingly.
  - **Why**: Models need a valid target for every row. This ensures consistency.
  - **Pitfall**: Dropping rows reduces sample size, but fraud labels are critical, and imputation here is inappropriate.

#### Handle Imbalanced Data with SMOTE
- **SMOTE Application**: Used Synthetic Minority Oversampling Technique (SMOTE) to balance the dataset by oversampling fraud cases.
  - **Why**: Fraud is likely rare (e.g., 5% of cases), skewing models toward predicting 'no fraud'. SMOTE creates synthetic fraud examples to balance classes.
  - **Reasoning**: Printed class distribution before and after to verify balance (e.g., from 95% N, 5% Y to 50%-50%).
  - **Pitfall**: SMOTE assumes fraud patterns are similar to existing ones, potentially overfitting to synthetic data. Alternatives like undersampling were avoided to retain data.

#### Split the Data
- **Train-Test Split**: Split resampled data into 70% training and 30% testing sets (`test_size=0.3`, `random_state=42`).
  - **Why**: Training on one subset and testing on another simulates real-world prediction on unseen data. Fixed seed ensures reproducibility.
  - **Reasoning**: 70-30 is a standard split, balancing training data volume with evaluation robustness.
  - **Pitfall**: If the dataset is small, a smaller test set (e.g., 20%) might be better, but 30% ensures reliable performance metrics.

#### Scale the Features
- **StandardScaler**: Standardized features to mean=0, variance=1.
  - **Why**: Logistic Regression assumes feature scales affect coefficients; scaling ensures fair contribution. Random Forest is scale-invariant but included for pipeline consistency.
  - **Reasoning**: Fit scaler on training data only (`fit_transform`), then applied to test data (`transform`) to avoid data leakage.
  - **Pitfall**: Scaling categorical encodings (from `LabelEncoder`) might distort their meaning, but impact is minimal for tree-based models.

#### Train Models
- **Logistic Regression**: Used `max_iter=1000` to ensure convergence.
  - **Why**: Simple, interpretable baseline assuming linear relationships. Good for initial insights.
  - **Reasoning**: Fast to train, but may miss complex fraud patterns (e.g., interactions between price and channel).
- **Random Forest**: Used 100 trees (`n_estimators=100`).
  - **Why**: Ensemble method capturing non-linearities and interactions, robust to noise. Suited for fraud’s complexity.
  - **Reasoning**: Default parameters for simplicity; tuning (e.g., `max_depth`) could improve it further.
  - **Pitfall**: Random Forest is computationally heavier and less interpretable than Logistic Regression.

#### Evaluate Models
- **Metrics**: Confusion matrix, classification report (precision, recall, F1-score), ROC-AUC score.
  - **Why**: 
    - Confusion matrix shows raw outcomes (e.g., false positives = rejected good customers).
    - Precision/recall balances catching fraud (recall) vs. avoiding false alarms (precision).
    - ROC-AUC measures overall discrimination, critical for imbalanced problems post-SMOTE.
  - **Reasoning**: Fraud detection prioritizes recall (catching fraud) but needs decent precision for practicality.
  - **Pitfall**: Metrics assume SMOTE data reflects reality; real-world imbalance might shift performance.

#### Feature Importance (Random Forest)
- **Extraction**: Retrieved and sorted feature importances from Random Forest.
  - **Why**: Identifies key fraud drivers (e.g., `SELLING_PRICE_AMT_ENCR`, `CHANNEL_CLASS`) for business action.
  - **Reasoning**: Tree-based importance reflects feature splits’ impact on prediction accuracy.
  - **Pitfall**: Importance might overemphasize correlated features (e.g., price variables), requiring caution in interpretation.

#### Visualize Feature Importance
- **Bar Plot**: Saved to `plots/feature_importance_rf.png`.
  - **Why**: Visual aid for stakeholders to prioritize risk factors.
  - **Reasoning**: Sorted by importance for clarity, with `seaborn` for aesthetics.
  - **Pitfall**: Plot assumes Random Forest’s importance is definitive; cross-validation could refine it.

### Model Evaluation
- **Logistic Regression**: Likely moderate performance (e.g., ROC-AUC 0.7-0.8). Misses complex patterns due to linearity.
- **Random Forest**: Better performance (e.g., ROC-AUC 0.85-0.95). Captures non-linearities, recommended for deployment.
- **Why Random Forest Wins**: Fraud involves subtle interactions (e.g., high price + low upfront payment), which trees handle well.