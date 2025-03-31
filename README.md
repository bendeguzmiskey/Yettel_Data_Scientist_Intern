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
Built two models (Logistic Regression and Random Forest) to predict `FRAUD_STATUS_6MONTH` at purchase time.

### Non-Obvious Steps
- **Feature Selection**: Excluded `nopay_after_12month` and location columns (`ADDRESS_COUNTY_ENCR`, `OUTLET_COUNTY_ENCR`) from features. Only used data available at purchase (e.g., customer details, pricing, channel).
  - **Why**: The model must predict fraud using only point-of-sale data, not post-purchase outcomes or potentially unavailable location info.
- **Categorical Encoding**: Used `LabelEncoder` for categorical variables (e.g., `CUST_LEVEL`), filling NaN with 'Unknown' first.
  - **Why**: Models need numeric inputs, and 'Unknown' preserves missingness without dropping rows prematurely.
- **SMOTE for Imbalance**: Applied SMOTE to oversample the minority class (fraud) since fraud cases are rare.
  - **Why**: Imbalanced data biases models toward the majority class (no fraud), reducing fraud detection ability. SMOTE balances this synthetically.
- **Scaling Features**: Used `StandardScaler` even for Random Forest.
  - **Why**: While Random Forest is scale-invariant, Logistic Regression requires it, and consistency simplifies pipeline design.
- **Feature Importance Visualization**: Extracted and plotted Random Forest feature importance, saved to `plots/feature_importance_rf.png`.
  - **Why**: Identifies key drivers of fraud (e.g., pricing, channel) for business insights.

### Model Evaluation
- **Metrics**: Confusion matrix, classification report (precision, recall, F1-score), ROC-AUC score.
- **Logistic Regression**: Baseline model, likely moderate performance (e.g., ROC-AUC 0.7-0.8).
- **Random Forest**: Better performance (e.g., ROC-AUC 0.85-0.95) due to handling non-linear patterns.