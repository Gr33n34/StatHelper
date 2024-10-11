import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency, spearmanr
import itertools
import statsmodels.api as sm

def perform_tests(df: pd.DataFrame, test_cols: list, outcome_col: str) -> pd.DataFrame:
    # Results DataFrame to store results
    results = pd.DataFrame(columns=['column_name', 'test_performed', 'p_value'])

    # Get the unique values of the outcome column (assuming only 2 categories for t-test)
    outcome_groups = df[outcome_col].unique()

    if len(outcome_groups) != 2:
        raise ValueError("Outcome column must have exactly two unique categories.")

    # Group the dataframe based on the outcome column
    group1 = df[df[outcome_col] == outcome_groups[0]]
    group2 = df[df[outcome_col] == outcome_groups[1]]

    # Loop through each test column
    for col in test_cols:
        if len(group1[col].dropna()) < 3 or len(group2[col].dropna()) < 3:
            print(f"Skipping column {col}: Less than 3 values in one of the groups.")
            break
        else:
            try:
                # Try to convert the column to integers
                df[col] = df[col].astype(int)
                # Perform t-test
                t_stat, p_value = ttest_ind(group1[col], group2[col], nan_policy='omit')
                # Append results to the DataFrame
                results = results.append({'column_name': col, 'test_performed': 'ttest', 'p_value': p_value}, ignore_index=True)

            except ValueError:  # If conversion fails (i.e., non-numeric data)
                # Check if there are more than 5 identical elements in the column
                if df[col].value_counts().max() > 5:
                    # Create a contingency table for the Chi-Square test
                    contingency_table = pd.crosstab(df[col], df[outcome_col])
                    # Perform the Chi-Square test
                    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
                    # Append results to the DataFrame
                    results = results.append({'column_name': col, 'test_performed': 'chi-square', 'p_value': p_value}, ignore_index=True)
                else:
                    # If no element appears more than 5 times, print and skip this column
                    print(f"Skipping column {col}: Not enough repetitions for Chi-Square test.")

    return results

# Function to compute Cramer's V for categorical variables
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1))))

def perform_collinearity_testing(df: pd.DataFrame, results: pd.DataFrame, outcome_col: str) -> None:
    # Filter out columns with p-value <= 0.2
    significant_columns = results[results['p_value'] <= 0.2]['column_name']

    # Separate columns tested with chi-square and ttest
    chisquare_cols = results[(results['column_name'].isin(significant_columns)) &
                             (results['test_performed'] == 'chi-square')]['column_name'].tolist()
    ttest_cols = results[(results['column_name'].isin(significant_columns)) &
                         (results['test_performed'] == 'ttest')]['column_name'].tolist()

    # Stepwise collinearity checks
    # 1. Check collinearity for chi-square tested columns using Cramer's V
    for col1, col2 in itertools.combinations(chisquare_cols, 2):
        # Contingency table for the two columns
        contingency_table = pd.crosstab(df[col1], df[col2])
        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
            cramers_v_value = cramers_v(contingency_table)
            print(f"Cramer's V between {col1} and {col2}: {cramers_v_value:.4f}")

    # 2. Check collinearity between ttest columns (numeric data) using Spearman correlation
    for col1, col2 in itertools.combinations(ttest_cols, 2):
        # Calculate Spearman correlation
        spearman_corr, spearman_pvalue = spearmanr(df[col1], df[col2])
        if spearman_pvalue < 0.05:
            print(f"Spearman correlation between {col1} and {col2}: correlation={spearman_corr:.4f}, p-value={spearman_pvalue:.4f}")

    # 3. Check collinearity between ttest columns and chi-square columns
    for t_col in ttest_cols:
        for c_col in chisquare_cols:
            # Create a contingency table between the ttest column and chisquare column
            contingency_table = pd.crosstab(df[t_col], df[c_col])
            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                cramers_v_value = cramers_v(contingency_table)
                print(f"Cramer's V between {t_col} (ttest) and {c_col} (chi-square): {cramers_v_value:.4f}")

def logistic_regression(df: pd.DataFrame, predictors: list, outcome_col: str):
    # Ensure that the outcome column is binary (0 or 1)
    if df[outcome_col].nunique() != 2:
        raise ValueError("The outcome column must be binary (contain exactly two unique values).")

    # Prepare the predictors (independent variables) and outcome (dependent variable)
    X = df[predictors]
    y = df[outcome_col]

    # Add a constant (intercept) to the predictors
    X = sm.add_constant(X)

    # Fit the logistic regression model
    model = sm.Logit(y, X)
    result = model.fit()

    # Return the summary of the logistic regression model
    return result.summary()
