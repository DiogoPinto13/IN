# # import numpy as np
# # import pandas as pd


# # df = pd.read_excel("crime_dataset.xlsx", engine="openpyxl")
# # # print(df["LOCATION"])
# # # print(df.columns.values)
# # # df = df[['Crm Cd', 'Crm Cd Desc']]
# # # print(df)

# # #print(len(np.unique(np.array(df['Crm Cd']))))

# # import pandas as pd

# # # Try different delimiter options
# # #df = pd.read_csv("crime_dataset.csv", delimiter=",", error_bad_lines=False, engine="python")

# # severity_counts = df.groupby(["department_description", "severity_code"]).size().unstack(fill_value=0)

# # # Add a column for total count
# # severity_counts["total"] = severity_counts.sum(axis=1)

# # # Find the department with the highest total severity count
# # max_department = severity_counts["total"].idxmax()
# # max_count = severity_counts["total"].max()

# # # Print results
# # print("Severity counts per department:\n", severity_counts)
# # print("\nDepartment with highest severity count:", max_department, "with", max_count, "cases")

# import pandas as pd

# import pandas as pd

# def count_invalid_rows(df):
#     """
#     Counts rows where:
#     - 'Vict Age' is <= 0 or NULL
#     - 'Vic Sex' is NULL
#     - 'Vict Descent' is NULL
#     Also calculates the percentage of these invalid rows.

#     Parameters:
#     df (pd.DataFrame): The input DataFrame

#     Returns:
#     pd.DataFrame: A DataFrame with counts and percentages for each condition
#     """
#     total_rows = len(df)

#     # Count invalid rows for each column
#     age_invalid = df['Vict Age'].isnull().sum() + (df['Vict Age'] <= 0).sum()
#     sex_null = df['Vict Sex'].isnull().sum()
#     descent_null = df['Vict Descent'].isnull().sum()

#     # Calculate percentages
#     age_percent = (age_invalid / total_rows) * 100
#     sex_percent = (sex_null / total_rows) * 100
#     descent_percent = (descent_null / total_rows) * 100

#     # Create summary DataFrame
#     summary_df = pd.DataFrame({
#         'Condition': ['Vict Age ≤ 0 or NULL', 'Vic Sex NULL', 'Vict Descent NULL'],
#         'Count': [age_invalid, sex_null, descent_null],
#         'Percentage': [age_percent, sex_percent, descent_percent]
#     })

#     return summary_df



# # Example usage:
# #df = pd.read_excel("crime_dataset-3_excel.xlsx", engine="openpyxl")
# df = pd.read_csv("Crime_Data_from_2020_to_Present.csv")
# print(df.columns.values)
# print(count_invalid_rows(df))

import pandas as pd

def missing_data_summary(df):
    """
    Calculates the count and percentage of missing data for all columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame

    Returns:
    pd.DataFrame: A DataFrame with counts and percentages of missing data for each feature
    """
    total_rows = len(df)
    
    # Count missing values for each column
    missing_count = df.isnull().sum()
    
    # Calculate percentage of missing values
    missing_percentage = (missing_count / total_rows) * 100
    
    # Combine into a summary DataFrame
    summary_df = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage': missing_percentage
    })
    
    # Optionally, sort by highest percentage of missing data
    summary_df = summary_df[summary_df['Missing Count'] > 0].sort_values(by='Missing Percentage', ascending=False)
    
    return summary_df


# Example usage:
df = pd.read_csv("Crime_Data_from_2020_to_Present.csv")
extra_df = pd.read_excel("crime_dataset-3_excel.xlsx")
combined_df = pd.concat([df, extra_df], ignore_index=True)

print(combined_df.head())
combined_df.to_csv("merged_dataframe.csv", index=False)
print(extra_df)
#print(missing_data_summary(df))
