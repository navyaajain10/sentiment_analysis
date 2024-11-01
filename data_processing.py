import pandas as pd

# read dataset
data_dictionary = pd.read_excel('/Users/vjain/Downloads/case_study/practice_case_study_data.xlsx', sheet_name = 'Data Dictionary')
df = pd.read_excel('/Users/vjain/Downloads/case_study/practice_case_study_data.xlsx', sheet_name = 'Practice Test Case 4 Data')

# print schema
print("Schema:")
print(df.info())

# count unique IDs
unique_ids = df['ID'].nunique()
print("Unique Ids: ", unique_ids)

# count total missing values
count_missing = df.isnull().sum()
print("Total Missing Values:")
print(count_missing)

# drop rows with missing values & duplicates
cleaned_df = df.dropna().drop_duplicates(subset=['ID'])

# save cleaned data
cleaned_df.to_csv('cleaned_data.csv', index=False)

count_cleaned_rows = cleaned_df.shape[0]
print('Rows after cleaning: ', count_cleaned_rows)

# percentage of positive labels
positive_percentage = cleaned_df['Label'].mean() * 100
print("Positive Percentage: ", round(positive_percentage, 2), '%')