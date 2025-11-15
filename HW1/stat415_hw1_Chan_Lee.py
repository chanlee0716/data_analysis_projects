import numpy as np               # Numerical operations
import pandas as pd             # DataFrames, CSVs, missing data handling
import matplotlib.pyplot as plt  # Basic plotting
import seaborn as sns            # Statistical plots, heatmaps
import plotly.express as px      # Interactive plots
import csv                      # For reading/writing CSVs
import json                     # For reading/writing JSON
import pickle                   # For saving/loading Python objects

# Load your data
file_path = 'data/CreditCardFraud_updated.csv'
clean_data = pd.read_csv(file_path)

# Check for duplicated columns
duplicated_columns = clean_data.columns[clean_data.columns.duplicated()]
if not duplicated_columns.empty:
    #print("Duplicated columns found:", duplicated_columns)
    # Drop duplicated columns
    clean_data = clean_data.loc[:, ~clean_data.columns.duplicated()]

# Check for columns with identical data
duplicate_cols = []
print_cols = []
cols = clean_data.columns

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if clean_data[cols[i]].equals(clean_data[cols[j]]):
            duplicate_cols.append(cols[j])
            print_cols.append((cols[i], cols[j]))

clean_data = clean_data.drop(columns = duplicate_cols)
#print("Columns with identical data:", print_cols)


# Check for columns with entirely missing data
missing_data_columns = clean_data.columns[clean_data.isnull().all()]
if not missing_data_columns.empty:
    #print("Columns with entirely missing data:", missing_data_columns)
    # Drop columns with entirely missing data
    clean_data = clean_data.drop(columns=missing_data_columns)

clean_data = clean_data.drop(columns="Unnamed: 0")

# # Display the cleaned data
#print("Cleaned data:")
# print(data.head())

# print("Var Names:")
# print(list(clean_data.columns))


############################ Removing Outliers ############################

# # Function to print number of outliers and plot histograms in a grid
def question_2():
    # def plot_histograms_in_grid(data, exclude):
    #     numerical_cols = [col for col in data.select_dtypes(include=[np.number]).columns if col not in exclude] 
    #     num_cols = len(numerical_cols)
    #     num_rows = (num_cols // 3) + (num_cols % 3 > 0)  # Calculate number of rows needed for 3 columns per row

    #     fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 3))
    #     axes = axes.flatten()  # Flatten the axes array for easy iteration

    #     for i, column in enumerate(numerical_cols):
    #         Q1 = data[column].quantile(0.25)
    #         Q3 = data[column].quantile(0.75)
    #         IQR = Q3 - Q1
    #         lower_bound = Q1 - 1.5 * IQR
    #         upper_bound = Q3 + 3 * IQR
    #         outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    #         print(f"Outliers detected in {column}: {len(outliers)}")

    #         # Plot histogram
    #         axes[i].hist(data[column], bins=30, alpha=0.7, color='blue', edgecolor='black')
    #         axes[i].set_title(f'Histogram of {column}')
    #         axes[i].set_xlabel(column)
    #         axes[i].set_ylabel('Frequency')
    #         axes[i].grid(axis='y', alpha=0.75)

    #     # Hide any unused subplots
    #     for j in range(i + 1, len(axes)):
    #         fig.delaxes(axes[j])

    #     plt.tight_layout()
    #     plt.show()

    # plot_histograms_in_grid(clean_data, ['accountNumber', 'cardCVV', 'enteredCVV', 'cardLast4Digits'])







    # # Detect and handle outliers for all numerical columns
    # def detect_and_cap_outliers(data):
    #     numerical_cols = data.select_dtypes(include=[np.number]).columns
    #     for column in numerical_cols:
    #         Q1 = data[column].quantile(0.25)
    #         Q3 = data[column].quantile(0.75)
    #         IQR = Q3 - Q1
    #         lower_bound = Q1 - 1.5 * IQR
    #         upper_bound = Q3 + 3 * IQR

    #         # Optional: Print number of outliers detected
    #         outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    #         print(f"Outliers detected in {column}: {len(outliers)}")

    #         # Cap outliers
    #         data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
    #         data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
    #     return data

    # # Visualize the data for one of the columns
    # plt.figure(figsize=(10, 6))
    # plt.boxplot(clean_data['creditLimit'])  # Replace 'creditLimit' with any other numerical column name to visualize
    # plt.title('Box plot of creditLimit before handling outliers')
    # plt.show()

            
    print("old length", len(clean_data))# Apply the function to clean_data
    # clean_data_imputed = detect_and_cap_outliers(clean_data.copy())
    # #print("new length", len(clean_data))

    # # Visualize the data for one of the columns
    # plt.figure(figsize=(10, 6))
    # plt.boxplot(clean_data_imputed['creditLimit'])  # Replace 'creditLimit' with any other numerical column name to visualize
    # plt.title('Box plot of creditLimit after handling outliers')
    # plt.show()



    # Function to remove all outliers for all numerical variables exept those exclude.
    # Use the Tuker Method for Identifying Outliers
    def remove_outliers(data, LB_factor, UB_factor, exclude):
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for column in numerical_cols:
            if column not in exclude:
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - LB_factor * IQR
                upper_bound = Q3 + UB_factor * IQR
                data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        return data

    clean_data = remove_outliers(clean_data, 1.5, 1.5, ['posConditionCode'])

    # #  Print number of outliers detected
    # def print_certain_outliers(data, numerical_cols):
    #     for column in numerical_cols:
    #         Q1 = data[column].quantile(0.25)
    #         Q3 = data[column].quantile(0.75)
    #         IQR = Q3 - Q1
    #         lower_bound = Q1 - 1.5 * IQR
    #         upper_bound = Q3 + 3 * IQR
    #         outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    #         print(f"Outliers detected in {column}: {len(outliers)}")
            
    #         # Plot histogram with dynamic labels
    #         plt.figure(figsize=(10, 6))
    #         plt.hist(data[column], bins=30, alpha=0.7, color='blue', edgecolor='black')
    #         plt.title(f'Histogram of {column}')
    #         plt.xlabel(column)
    #         plt.ylabel('Frequency')
    #         plt.grid(axis='y', alpha=0.75)
    #         plt.show()


    #print("NUMBER OF OUTLIERS")
    #print_certain_outliers(clean_data.copy(), ['posEntryMode', 'posConditionCode'])

    #print("hello1", clean_data['posConditionCode'].unique())

    # Function to remove certain outliers
    def remove_certain_outliers_2(data, lower_bound, upper_bound, numerical_cols):
        for column in numerical_cols:
            # identify and print the outliers
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
            print(f"Outliers detected in {column}: {len(outliers)}")

            # remove them
            data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

            # # Plot histogram with dynamic labels
            # plt.figure(figsize=(10, 6))
            # plt.hist(data[column], bins=30, alpha=0.7, color='blue', edgecolor='black')
            # plt.title(f'Histogram of {column}')
            # plt.xlabel(column)
            # plt.ylabel('Frequency')
            # plt.grid(axis='y', alpha=0.75)
            # plt.show()

            return data

    clean_data = remove_certain_outliers_2(clean_data, 0, 40, ['posConditionCode'])

    #print("new length", len(clean_data))# Apply the function to clean_data


    #  # Print number of outliers detected
    # def print_outliers(data):
    #     numerical_cols = data.select_dtypes(include=[np.number]).columns
    #     for column in numerical_cols:
    #         Q1 = data[column].quantile(0.25)
    #         Q3 = data[column].quantile(0.75)
    #         IQR = Q3 - Q1
    #         lower_bound = Q1 - 1.5 * IQR
    #         upper_bound = Q3 + 3 * IQR
    #         outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    #         print(f"Outliers detected in {column}: {len(outliers)}")
            
    #         # Plot histogram with dynamic labels
    #         plt.figure(figsize=(10, 6))
    #         plt.hist(clean_data[column], bins=30, alpha=0.7, color='blue', edgecolor='black')
    #         plt.title(f'Histogram of {column}')
    #         plt.xlabel(column)
    #         plt.ylabel('Frequency')
    #         plt.grid(axis='y', alpha=0.75)
    #         plt.show()

    #print_outliers(clean_data.copy())

    # Display the entire 'transactionAmount' column
    #print(clean_data['transactionAmount'])
    # Display unique values in the 'transactionAmount' column
    #print("hello", clean_data['posConditionCode'].unique())


################################## Question 3 ##################################

# # Identify columns with missing values and print the count
# missing_values_count = clean_data.isnull().sum()
# print("Missing values in each column:")
# print(missing_values_count[missing_values_count > 0])

# print("acqCountry values", clean_data['acqCountry'].unique())
# print("merchantCountryCode values", clean_data['merchantCountryCode'].unique())
# print("transactionType values", clean_data['transactionType'].unique())


    # # Impute missing values using the mode for categorical columns
    # for column in ['acqCountry', 'merchantCountryCode', 'transactionType', 'posEntryMode', 'posConditionCode']:
    #     mode_value = clean_data[column].mode()[0]
    #     clean_data[column].fillna(mode_value, inplace=True)
    #     print(f"Imputed missing values in {column} with mode: {mode_value}")

    # # Verify that there are no more missing values in these columns
    # missing_values_count = clean_data.isnull().sum()
    # print("Missing values after imputation:")
    # print(missing_values_count[missing_values_count > 0])

    # # Visualize the distribution of each categorical variable
    # fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # # Plot for acqCountry
    # aqs = clean_data['acqCountry'].value_counts()
    # axes[0].bar(aqs.index, aqs.values, color='skyblue')
    # axes[0].set_title('Distribution of acqCountry')
    # axes[0].set_xlabel('Country')
    # axes[0].set_ylabel('Frequency')
    # axes[0].tick_params(axis='x', rotation=90)

    # # Plot for merchantCountryCode
    # mcc = clean_data['merchantCountryCode'].value_counts()
    # axes[1].bar(mcc.index, mcc.values, color='lightgreen')
    # axes[1].set_title('Distribution of merchantCountryCode')
    # axes[1].set_xlabel('Country Code')
    # axes[1].set_ylabel('Frequency')
    # axes[1].tick_params(axis='x', rotation=90)

    # # Plot for transactionType
    # tt = clean_data['transactionType'].value_counts()
    # axes[2].bar(tt.index, tt.values, color='salmon')
    # axes[2].set_title('Distribution of transactionType')
    # axes[2].set_xlabel('Transaction Type')
    # axes[2].set_ylabel('Frequency')
    # axes[2].tick_params(axis='x', rotation=90)

    # plt.tight_layout()
    # plt.show()

    # # Analyze the missing data pattern using a heatmap
    # plt.figure(figsize=(12, 6))
    # sns.heatmap(clean_data.isnull(), cbar=False, cmap='viridis')
    # plt.title('Missing Data Pattern')
    # plt.show()

    missing_counts = clean_data.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    missing_cols = missing_counts.index

    print("Columns with missing values:\n")
    print(missing_counts)

    # Create missing-value indicator columns (optional, can keep if needed)
    for col in missing_cols:
        clean_data[col + '_missing'] = clean_data[col].isnull().astype(int)

    # Replace missing values in selected categorical columns with a new category
    for column in ['acqCountry', 'merchantCountryCode', 'transactionType', 'posEntryMode', 'posConditionCode']:
        clean_data[column] = clean_data[column].fillna('Missing')
        print(f"Added 'Missing' category for null values in {column}")


####################################### Question 4 #######################################

def question_4():
    # Identify potential time-related columns based on common naming conventions
    time_columns = [col for col in clean_data.columns if 'date' in col.lower() or 'time' in col.lower()]
    print("Potential time-related columns:", time_columns)

    # Convert time-related columns to datetime format
    for col in time_columns:
        clean_data[col] = pd.to_datetime(clean_data[col], errors='coerce')
        print(f"Converted {col} to datetime format.")

    # Extract meaningful features from time variables
    for col in time_columns:
        clean_data[f'{col}_year'] = clean_data[col].dt.year
        clean_data[f'{col}_month'] = clean_data[col].dt.month
        clean_data[f'{col}_day'] = clean_data[col].dt.day
        clean_data[f'{col}_hour'] = clean_data[col].dt.hour
        clean_data[f'{col}_weekday'] = clean_data[col].dt.weekday
        print(f"Extracted features from {col}.")

    # Display the first few rows to verify the changes
    print(clean_data.head())

    # Identify and drop columns that contain only NA values
    na_only_columns = clean_data.columns[clean_data.isnull().all()]
    clean_data.drop(columns=na_only_columns, inplace=True)

    # Print the names of the columns that were removed
    print("Removed columns with only NA values:", na_only_columns.tolist())

    # Display the head of only the newly created time-related feature columns
    new_time_features = [col for col in clean_data.columns if any(suffix in col for suffix in ['_year', '_month', '_day', '_hour', '_weekday'])]
    print(clean_data[new_time_features].head())

def question_5():
    # print(clean_data[
    #     (clean_data['cardCVV'].astype(str).str.len() < 2) |
    #     (clean_data['enteredCVV'].astype(str).str.len() < 2) |
    #     (clean_data['cardLast4Digits'].astype(str).str.len() < 4)
    # ][['cardCVV', 'enteredCVV', 'cardLast4Digits']])

    # Create a flag for valid CVV length
    clean_data['is_valid_cardCVV'] = clean_data['cardCVV'].astype(str).str.len() == 3
    clean_data['is_valid_enteredCVV'] = clean_data['enteredCVV'].astype(str).str.len() == 3

    # Create a flag for valid card last 4 digits length
    clean_data['is_valid_cardLast4Digits'] = clean_data['cardLast4Digits'].astype(str).str.len() == 4

    # Print the first few rows of the new flags to verify
    #print(clean_data[['is_valid_cardCVV', 'is_valid_enteredCVV', 'is_valid_cardLast4Digits']].head())
    print(len(clean_data[clean_data['is_valid_cardCVV'] == False]['cardCVV'])) # all card CVV values are good
    print(len(clean_data[clean_data['is_valid_enteredCVV'] == False]['enteredCVV'])) # all card CVV values are good
    print(clean_data[clean_data['is_valid_enteredCVV'] == False]['enteredCVV']) # all card CVV values are good

    # change all these numbers into strings
    clean_data['cardCVV_str'] = [str(cvv) for cvv in clean_data['cardCVV']]
    clean_data['enteredCVV_str'] = [str(cvv).zfill(3) if cvv < 100 else str(cvv) for cvv in clean_data['enteredCVV']]
    clean_data['cardLast4Digits_str'] = [str(cvv).zfill(4) if cvv < 1000 else str(cvv) for cvv in clean_data['cardLast4Digits']]

    # Check that 'enteredCVV_str' has exactly 3 characters
    invalid_enteredCVV_str = clean_data[clean_data['enteredCVV_str'].str.len() != 3]
    print(f"Number of invalid 'enteredCVV_str' entries: {len(invalid_enteredCVV_str)}")

    # Check that 'cardLast4Digits_str' has exactly 4 characters
    invalid_cardLast4Digits_str = clean_data[clean_data['cardLast4Digits_str'].str.len() != 4]
    print(f"Number of invalid 'cardLast4Digits_str' entries: {len(invalid_cardLast4Digits_str)}")

    print()

def question_6():
    # Prepare the data for visualization, filtering only fraud cases
    fraud_analysis = clean_data[clean_data['isFraud'] == True][['cardCVV', 'enteredCVV', 'cardLast4Digits']]

    # Create a bar chart for each variable
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Plot for cardCVV
    sns.countplot(data=fraud_analysis, x='cardCVV', ax=axes[0],
                  order=fraud_analysis['cardCVV'].value_counts().sort_values().index)
    axes[0].set_title('Fraud by cardCVV')
    axes[0].set_xlabel('cardCVV')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=90)

    # Plot for enteredCVV
    sns.countplot(data=fraud_analysis, x='enteredCVV', ax=axes[1],
                  order=fraud_analysis['enteredCVV'].value_counts().sort_values().index)
    axes[1].set_title('Fraud by enteredCVV')
    axes[1].set_xlabel('enteredCVV')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=90)

    # Plot for cardLast4Digits
    sns.countplot(data=fraud_analysis, x='cardLast4Digits', ax=axes[2],
                  order=fraud_analysis['cardLast4Digits'].value_counts().sort_values().index)
    axes[2].set_title('Fraud by cardLast4Digits')
    axes[2].set_xlabel('cardLast4Digits')
    axes[2].set_ylabel('Count')
    axes[2].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()


def question_7():
    # Calculate the IQR for transactionAmount
    Q1 = clean_data['transactionAmount'].quantile(0.25)
    Q3 = clean_data['transactionAmount'].quantile(0.75)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the data to exclude outliers
    filtered_data = clean_data[(clean_data['transactionAmount'] >= lower_bound) & 
                               (clean_data['transactionAmount'] <= upper_bound)]

    # Plot a histogram for transactionAmount without outliers
    plt.figure(figsize=(12, 6))
    sns.histplot(filtered_data['transactionAmount'], bins=50, kde=False, color='skyblue')
    plt.title('Histogram of Transaction Amount (Without Outliers)')
    plt.xlabel('Transaction Amount')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    # Plot a density plot for transactionAmount without outliers
    plt.figure(figsize=(12, 6))
    sns.kdeplot(filtered_data['transactionAmount'], shade=True, color='salmon')
    plt.title('Density Plot of Transaction Amount (Without Outliers)')
    plt.xlabel('Transaction Amount')
    plt.ylabel('Density')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    # Plot a histogram for the log-transformed transactionAmount without outliers
    plt.figure(figsize=(12, 6))
    log_transaction_amount = np.log1p(filtered_data['transactionAmount'])  # Use log1p to handle zero values
    sns.histplot(log_transaction_amount, bins=50, kde=False, color='lightgreen')
    plt.title('Histogram of Log-Transformed Transaction Amount (Without Outliers)')
    plt.xlabel('Log(Transaction Amount + 1)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()



def plot_fraud_rate_by_category():
    # List of categorical columns to analyze
    categorical_columns = [
        #'merchantCategoryCode',
        #'posEntryMode',
        #'transactionType',
        'posConditionCode',
        'merchantCountryCode'
    ]

    # Set up the figure for subplots
    fig, axes = plt.subplots(len(categorical_columns), 1, figsize=(14, 5 * len(categorical_columns)))

    for i, column in enumerate(categorical_columns):
        # Calculate fraud rate for each category
        fraud_rate = clean_data.groupby(column)['isFraud'].mean().sort_values(ascending=False)

        # Plot the fraud rate
        sns.barplot(x=fraud_rate.index, y=fraud_rate.values, ax=axes[i], palette='viridis')
        axes[i].set_title(f'Fraud Rate by {column}', fontsize=14, weight='bold')
        axes[i].set_xlabel(column, fontsize=12)
        axes[i].set_ylabel('Fraud Rate', fontsize=12)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(pad=3.0)
    plt.show()




if __name__ == "__main__":
    #question_7()
    #print(clean_data['transactionAmount'].head())
    #plot_fraud_rate_by_category()
    # Check for missing values
    # Identify columns with missing values
    

















