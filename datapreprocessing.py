import pandas as pd
import sklearn.preprocessing as skl
import sklearn.model_selection as train_test_split


print("''''''''''''''''''''''\n\nOriginal Data\n\n''''''''''''''''''''''")
# Identifying Missing Values
data = pd.read_csv(r'D:\SEMESTER-SE-6\DATA SCIENCE\Data Preprocessing\data.csv')
print("Original DataFrame:\n", data)  # Print the raw data

print("''''''''''''''''''''''\n\nMissing Values\n\n''''''''''''''''''''''")
missing_values = data.isnull().sum()
print("\nMissing values per column:\n", missing_values)  # Print missing value counts

# Filling missing values with mean of the respective column => AGE
print("''''''''''''''''''''''\n\nData Filling\n\n''''''''''''''''''''''")
data['age'] = data['age'].fillna(data['age'].mean())
print("\nUpdated DataFrame after filling Age:\n", data)  # Print updated data

# For the income column we are dropping the rows with missing values
print("''''''''''''''''''''''\n\nDropping Values\n\n''''''''''''''''''''''")
data.dropna(subset=['income'], inplace=True)
print("\nUpdated DataFrame after missing income values dropped:\n", data)  # Print updated data

# Encoding categorical variables to avoid the dummy variable trap while doing regression
print("''''''''''''''''''''''\n\nEncoding Values\n\n''''''''''''''''''''''")
data = pd.get_dummies(data, columns=['gender', 'city'], drop_first=False)
print("\nUpdated DataFrame after encoding categorical variables:\n", data)  # Print updated data

# Normalizing the income data so that it falls within the range of 0 to 1
print("''''''''''''''''''''''\n\nNormalizing Data\n\n''''''''''''''''''''''")
scaler = skl.MinMaxScaler()
data['income'] = scaler.fit_transform(data[['income']])
print("\nUpdated DataFrame after normalizing income:\n", data)  # Print updated data

# Finally, we can split the data into training and testing sets
print("''''''''''''''''''''''\n\nSplitting Data\n\n''''''''''''''''''''''")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
print("\nTest Data:\n", data)
print("\nTrain Data:\n", data)
