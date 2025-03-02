import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_data(n_samples=100):
    np.random.seed(32)
    rooms = np.random.randint(2, 8, size=n_samples)
    base_price = 25 + (rooms * 3) + np.random.normal(0, 5, n_samples)
    crime_rate = 12 - (base_price / 5) + np.random.normal(0, 0.5, n_samples)
    crime_rate = np.clip(crime_rate, 0, 10)
    distance_to_center = np.random.uniform(1, 20, size=n_samples)
    distance_to_center = np.clip(distance_to_center, 1, 20)
    
    data = {
        'Price': base_price,
        'Rooms': rooms,
        'Crime_Rate': crime_rate,
        'Distance_to_Center': distance_to_center
    }
    return pd.DataFrame(data)

def calculate_statistics(df):
    print("Central Tendency Measures:")
    print("Mean:\n", df.mean())
    print("\nMedian:\n", df.median())
    print("\nMode:\n", df.mode().iloc[0])
    
    print("\nVariability Measures:")
    print("Standard Deviation:\n", df.std())
    print("\nVariance:\n", df.var())
    
    print("\nRange Analysis:")
    for column in df.columns:
        print(f"\nRange for {column}:")
        print(f"Minimum: {df[column].min()}")
        print(f"Maximum: {df[column].max()}")

def plot_histograms(df):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.histplot(df['Price'], kde=True)
    plt.title('Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 2)
    sns.histplot(df['Crime_Rate'], kde=True)
    plt.title('Crime Rate Distribution')
    plt.xlabel('Crime Rate')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 3)
    sns.histplot(df['Distance_to_Center'], kde=True)
    plt.title('Distance to Center Distribution')
    plt.xlabel('Distance to Center')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def plot_boxplots(df):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.boxplot(x=df['Price'])
    plt.title('Price Box Plot')
    plt.xlabel('Price')
    
    plt.subplot(1, 3, 2)
    sns.boxplot(x=df['Crime_Rate'])
    plt.title('Crime Rate Box Plot')
    plt.xlabel('Crime Rate')
    
    plt.subplot(1, 3, 3)
    sns.boxplot(x=df['Distance_to_Center'])
    plt.title('Distance to Center Box Plot')
    plt.xlabel('Distance to Center')
    
    plt.tight_layout()
    plt.show()

# Main execution
df = generate_data()
df.to_csv('synthetic_boston_housing_data.csv', index=False)
print("Synthetic dataset generated and saved.")

# Load data
df_boston = pd.read_csv('synthetic_boston_housing_data.csv')

# Calculate statistics
calculate_statistics(df_boston)

# Create visualizations
plot_histograms(df_boston)
plot_boxplots(df_boston)
