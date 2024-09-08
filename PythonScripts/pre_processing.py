import pandas as pd

try:
    df = pd.read_csv('/Users/aditya/Documents/SRH/Thesis/DataSets/adg0021_er_2024_03_13.csv', delimiter='/t')
except pd.errors.ParserError as e:
    print(f"Error parsing file: {e}")

# Keep only the 'Transkript' column 
if 'df' in locals():
    df_cleanedDataset = df[['Transkript']]

    # Save the cleaned dataset to a new CSV file
    df_cleanedDataset.to_csv('/Users/aditya/Documents/SRH/Thesis/6September2024/cleaned.csv', index=False)

    # Display the first few rows of the cleaned dataset
    print(df_cleanedDataset.head())
