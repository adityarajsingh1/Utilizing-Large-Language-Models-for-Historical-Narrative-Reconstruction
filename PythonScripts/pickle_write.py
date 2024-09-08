import pandas as pd
import pickle

# Step 1: Load the CSV file into a DataFrame
csv_file_path = '/Users/aditya/Documents/SRH/Thesis/cleaned.csv'
interview_data = pd.read_csv(csv_file_path)

# Step 2: Save the DataFrame to a pickle file
pickle_file_path = '/Users/aditya/Documents/SRH/Thesis/transcript.pkl'
with open(pickle_file_path, 'wb') as file:
    pickle.dump(interview_data, file)

print(f"Interview dataset from {csv_file_path} saved to {pickle_file_path}")
