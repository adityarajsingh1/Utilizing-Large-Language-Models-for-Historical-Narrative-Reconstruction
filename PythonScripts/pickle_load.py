import pickle

# Path to your pickle file
pickle_file_path = '/Users/aditya/Documents/SRH/Thesis/TranskriptOnly1.pkl'

# Load the data from the pickle file
with open(pickle_file_path, 'rb') as file:
    interview_data = pickle.load(file)

# Now 'interview_data' contains your dataset, which you can use for further processing
print(interview_data)
