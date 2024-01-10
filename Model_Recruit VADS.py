#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install --upgrade scikit-learn


# In[2]:


get_ipython().system('pip freeze > requirements.txt')


# In[3]:


# 1. Model training
import pandas as pd  # Add this import statement
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
import pickle


# In[4]:


# Load the training data
train_data_path = r"D:\1 ISB\Term 2\FP\FP project\Trainingdataset_data.csv"
train_data = pd.read_csv(train_data_path)

# Preprocess the 'Relevancy score' column to handle percentage values
train_data['Relevancy score'] = train_data['Relevancy score'].str.rstrip('%').astype('float') / 100.0

# Extract relevant columns for training
train_features = train_data[['sorted_skills', 'Certification', 'Experience']]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_features.astype(str).agg(' '.join, axis=1))

# Train the Linear Regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, train_data['Relevancy score'])

# Save the model using pickle
model_filename = 'Recruit_VADS_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the vectorizer
vectorizer_filename = 'Tfidf_Vectorizer.pkl'
with open(vectorizer_filename, 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model trained and saved as", model_filename)
print("Vectorizer saved as", vectorizer_filename)


# In[5]:


# Load the model and vectorizer
import pickle
model = pickle.load(open('Recruit_VADS_model.pkl', 'rb'))
vectorizer = pickle.load(open('Tfidf_Vectorizer.pkl', 'rb'))

# Load the resume data
import pandas as pd
resume_data_path = r"D:\1 ISB\Term 2\FP\FP project\Modifiedresumedata_data.csv"
resume_data = pd.read_csv(resume_data_path)

# Define a function that takes the input from the UI and returns the relevancy score
def get_relevancy_score(job_title, skills, certification, experience):
    # Create a vector from the input
    input_features = [job_title, skills, certification, experience]
    input_vector = vectorizer.transform(input_features).toarray()
    
    # Compute the cosine similarity with the model
    similarity = model.dot(input_vector.T)
    
    # Sort the candidates by descending order of similarity
    sorted_indices = similarity.argsort(axis=0)[::-1]
    sorted_similarity = similarity[sorted_indices]
    
    # Format the output as a dataframe with candidate name, email and relevancy score
    output = pd.DataFrame()
    output['Candidate Name'] = resume_data['Candidate Name'][sorted_indices].squeeze()
    output['Email ID'] = resume_data['Email ID'][sorted_indices].squeeze()
    output['Relevancy Score'] = (sorted_similarity * 100).round(2).squeeze()
    output['Relevancy Score'] = output['Relevancy Score'].astype(str) + '%'
    
    return output


# In[6]:


# Load the model and vectorizer
model = pickle.load(open('Recruit_VADS_model.pkl', 'rb'))
vectorizer = pickle.load(open('Tfidf_Vectorizer.pkl', 'rb'))

# Model Evaluation for Linear Regression Model
test_data_path = r"D:\1 ISB\Term 2\FP\FP project\Testingdataset_data.csv"
test_data = pd.read_csv(test_data_path)

# Extract relevant columns for testing
test_features = test_data[['sorted_skills', 'Certification', 'Experience']]

# Create a TF-IDF vectorizer using the same vectorizer from training
X_test = vectorizer.transform(test_features.astype(str).agg(' '.join, axis=1))

# Use the trained model to predict on the test data
predictions = model.predict(X_test)

# Create a DataFrame to store the predictions and actual values for comparison
evaluation_results = pd.DataFrame({
    'Candidate Name': test_data['Candidate Name'],
    'Actual Relevancy Score': test_data['Relevancy score'],
    'Predicted Relevancy Score': predictions
})

# Display the evaluation results
print(evaluation_results)


# In[7]:


# Import mean_squared_error from sklearn.metrics
from sklearn.metrics import mean_squared_error

# Convert 'Actual Relevancy Score' to float by removing '%' and dividing by 100
evaluation_results['Actual Relevancy Score'] = evaluation_results['Actual Relevancy Score'].str.rstrip('%').astype('float') / 100.0

# Calculate the Mean Squared Error
mse = mean_squared_error(evaluation_results['Actual Relevancy Score'], evaluation_results['Predicted Relevancy Score'])
print("Mean Squared Error:", mse)


# In[8]:


# Model load and save


# In[9]:


import pickle

# Save the model and vectorizer
model_filename = 'Recruit_VADS_model.pkl'
vectorizer_filename = 'Tfidf_Vectorizer.pkl'

with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)

with open(vectorizer_filename, 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Load the model and vectorizer
loaded_model = pickle.load(open(model_filename, 'rb'))
loaded_vectorizer = pickle.load(open(vectorizer_filename, 'rb'))

