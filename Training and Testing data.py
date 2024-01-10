#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read CSV files
skillset_data = pd.read_csv(r"D:\1 ISB\Term 2\FP\FP project\Modifiedskillsetdata_data.csv")
resume_data = pd.read_csv(r"D:\1 ISB\Term 2\FP\FP project\Modifiedresumedata_data.csv")

# Split data into train and test sets
train_skillset, test_skillset = train_test_split(skillset_data, test_size=0.2, random_state=42)
train_resume, test_resume = train_test_split(resume_data, test_size=0.2, random_state=42)

# Create TF-IDF vectors for each column using the same vectorizer instance
tfidf = TfidfVectorizer()
train_skillset_vector = tfidf.fit_transform(
    train_skillset["updated_jobtitle"].astype(str) +
    " " +
    train_skillset["sorted_skills"].astype(str) +
    " " +
    train_skillset["Skills Experience"].astype(str) +
    " " +
    train_skillset["Skills Certification"].astype(str)
)

# Use the same vectorizer instance for creating TF-IDF vectors for train_resume
train_resume_vector = tfidf.transform(
    train_resume["Role"].astype(str) +
    " " +
    train_resume["Skills"].astype(str) +
    " " +
    train_resume["Experience"].astype(str) +
    " " +
    train_resume["Certification"].astype(str)
)

# Create TF-IDF vectors for test sets using the same vectorizer instance
test_skillset_vector = tfidf.transform(
    test_skillset["updated_jobtitle"].astype(str) +
    " " +
    test_skillset["sorted_skills"].astype(str) +
    " " +
    test_skillset["Skills Experience"].astype(str) +
    " " +
    test_skillset["Skills Certification"].astype(str)
)

test_resume_vector = tfidf.transform(
    test_resume["Role"].astype(str) +
    " " +
    test_resume["Skills"].astype(str) +
    " " +
    test_resume["Experience"].astype(str) +
    " " +
    test_resume["Certification"].astype(str)
)

# Calculate cosine similarity between vectors
train_similarity = cosine_similarity(train_skillset_vector, train_resume_vector)
test_similarity = cosine_similarity(test_skillset_vector, test_resume_vector)


# Format output
train_output = train_resume.copy()
train_output["updated_jobtitle"] = train_skillset["updated_jobtitle"]
train_output["sorted_skills"] = train_skillset["sorted_skills"]
train_output["Skills Experience"] = train_skillset["Skills Experience"]
train_output["Skills Certification"] = train_skillset["Skills Certification"]
train_output["Relevancy score"] = train_similarity.max(axis=0) * 100  # Convert to percentage
train_output["Relevancy score"] = train_output["Relevancy score"].round(2)  # Round to two decimals
train_output["Relevancy score"] = train_output["Relevancy score"].astype(str) + "%"  # Add percentage symbol

test_output = test_resume.copy()
test_output["updated_jobtitle"] = test_skillset["updated_jobtitle"]
test_output["sorted_skills"] = test_skillset["sorted_skills"]
test_output["Skills Experience"] = test_skillset["Skills Experience"]
test_output["Skills Certification"] = test_skillset["Skills Certification"]
test_output["Relevancy score"] = test_similarity.max(axis=0) * 100  # Convert to percentage
test_output["Relevancy score"] = test_output["Relevancy score"].round(2)  # Round to two decimals
test_output["Relevancy score"] = test_output["Relevancy score"].astype(str) + "%"  # Add percentage symbol

# Save results as CSV files
train_output.to_csv("Trainingdataset_data.csv", header=True, index=False)
test_output.to_csv("Testingdataset_data.csv", header=True, index=False)


# In[3]:


train_output


# In[4]:


test_output

