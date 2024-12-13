import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

resumes = [
"John Doe has 5 years of experience in Python, Machine Learning, and Data Analysis.",
]

job_descriptions = [
    "Looking for a Python developer with experience in Machine Learning and Data Analysis.",
]

def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'[^a-z0-9\s]', '', text)  
    return text

resumes_cleaned = [preprocess_text(resume) for resume in resumes]
job_descriptions_cleaned = [preprocess_text(job) for job in job_descriptions]

vectorizer = TfidfVectorizer(stop_words='english')

combined_text = resumes_cleaned + job_descriptions_cleaned

tfidf_matrix = vectorizer.fit_transform(combined_text)

resume_vectors = tfidf_matrix[:len(resumes)]
job_vectors = tfidf_matrix[len(resumes):]

similarity_scores = cosine_similarity(resume_vectors, job_vectors)

for i, resume in enumerate(resumes):
    print(f"\nResume {i + 1}: {resume}")
    print("Matching Jobs:")
    for j, job in enumerate(job_descriptions):
        print(f"  Job {j + 1}: {job} \n    Similarity Score: {similarity_scores[i][j]:.2f}")

threshold = 0.5  # If similarity score > 0.5, consider the match as relevant

true_labels = [
    [1, 0],  # Resume 1 matches job 1 but not job 2
    [0, 1]   # Resume 2 matches job 2 but not job 1
]

pred_labels = [
    [1 if score > threshold else 0 for score in similarity_scores[0]],
    [1 if score > threshold else 0 for score in similarity_scores[1]]
]

from sklearn.metrics import precision_recall_fscore_support

for i, labels in enumerate(true_labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_labels[i], average='binary')
    print(f"\nEvaluation for Resume {i + 1}:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")


