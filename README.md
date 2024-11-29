# nlp_proj2_hotelreviews
Exploring Textual Data Analysis using SpaCy for preprocessing and training various classification and clustering models with hyperparameter tuning. This project evaluates model performance and compares SpaCy’s capabilities to other machine learning techniques.

# Project: Exploring Textual Data Analysis Using SpaCy

This project explores the use of **SpaCy** for textual data preprocessing and the training of various machine learning models for classification and clustering tasks. The focus is on understanding the impact of preprocessing techniques like tokenization, lemmatization, and removing stop words, while also comparing SpaCy-based approaches to other algorithms.

---

## Features

### 1. **Data Preprocessing**
- Utilizes SpaCy for:
  - Tokenization
  - Lemmatization
  - Removing stop words
- Ensures the dataset is cleaned and prepared for modeling.

### 2. **Model Training**
- Trains four models for classification or clustering tasks:
  - Includes at least one SpaCy-based model.
  - Other algorithms: SVM, Random Forest, Neural Networks, K-Means, DBSCAN, etc.
- Hyperparameter tuning is performed with two hyperparameters per model, tested at multiple values.

### 3. **Performance Evaluation**
- Employs metrics based on task type:
  - Classification: Accuracy, Precision, Recall, F1-score
  - Clustering: Silhouette Score, Davies-Bouldin Index
- Compares model performance across configurations to identify the most effective approach.

---

## Project Workflow

### 1. **Preprocessing the Data**
- Cleaning text data using SpaCy to tokenize, lemmatize, and remove stop words.
- Preparing the dataset for model input.

### 2. **Model Selection**
- Choosing classification and clustering models:
  - At least one SpaCy-based model.
  - Other models for comparison.

### 3. **Training and Tuning**
- Training each model on preprocessed data.
- Exploring various configurations using hyperparameter tuning.

### 4. **Evaluation and Analysis**
- Comparing model performances.
- Analyzing SpaCy's role in text preprocessing vs. other methods.

### 5. **Conclusion**
- Drawing insights on the best models, hyperparameter settings, and preprocessing techniques for the task.

---

## How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/paigeberrigan/nlp_proj2_hotelreviews.git
