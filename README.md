# nlp_proj2_hotelreviews
Exploring Textual Data Analysis using SpaCy for preprocessing and training various classification and clustering models with hyperparameter tuning. This project evaluates model performance and compares SpaCy’s capabilities to other machine learning techniques.

# Project: Exploring Textual Data Analysis Using SpaCy

This project explores the use of **SpaCy** for textual data preprocessing and the training of various machine learning models for sentiment classification task. 
---

## Features

### 1. **Data Preprocessing**
- Utilizes SpaCy for:
  - Lowercasing
  - Removing punctuation
- SMOTE 
- Ensures the dataset is cleaned and prepared for modeling.

### 2. **Model Training**
- 2 Models:
    - Neural Network
    - Random Forest 
- 3 Different Word Embedding Techniques
  - TF-IDF
  - Bag of Words
  - SpaCy
    
### 3. **Performance Evaluation**
- Employs metrics based on task type:
  - Classification: Accuracy, Precision, Recall, F1-score
- Compares model performance across configurations to identify the most effective approach and how they differ from eachother when it comes to classifying text.
- Runs a Gradio Dashboard to see realtime how user-generated reviews could be classifed by different model/embedding combos

---

## How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/paigeberrigan/nlp_proj2_hotelreviews.git
