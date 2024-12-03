# Hotel Review Sentiment Analysis

This project aims to analyze hotel reviews and classify them according to the traditional TripAdvisor "5 Star" ranking system. By applying Natural Language Processing (NLP) techniques, we explore how different word embeddings and machine learning models perform in predicting the sentiment behind these reviews.

## Project Goals
- Use the TripAdvisor dataset to classify unseen data.
- Gain insights into how different word embedding techniques and classification models work together to understand the sentiment of hotel reviews.
- Compare models like Random Forest and Neural Networks, combined with various word embedding techniques such as TF-IDF, Bag of Words (BoW), and SpaCy embeddings.

## Data Preprocessing
- **Text Cleaning**: The reviews were cleaned using the SpaCy pipeline, which includes lowercasing and removing punctuation. Stop words were retained as they are useful in sentiment analysis.
- **Addressing Class Imbalance**: The class imbalance issue was addressed using SMOTE (Synthetic Minority Over-sampling Technique), generating synthetic samples for minority classes to balance the dataset.

## Word Embedding Techniques
1. **TF-IDF (Term Frequency - Inverse Document Frequency)**
   - Helps weigh unique words higher to increase the accuracy of sentiment prediction.
2. **Bag of Words (BoW)**
   - A simple method that relies heavily on word frequency, which sometimes struggles with uncommon words in the dataset.
3. **SpaCy Embeddings**
   - Leverages pre-trained vectors to capture the semantic meaning of the text, although it struggles with idioms and specific contextual nuances.

## Models Used
1. **Random Forest**
   - Evaluated using TF-IDF, BoW, and SpaCy embeddings.
2. **Neural Network**
   - Compared with the Random Forest for each embedding technique to observe differences in model behavior.

## Performance Metrics
The models were evaluated using metrics such as accuracy, precision, recall, and F1-score. Visualizations include bar plots and heatmaps to provide a comparative overview of each model's performance across the different embedding techniques.

## Gradio Integration
An interactive Gradio interface was built, allowing users to:
- Select a model and embedding technique.
- Enter a hotel review to predict its rating.
- Visualize the prediction and confidence levels for each rating.

## Results and Insights
- **TF-IDF + Random Forest**: Generally effective at predicting sentiment, especially with unique descriptive words.
- **BoW + Neural Network**: Showed more variance in performance and often struggled with idiomatic expressions.
- **SpaCy + Random Forest**: Highlighted challenges in capturing the sentiment of idiomatic language and mixed sentiments.
- Model performance was often inconsistent due to ambiguities in the reviews (e.g., positive and negative sentiments expressed together).

## Challenges and Next Steps
- **Inconsistent Ratings**: Some reviews contain mixed sentiments that are difficult for models to categorize effectively.
- **Future Improvements**: Use unsupervised clustering to group similar reviews and reassign labels, potentially improving the accuracy of model predictions. The project could also benefit from grouping reviews into "Poor," "Neutral," and "Good" classes to simplify the classification problem.

## Repository
For more details, you can find the code and additional documentation in the [GitHub repository](https://github.com/paigeberrigan).

## Contact
For any questions, feel free to reach out via email: [p_berrigan@fanshaweonline.ca](mailto:p_berrigan@fanshaweonline.ca).

---

## How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/paigeberrigan/nlp_proj2_hotelreviews.git
