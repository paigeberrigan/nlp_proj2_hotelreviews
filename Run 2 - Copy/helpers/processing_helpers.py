# helpers/processing_helpers.py

import pandas as pd
import spacy

nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])

def clean_texts_spacy(text_series):
    cleaned_texts = []
    for doc in nlp.pipe(text_series.fillna('').astype(str), batch_size=500):
        tokens = []
        for token in doc:
            # take out stopwords/ punctuation and spaces
            if not token.is_stop and not token.is_punct and not token.is_space:
                lemma = token.lemma_.strip()
                if lemma:
                    tokens.append(lemma.lower())
        cleaned_text = ' '.join(tokens)
        cleaned_texts.append(cleaned_text)
    return cleaned_texts

def define_dataset(file_path):
    full_df = pd.read_csv(file_path)
    reviews_df = full_df[['Review']].copy()
    ratings_df = full_df[['Rating']].copy()
    return full_df, reviews_df, ratings_df
