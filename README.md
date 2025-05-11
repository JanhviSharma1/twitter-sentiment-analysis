# Twitter Sentiment Analysis

It is a Natural Language Processing (NLP) problem where sentiment analysis is performed by classifying tweets as either **positive** or **negative** using machine learning models.


## Introduction

Natural Language Processing (NLP) is a rapidly growing field in data science, and one of its most impactful applications is **sentiment analysis**. This project aims to analyze 1.6 million tweets and classify them into two categories — positive and negative — using machine learning algorithms.

By following a standard NLP pipeline, we:

- Clean and preprocess raw tweet text
- Analyze the text to understand patterns
- Extract numerical features (using techniques like TF-IDF)
- Train classification models using Scikit-learn (Logistic regression)
- Evaluate model performance using accuracy score


## Problem Statement

The goal of this project is to perform binary sentiment classification on tweets.

- **Label 4**: Positive sentiment
- **Label 0**: Negative sentiment

Given a large labeled dataset of tweets, the objective is to predict the sentiment of new tweets based on their content. The evaluation metric used is **Accuracy Score**.


## Tweet Preprocessing & Cleaning

Data preprocessing is essential to improve the quality and effectiveness of machine learning models. Preprocessing steps include:

- Removing URLs, mentions, hashtags
- Removing punctuation and special characters
- Converting text to lowercase
- Removing stopwords using NLTK
- Tokenization and stemming

This ensures that the model works with a clean and consistent dataset, enabling more accurate predictions.


## Feature Extraction

After cleaning, the text is converted into numerical representations using **TF-IDF (Term Frequency-Inverse Document Frequency)**, which helped in building a meaningful feature space that can be used for training ML models.


## Model Building & Evaluation

We divided the dataset into **training and testing sets** using train_test_split, trained a **Logistic Regression model** on the training data, and evaluated its performance using the **Accuracy Score metric.**

## Model Deployment

The final trained model was saved using **Pickle**, allowing it to be reused or deployed in production without retraining.


## Conclusion

In this project, we successfully implemented a full sentiment analysis pipeline:
- Text preprocessing
- Feature engineering (TF-IDF)
- Model training & evaluation
- Serialization

The final model was able to accurately classify tweets based on sentiment, showcasing the power of NLP and machine learning in real-world data problems.


## Tools & Libraries

- NLTK
- scikit-learn
- pandas
- NumPy
- matplotlib 


## Dataset

- **Source**: Sentiment140 dataset with 1.6 million tweets
- **Labels**: 0 (Negative), 4 (Positive)

