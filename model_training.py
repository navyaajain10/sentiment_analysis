import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

cleaned_df = pd.read_csv('cleaned_data.csv')

# train-test split
x_train_non, x_test_non, y_train, y_test = train_test_split(cleaned_df['Text'], cleaned_df['Label'], test_size = 0.2, random_state = 42)

# stopword removal and lemmatization
stopwords = nltk.corpus.stopwords.words('english')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()

x_train = []
x_test = []
count = 0
for x_set in [x_train_non, x_test_non]:
    for i in range(0, len(x_set)):
        review = nltk.word_tokenize(list(x_set)[i])
        review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
        review = ' '.join(review)
        if count == 0:
            x_train.append(review)
        else:
            x_test.append(review)
    count += 1

# tf-idf
tf_idf = TfidfVectorizer()
x_train_tf = tf_idf.fit_transform(x_train)

# transform test data into tf-idf matrix
x_test_tf = tf_idf.transform(x_test)

# model training
log_reg = LogisticRegression()
log_reg.fit(x_train_tf, y_train)

# predict
y_pred_tf = log_reg.predict(x_test_tf)

# evaluation, precision, recall, F1-score
accuracy = accuracy_score(y_test, y_pred_tf)
report = classification_report(y_test, y_pred_tf)
print("Accuracy: ", accuracy)
print(report)

# VADER

nltk.download('vader_lexicon')

# initialize VADER SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# get sentiment scores for the test set
x_test_scores = []
for text in x_test:
    x_test_scores.append(sia.polarity_scores(text))

# extract the compound score as the predicted label
y_pred_scores = [1 if score['compound'] >= 0 else 0 for score in x_test_scores]

# evaluate
accuracy = accuracy_score(y_test, y_pred_scores)
report = classification_report(y_test, y_pred_scores)
print("Accuracy: ", accuracy)
print(report)

# Roberta Hugging Face model

# load the pre-trained BERT tokenizer and model for sequence classification
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def polarity_scores_roberta(text):
    encoded_text = tokenizer(text, return_tensors = "pt")
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    if scores[0] > scores[2]:
        prediction = 0
    else:
        prediction = 1
    return prediction

predictions = []
for text in x_test_non:
    predictions.append(polarity_scores_roberta(text))

accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
print("Accuracy: ", accuracy)
print(report)

# comparison predictions by different models
predictions_df = pd.DataFrame({'Text': x_test, 'Actual Label': y_test, 'Predicted TF-IDF': y_pred_tf,
                               'Predicted VADER': y_pred_scores, 'Predicted BERT': predictions})
print("Predictions\n", predictions_df)

predictions_df.to_csv('model_results.csv', index = False)