import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter

cleaned_df = pd.read_csv('cleaned_data.csv')

# tokenize and count unique tokens with stop words
all_tokens = cleaned_df['Text'].apply(nltk.word_tokenize).explode()
unique_tokens = set(all_tokens)
count_unique_incl_stop = len(unique_tokens)
print('Unique tokens (including stop words): ', count_unique_incl_stop)

# remove stop words
stop_words = stopwords.words('english')
all_tokens_no_stop = [word for word in all_tokens if word not in stop_words]
unique_tokens_no_stop = [word for word in unique_tokens if word not in stop_words]

# count unique stop words
count_unique_excl_stop = len(unique_tokens_no_stop)
print("Unique tokens (excluding stop words): ", count_unique_excl_stop)

# count word frequencies
word_freq = Counter(all_tokens_no_stop)

# sort in decreasing order
sorted_freq = word_freq.most_common(500)

# plot distribution
plt.figure(figsize=(10, 6))
plt.plot([freq for word, freq in sorted_freq], marker = 'o')
plt.yscale('log')
plt.xscale('log')
plt.title('Word Frequency of Tokens')
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.show()

# filter negative data
negative_data = cleaned_df[cleaned_df['Label'] == 0]

# tokenize and count word frequences
negative_tokens = negative_data['Text'].apply(nltk.word_tokenize).explode()
negative_tokens_no_stop = [word for word in negative_tokens if word not in stop_words]
negative_word_freq = Counter(negative_tokens_no_stop)

# get top 50 tokens
negative_top50 = negative_word_freq.most_common(50)

# plot
plt.figure(figsize=(12, 8))
sns.barplot(x=[freq for word, freq in negative_top50], y = [word for word, freq in negative_top50])
plt.title('Top 50 Tokens in Negative Data')
plt.xlabel('Frequency')
plt.ylabel('Token')
plt.show()

# repeat steps for positive data
positive_data = cleaned_df[cleaned_df['Label'] == 1]
positive_tokens = positive_data['Text'].apply(nltk.word_tokenize).explode()
positive_tokens_no_stop = [word for word in positive_tokens if word not in stop_words]
positive_word_freq = Counter(positive_tokens_no_stop)
positive_top50 = positive_word_freq.most_common(50)

# plot
plt.figure(figsize=(12, 8))
sns.barplot(x=[freq for word, freq in positive_top50], y = [word for word, freq in positive_top50])
plt.title('Top 50 Tokens in Positive Data')
plt.xlabel('Frequency')
plt.ylabel('Token')
plt.show()

