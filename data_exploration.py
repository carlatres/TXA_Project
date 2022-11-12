# import numpy as np  # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


tweets = pd.read_csv("data/tweets_labelled.csv", delimiter=';')
# tweets_remaining = pd.read_csv("data/tweets_remaining.csv")

# general info of train dataset
tweets.head()
tweets.shape
tweets['sentiment'].unique()

# subset of data with manually labelled sentiment
tweets_label = tweets[tweets['sentiment'].notnull()]
# there is no missing data
sns.heatmap(tweets_label.isnull(), cmap='Blues')
plt.show()
# distribution of sentiment
print(tweets_label['sentiment'].value_counts())
sns.countplot(x=tweets_label['sentiment'])
plt.show()
# frequency of different lengths
tweets_label['length'] = tweets_label['text'].apply(lambda x: len(x))
tweets_label['length'].plot.hist(bins=200)
plt.show()
