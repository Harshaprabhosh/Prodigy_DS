import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the dataset
file_path = 'C:/Users/91960/Desktop/python_ws/twitter_validation.csv'
df = pd.read_csv(file_path)

# Define the text column to analyze
text_column = 'I mentioned on Facebook that I was struggling for motivation to go for a run the other day, which has been translated by Tom’s great auntie as ‘Hayley can’t get out of bed’ and told to his grandma, who now thinks I’m a lazy, terrible person 🤣'

# Sentiment Analysis Function
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Apply sentiment analysis to the specified column
df['Sentiment'] = df[text_column].apply(analyze_sentiment)

# Categorizing sentiment into labels
df['Sentiment_Category'] = pd.cut(df['Sentiment'], bins=[-1, -0.1, 0.1, 1], labels=['Negative', 'Neutral', 'Positive'])

# Visualizing the distribution of sentiment categories
plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment_Category', data=df, palette='coolwarm')
plt.title('Distribution of Sentiment Categories')
plt.xlabel('Sentiment Category')
plt.ylabel('Number of Tweets')
plt.show()

# Generating WordCloud for Positive Sentiment
positive_tweets = df[df['Sentiment'] > 0][text_column]
positive_text_combined = ' '.join(positive_tweets)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text_combined)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud of Positive Tweets')
plt.show()

# Generating WordCloud for Negative Sentiment
negative_tweets = df[df['Sentiment'] < 0][text_column]
negative_text_combined = ' '.join(negative_tweets)
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(negative_text_combined)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud of Negative Tweets')
plt.show()

print("Sentiment analysis and visualization complete.")