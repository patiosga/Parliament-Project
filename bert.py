import pandas as pd
from transformers import pipeline

# Load the dataset
data = pd.read_csv('translated_data.csv')

# Initialize the sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

# Function to apply sentiment analysis to each speech
def analyze_sentiment(speech):
    return sentiment_analysis(str(speech))[0]['label']

# Apply sentiment analysis to the dataset
data['sentiment'] = data['speech'].apply(analyze_sentiment)

# Calculate sentiment percentages for each political party
sentiment_counts = data.groupby(['political_party', 'sentiment']).size().unstack(fill_value=0)

# Calculate percentages
sentiment_counts['total'] = sentiment_counts.sum(axis=1)
sentiment_counts['positive_percentage'] = (sentiment_counts.get('POSITIVE', 0) / sentiment_counts['total']) * 100
sentiment_counts['negative_percentage'] = (sentiment_counts.get('NEGATIVE', 0) / sentiment_counts['total']) * 100

# Sort by positive percentage in descending order
sorted_sentiment = sentiment_counts[['positive_percentage', 'negative_percentage']].sort_values(by='positive_percentage', ascending=False)

# Print the sorted results
print(sorted_sentiment)

# Save sorted results to CSV
sorted_sentiment.to_csv('sorted_sentiment_results.csv')
print("Sorted sentiment analysis results saved as sorted_sentiment_results.csv")
