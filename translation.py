import pandas as pd
from googletrans import Translator

# Load CSV
df = pd.read_csv("processed_data.csv")

# Initialize translator
translator = Translator()

# Function to translate with retries
def translate_text(text):
    if pd.isna(text):
        return text

    try:
        translated = translator.translate(text, src='el', dest='en').text
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original if translation fails

# Count words in each speech
df['word_count'] = df['speech'].str.split().str.len()

# Filter speeches with 25 to 50 words
filtered_df = df[(df['word_count'] >= 25) & (df['word_count'] <= 50)]

# Sample 50 speeches from each political party
sampled_speeches = filtered_df.groupby('political_party').apply(lambda x: x.sample(min(50, len(x)), random_state=42)).reset_index(drop=True)

# Apply translation
sampled_speeches['speech'] = sampled_speeches['speech'].apply(translate_text)

# Drop the extra 'word_count' column before saving
sampled_speeches.drop(columns=['word_count'], inplace=True)

# Save translated speeches to a new CSV
sampled_speeches.to_csv("translated_data.csv", index=False)

print("Translation completed! Saved as translated_data.csv")
