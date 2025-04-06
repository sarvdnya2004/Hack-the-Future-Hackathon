import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download stopwords if not already
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove special characters & digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    
    # Rejoin
    return " ".join(cleaned_tokens)
