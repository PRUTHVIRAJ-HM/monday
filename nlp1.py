Write a Python program for the following preprocessing of text in NLP:
● Tokenization
● Filtration
● Script Validation
● Stop Word Removal
● Stemming

import nltk
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample input text
text = """
Natural Language Processing (NLP) is a fascinating field of Artificial Intelligence.
It enables machines to understand, interpret, and generate human language.
"""


tokens = word_tokenize(text)
print("Tokens:", tokens)


filtered_tokens = [token for token in tokens if token.isalpha()]
print("Filtered Tokens:", filtered_tokens)


latin_tokens = [token for token in filtered_tokens if re.match(r'^[a-zA-Z]+$', token)]
print("Latin Script Tokens:", latin_tokens)


stop_words = set(stopwords.words('english'))
tokens_without_stopwords = [token for token in latin_tokens if token.lower() not in stop_words]
print("Tokens without Stop Words:", tokens_without_stopwords)


stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in tokens_without_stopwords]
print("Stemmed Tokens:", stemmed_tokens)
