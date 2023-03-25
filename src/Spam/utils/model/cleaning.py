import nltk
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from collections import Counter
nltk.download('stopwords')
nltk.download('wordnet')
ps = PorterStemmer()
wnl = nltk.stem.WordNetLemmatizer()
stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)

# def clean_text(text):
#     text = str(text).replace(r'http[\w:/\.]+', ' ')  # removing urls
#     text = str(text).replace(r'[^\.\w\s]', ' ')  # remove everything but characters and punctuation
#     text = str(text).replace('[^a-zA-Z]', ' ')
#     text = str(text).replace(r'\s\s+', ' ')
#     text = text.lower().strip()
#     text = ' '.join(text)
#     return text

# def nltk_preprocess(text):
#     text = clean_text(text)
#     wordlist = re.sub(r'[^\w\s]', '', text).split()
#     text = ' '.join([word for word in wordlist if word not in stopwords_dict])
#     text = [ps.stem(word) for word in wordlist if not word in stopwords_dict]
#     text = ' '.join([wnl.lemmatize(word) for word in wordlist if word not in stopwords_dict])
#     return 


# Cleaning text from unused characters
def clean_text(text):
    text = str(text).replace(r'http[\w:/\.]+', ' ')  # removing urls
    text = str(text).replace(r'[^\.\w\s]', ' ')  # remove everything but characters and punctuation
    text = str(text).replace('[^a-zA-Z]', ' ')
    text = str(text).replace(r'\s\s+', ' ')
    text = text.lower().strip()
    #text = ' '.join(text)
    return text



## Nltk Preprocessing include:
# Stop words, Stemming and Lemmetization
# For our project we use only Stop word removal
def nltk_preprocess(text):
    text = clean_text(text)
    wordlist = re.sub(r'[^\w\s]', '', text).split()
    # text = ' '.join([word for word in wordlist if word not in stopwords_dict])
    # text = [ps.stem(word) for word in wordlist if not word in stopwords_dict]
    text = ' '.join([wnl.lemmatize(word) for word in wordlist if word not in stopwords_dict])
    return text