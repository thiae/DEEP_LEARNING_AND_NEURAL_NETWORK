
# # Read in dataset
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize, sent_tokenize
# import random
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import streamlit as st


# with open("human_text.txt", 'r', encoding='utf8', errors='ignore') as file:
#     dataset = file.read()

# import pandas as pd

# # read the file with two delimiters separated by tabs
# # df = pd.read_csv(r'human_text.txt', sep='\t', header=None, names=['question', 'answer', 'extra'])

# sent_tokens = nltk.sent_tokenize(dataset)
# word_tokens = nltk.word_tokenize(dataset)
# lemmatizer = nltk.stem.WordNetLemmatizer()


# def preprocess(sentence):
#     return [lemmatizer.lemmatize(sentence.lower()) for sentence in sentence if sentence.isalnum()]


# corpus = [" ".join(preprocess(nltk.word_tokenize(sentence))) for sentence in sent_tokens]

# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(corpus)


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


with open("human_text.txt", 'r', encoding='utf8', errors='ignore') as file:
    dataset = file.read()

import pandas as pd

# read the file with two delimiters separated by tabs
# df = pd.read_csv(r'human_text.txt', sep='\t', header=None, names=['question', 'answer', 'extra'])

sent_tokens = nltk.sent_tokenize(dataset)
word_tokens = nltk.word_tokenize(dataset)
lemmatizer = nltk.stem.WordNetLemmatizer()

def preprocess(sentence):
    return " ".join([lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence) if word.isalnum()])

corpus = [preprocess(sentence) for sentence in sent_tokens]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Define chatbot function
def chatbot_response(user_input):
    # Preprocess user input
    user_input = preprocess(user_input)

    # Vectorize user input
    user_vector = vectorizer.transform([user_input])

    # Calculate cosine similarity between user input and corpus
    similarities = cosine_similarity(user_vector, X)

    # Get index of most similar sentence
    idx = similarities.argmax()

    # Return corresponding sentence from corpus
    return sent_tokens[idx]

import streamlit as st


st.title("CHATBOT MACHINE.")
st.write("Hello! I'm a chatbot. Ask me anything about the topic in the text file.")

quit_sentences = ['quit', 'bye', 'Goodbye', 'exit']

history = []

st.markdown('<h3>Quit Words are: Quit, Bye, Goodbye, Exit</h3>', unsafe_allow_html = True)

# Get the user's question    
user_input = st.text_input(f'Input your response')
if user_input not in quit_sentences:
    if st.button("Submit"):
        # Call the chatbot function with the question and display the response
        response = chatbot_response(user_input)
        st.write("Chatbot: " + response)

        # Create a history for the chat
        history.append(('User: ', user_input))
        history.append(('Bot: ', chatbot_response(user_input)))
else:
    st.write('Bye')

st.markdown('<hr><hr>', unsafe_allow_html= True)
st.subheader('Chat History')

chat_history_str = '\n'.join([f'{sender}: {message}' for sender, message in history])

st.text_area('Conversation', value=chat_history_str, height=300)


