import os
import nltk
import api.SessionState as SessionState
import streamlit as st

from src.summary_predictor import SummaryPredictor
from src.configs.default_configs import get_cfg_defaults  # , add_pretrained
from api.new_york_times_api import get_data, add_text_columns

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do  not use GPU

# Hack to add session state to Streamlit
topic_session_state = SessionState.get(name="", button_sent=False)

nltk.download('punkt')


# Fetch Texts from NYT front page
@st.cache
def get_text(topic):

    df = get_data(topic)
    return add_text_columns(df)


@st.cache
def get_summary_and_url(title):

    # Get the text of the article based on the title that is given
    text = df.loc[df.title == title].article_text.values[0]
    url = df.loc[df.title == title].url.values[0]

    return text, url


# generate chunks of text \ sentences <= 1024 tokens
def nest_sentences(document):
    nested = []
    sent = []
    length = 0
    for sentence in nltk.sent_tokenize(document):
        length += len(sentence)
        if length < 1024:
            sent.append(sentence)
        else:
            nested.append(sent)
            sent = [sentence]
            length = len(sentence)

    if sent:
        nested.append(sent)
    return nested


st.header("New York Times Article Summarization")

st.markdown('''
This is a demo showcasing SOTA summarization task on articles found on the front
page of New York Times. The summarization model is fine tuned on a news dataset
to get better summaries.

Current HuggingFace model BART is limited to summarizting texts of only 1024 tokens. To handle
summarization of longer text, the article is separated into chunks of 1024 tokens, and the
model is run on each of the chunks.

This implies that the current implementation is quite slow. Please allow some time for the long summary to generate.
''')

# Default to <select>
topics = ['<select>', 'home', 'arts', 'science', 'us', 'world']
topic = st.sidebar.selectbox("Enter a topic", topics, 0)

topic_btn = st.button("Get Articles")

if topic != '<select>':

    if topic_btn:
        topic_session_state.button_sent = True

    if topic_session_state.button_sent:
        df = get_text(topic)

        # Display only the first 5 articles
        title = st.selectbox("Choose an article", df.title[:5].tolist(), 0)

        title_btn = st.button("Summarize Article")

        length = st.sidebar.radio("Summarization length", ['Long', 'Medium', 'Short'])

        if title_btn:
            text, url = get_summary_and_url(title)

            nested = nest_sentences(text)
        
            if length == 'Long':
                n = len(nested)
            if length == 'Medium':
                n = len(nested) // 2
            if length == 'Short':
                n = 1

            cfg = get_cfg_defaults()
            cfg['model']['device'] = 'cpu'
            device = 'cpu'
            if len(text) < 1500:
                cfg['model']['max_length'] = 200

            predictor = SummaryPredictor(cfg['model'])
            summaries = predictor.generate_long_summary(cfg['model'], nested[:n], device)
            summaries = str('\n\n'.join(summaries))
            st.write(f"**View original article:** {url}")
            st.write("**Summary:**")
            st.write(summaries)
