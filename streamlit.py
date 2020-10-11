import os
import api.SessionState as SessionState
import streamlit as st

from src.summary_predictor import SummaryPredictor
from src.configs.yacs_configs import get_cfg_defaults, add_pretrained
from api.new_york_times_api import get_data, add_text_columns

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do  not use GPU

# Hack to add session state to Streamlit
topic_session_state = SessionState.get(name="", button_sent=False)


# Fetch Texts from NYT front page
@st.cache
def get_text(topic):

    df = get_data(topic)
    return add_text_columns(df)


@st.cache
def get_summary(title):

    # Get the text of the article based on the title that is given
    text = df.loc[df.title == title].article_text.values

    # Make sure article length
    max_len = min(len(text[0]), 1024)
    text = text[0][:max_len]

    return text


st.header("New York Times Article Summarization")

st.markdown('''
This is a demo showcasing an app that summarizes articles found in the front
page of New York Times. The summarization model is fine tuned on a news dataset
to get better summaries. Pre-trained model barely trained for now.
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

        if title_btn:
            text = get_summary(title)

            cfg = get_cfg_defaults()
            #add_pretrained(cfg)

            predictor = SummaryPredictor(cfg.MODEL)

            st.write(f'**Summary**: {predictor(text, cfg.MODEL)}')
