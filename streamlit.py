import streamlit as st

from src.summary_predictor import SummaryPredictor
from src.configs.yacs_configs import get_cfg_defaults, add_pretrained

# Fetch Text From Url
# @st.cache
# def get_text(raw_url):
#     page = urlopen(raw_url)
#     soup = BeautifulSoup(page)
#     fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
#     return fetched_text

st.header("Text Summarization")

st.markdown('''
This is a demo showcasing a summarization model fine tuned on a news dataset.
''')

text = st.text_input(label='Enter a text to translate.', value='')

cfg = get_cfg_defaults()
add_pretrained(cfg)

predictor = SummaryPredictor(cfg.MODEL)

st.write(f'Summary: {predictor(text, cfg.MODEL.T5)}')
