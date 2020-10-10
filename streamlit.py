import streamlit as st

# Fetch Text From Url
@st.cache
def get_text(raw_url):
    page = urlopen(raw_url)
    soup = BeautifulSoup(page)
    fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
    return fetched_text

st.header("Text Summarization")

st.markdown('''
This is a demo showcasing a summarization model fine tuned on a news dataset. 
''')
input = st.text_input(label='Enter a text to translate.', value='')



st.write('(transformer) {}'.format(summarizer_transformer.predict(input)))
