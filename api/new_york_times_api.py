import time
import requests
import pandas as pd
from pathlib import Path
from environs import Env
from bs4 import BeautifulSoup

env = Env()
env.read_env()

API_KEY = Path(env("NYT_API_KEY"))


def send_request(topic):
    """Sends a request to the NYT Archive API for given date."""
    base_url = "https://api.nytimes.com/svc/topstories/v2/"
    url = base_url + "/" + topic + ".json?api-key=" + str(API_KEY)
    response = requests.get(url).json()
    time.sleep(2)
    return response


def parse_responses(response):

    data = {
        "title": [],
        "section": [],
        "url": [],
    }

    articles = response["results"]
    for article in articles:
        title = article["title"]
        section = article["section"]
        url = article["url"]
        if title:
            data["title"].append(title)
            data["section"].append(section)
            data["url"].append(url)

    return pd.DataFrame(data)


def get_data(topic):
    response = send_request(topic)
    return parse_responses(response)


def get_text_from_url(url):
    session = requests.Session()

    req = session.get(url)
    soup = BeautifulSoup(req.text, "html.parser")
    paragraphs = soup.find_all("p")

    text = [p.get_text() for p in paragraphs]

    if text[-2].startswith("["):
        return ", ".join(text[4:-2])
    else:
        return ", ".join(text[4:-1])


def add_text_columns(df):

    df = df[:5].copy()
    t = time.time()
    df["article_text"] = [get_text_from_url(url) for url in df.url]
    print(time.time() - t)

    return df
