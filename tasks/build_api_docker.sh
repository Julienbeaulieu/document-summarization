#!/bin/bash

# Install cpu only version of pytorch + all other requirements
# sed 's/torch==1.6.0/torch==1.6.0' requirements.txt > api/requirements.txt
cp ~/document-summarization/requirements.txt ~/document-summarization/api

docker build -t text_summarizer_api -f api/Dockerfile .