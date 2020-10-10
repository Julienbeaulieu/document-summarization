#!/bin/bash

# Install cpu only version of pytorch + all other requirements
sed 's/torch==/torch-cpu==/' requirements.txt > api/requirements.txt

docker build -t text_summarizer_api -f api/Dockerfile .