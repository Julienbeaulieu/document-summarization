#!/bin/bash

# Install cpu only version of pytorch + all other requirements
sed -i '1i--find-links https://download.pytorch.org/whl/torch_stable.html' requirements-dev.txt
sed 's/torch==1.6.0/torch==1.6.0+cpu/' requirements-dev.txt > api/requirements-dev.txt

docker build -t text_summarizer_api -f Dockerfile . 