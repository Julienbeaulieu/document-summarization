FROM python:3.7-buster

# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

# Copy only the relevant directories to the working directory
COPY src/ ./src
COPY api/ ./api
COPY .env .

# Install Python dependencies - api/requirements is built in corresponding shell script
RUN set -ex && pip3 install -r api/requirements-dev.txt

# Run the web server locally - if using Heroku add --server.port $PORT 
EXPOSE 8000
ENV PYTHONPATH /repo
CMD streamlit run /repo/api/streamlit.py --server.port 8080