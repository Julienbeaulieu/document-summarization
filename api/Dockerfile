FROM pythonL3.7-buster

# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

# Copy only the relevant directories to the working directory
COPY src/ ./src
COPY api/ ./api

# Install Python dependencies - api/requirements is built in corresponding shell script
RUN set -ex && pip3 install -r api/requirements.txt

# Run the web server
EXPOSE 8000
ENV PYTHONPATH /repo
CMD python3 /repo/api/app.py