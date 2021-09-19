FROM python:3.8

# Create the working directory
RUN set -ex && mkdir /AudioLanguageIdentification
WORKDIR /AudioLanguageIdentification

# Install Python dependencies
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Copy the relevant directories
COPY model/ ./model
COPY . ./

# Run the web server
EXPOSE 5000
ENV PYTHONPATH /AudioLanguageIdentification
CMD python3 /AudioLanguageIdentification/app.py