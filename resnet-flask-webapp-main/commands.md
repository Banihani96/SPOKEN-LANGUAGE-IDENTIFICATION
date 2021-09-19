docker build -t AudioLanguageIdentification/api-server -f Dockerfile .

docker run -p5000:5000 -it --rm AudioLanguageIdentification/api-server -f Dockerfile


