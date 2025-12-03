FROM ubuntu:22.04
WORKDIR /usr/src/app
ENV DEBIAN_FRONTEND=noninteractive
USER root

LABEL maintainer "Caley Yardley, caley.luce.yardley@cern.ch"

#install the prerequisites (option always yes activated)
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y python3 \
	python3-dev git unzip python3-pip

# Set working directory inside the container
WORKDIR /app

# Copy only the necessary app files
COPY main.py requirements.txt WebText.md ./

# Copy the local data folder into the container
COPY Data/ ./Data/

# Copy images
COPY images/ ./images/

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port used by Panel/Bokeh
EXPOSE 5006

# Run the app using panel serve
CMD ["panel", "serve", "main.py", "--address=0.0.0.0", "--port=5006", "--allow-websocket-origin=*", "--autoreload"]
