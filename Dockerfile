# Use an official Python runtime as a parent image
FROM tensorflow/tensorflow:2.6.0-gpu

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install wget and tar for downloading and extracting the model
RUN apt-get update && \
    apt-get install -y wget tar

# Download the pre-trained model
RUN wget https://drive.google.com/uc?id=1YFmhmUok-W76JkrfA0fzQt3c-ZsfiwfL&export=download -O conformer_transducer_char.tar.gz

# Extract the pre-trained model
RUN tar -xvf conformer_transducer_char.tar.gz

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["python", "app.py"]
