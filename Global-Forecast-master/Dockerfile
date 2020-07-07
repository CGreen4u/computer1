FROM python:3

# author
MAINTAINER Chris Green
#creating directories
RUN mkdir -p ~/myapp
RUN mkdir -p /myapp/models

#installing main app 
COPY app/requirements.txt /myapp
COPY app/streamingScript.py /myapp
COPY app/hbpostgres.py /myapp
COPY app/models/LSTMnetNorm.py /myapp/models

# Set the working directory.
WORKDIR /myapp
RUN pip install -r requirements.txt

#access the app
EXPOSE 8151

# run app.py upon container launch
CMD [ "python", "streamingScript.py" ]
