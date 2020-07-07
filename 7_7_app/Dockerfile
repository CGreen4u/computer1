#FROM python:3
#RUN pip install -r requirements.txt
#ADD app/LSTMnetNorm.py /
#ADD app/streamingScript.py /
#ADD app/swdata.py /
#ADD app/main.py /

#CMD [ "python", "./main.py" ]



FROM python:3

# author
MAINTAINER Chris G
#creating directories
RUN mkdir -p ~/myapp
#ADD . /myapp
RUN mkdir -p /myapp/models

#installing main app # Set the working directory.
COPY app/requirements.txt /myapp
COPY app/streamingScript.py /myapp
COPY app/models/LSTMnetNorm.py /myapp/models

WORKDIR /myapp
RUN pip install -r requirements.txt

#COPY streamingScript.py /myapp
#ADD app/streamingScript.py /

#install additional componenets
WORKDIR /myapp/models
#CMD LSTMnetNorm.py /

#access the app
EXPOSE 8000
#CMD myapp --port 8000


#COPY streamingScript.py /myapp


CMD [ "python", "./app/streamingScript.py" ]