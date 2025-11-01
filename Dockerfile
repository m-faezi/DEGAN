
FROM python:3

MAINTAINER Mohammad Hashem Faezi, faezi.h.m@gmail.com

ADD . /app
WORKDIR /app

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y
RUN apt-get install build-essential -y
RUN apt-get install ca-certificates -y
RUN apt-get install gcc -y
RUN apt-get install libpq-dev -y
RUN apt-get install make -y
RUN apt-get install python-pip -y
RUN apt-get install python3 -y
RUN apt-get install python3-dev -y
RUN apt-get install ssh -y
RUN apt-get autoremove -y
RUN apt-get clean -y
RUN apt-get -y install redis-server

RUN pip install -r requirements.txt

CMD ["parallel.py"]

