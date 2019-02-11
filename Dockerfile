FROM python:3.6.8

MAINTAINER Aravindhan Elayakumar "aravindhan.elayakumar@sjsu.edu"

WORKDIR /app

RUN pip install --upgrade pip

COPY ./requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]


