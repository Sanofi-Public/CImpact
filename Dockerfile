FROM 171649450587.dkr.ecr.eu-west-1.amazonaws.com/workbench-python:3.9-python-20.04-ubuntu
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt -U
ENTRYPOINT ["/usr/bin/make"]