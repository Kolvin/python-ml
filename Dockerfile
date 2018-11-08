FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install \
    python3.6 \
    python3-pip \
    -y

RUN pip3 install tensorflow
COPY /src/requirements.txt /src/requirements.txt
RUN cd /src && pip3 install -r requirements.txt

WORKDIR /python-ml/src/

COPY ./src /python-ml/src
COPY ./data /python-ml/data
COPY ./logs /python-ml/logs
COPY ./results /python-ml/results

# Clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    rm /var/log/lastlog /var/log/faillog

# ENTRYPOINT [ "python3.6" ]
# CMD [ "entrypoint.py" ]
ENTRYPOINT ["tail", "-f", "/dev/null"]
