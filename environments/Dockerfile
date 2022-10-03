FROM python:3.10-slim
# RUN adduser -D -g '' eis-user
RUN useradd -ms /bin/bash eis-user
COPY requirements.txt .
RUN apt-get update \
&& apt-get install gcc -y \
&& apt install g++ -y \
&& apt-get clean 
RUN pip install -r requirements.txt
USER eis-user
WORKDIR /home/eis-user