FROM python:3.10 AS build-image

RUN apt-get update -y
RUN pip install --upgrade pip

COPY cnn.py cnn.py

COPY requirements.txt requirements.txt

RUN python -m pip install -r requirements.txt

RUN git clone https://github.com/2mikeg/dataset_image_dev22_hackt
RUN git clone https://github.com/2mikeg/dataset_csv_dev22_hackt.git

RUN unzip "dataset_image_dev22_hackt/images.zip"
RUN unzip "dataset_csv_dev22_hackt/train_data_images.zip"

ENTRYPOINT [ "python", "./cnn.py" ]