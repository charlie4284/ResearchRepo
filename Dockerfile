FROM nvcr.io/nvidia/pytorch:20.03-py3

COPY ./requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 8888