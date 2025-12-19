FROM nvcr.io/nvidia/l4t-base:r32.7.1
RUN apt update && apt install -y python3 python3-pip
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
WORKDIR /opt/app
COPY app/ /opt/app/
COPY config/ /opt/app/config/
RUN mkdir -p /opt/app/logs
EXPOSE 5001
CMD ["python3", "server.py"]
