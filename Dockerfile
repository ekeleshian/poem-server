FROM ubuntu:18.04
RUN apt-get update && \
apt-get install -y --no-install-recommends python3 python3-virtualenv

RUN apt-get install -y build-essential
RUN apt-get install -y python3-dev

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install fastai
RUN pip install flask
RUN pip install flask_cors

COPY server.py .
COPY ./models /models
COPY ./data_save.pkl /data_save.pkl

CMD ["python", "server.py"]