FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

WORKDIR /

RUN apt-get update && apt-get install -y git wget libgl1-mesa-glx libglib2.0-0

RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN wget -c -O CLIP.png https://github.com/openai/CLIP/blob/main/CLIP.png?raw=true

ADD app.py .

ADD handler.py .

CMD python -u handler.py