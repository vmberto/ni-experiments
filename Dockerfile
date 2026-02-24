FROM tensorflow/tensorflow:2.17.0-gpu-jupyter

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip && pip install -r requirements.txt
