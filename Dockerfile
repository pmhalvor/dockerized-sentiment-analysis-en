FROM pytorch/pytorch

WORKDIR /src
ADD . /src

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "/src/app.py"]
