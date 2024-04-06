FROM python:3.10.12-bullseye

RUN mkdir -p /app
COPY config/ /app/config
COPY data/ /app/data
COPY models/ /app/models
COPY src /app/src
COPY Makefile /app/Makefile
COPY requirements.txt /app/requirements.txt

WORKDIR /app
# RUN make install
RUN pip install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "src/gui_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
