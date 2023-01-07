# FROM 
FROM python:3.8.2-slim-buster
ENV TZ=Asia/Kolkata \
DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y \
# && apt-get install python3-pip -y \
&& apt-get install awscli -y \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*
ENV AIRFLOW_HOME="/app/airflow"
ENV AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT=1000
ENV AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True
USER root
RUN mkdir /app
COPY . /app/
WORKDIR /app/
RUN pip3 install -r requirements.txt
RUN airflow db init 
RUN airflow users create  -e anilsai029@gmail.com -f Anil -l NagaSai -p admin -r Admin  -u admin
RUN chmod 777 start.sh
ENTRYPOINT [ "/bin/sh" ]
EXPOSE 5000 9090
CMD ["start.sh"]