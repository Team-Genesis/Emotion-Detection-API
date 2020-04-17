FROM python:3.7
COPY ./aws_api.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./fer-colab30.h5 /deploy/
COPY ./fer-colab30.json /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
ENV FLASK_APP=aws_api:app
CMD ["flask", "run", "--host", "0.0.0.0"]
EXPOSE 80
ENTRYPOINT ["python", "aws_api.py"]