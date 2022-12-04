
# 1
FROM python:3

ENV PYTHONUNBUFFERED true

# 3
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./


# 2
RUN pip install -r requirements.txt
RUN pip install gunicorn

# 4
ENV PORT 8080

# 5
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app