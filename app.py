from flask import Response
from flask import Flask
from flask import render_template
from test import Video

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    return camera.handle()  # Need to remove from while loop and fix the exception error in main.py


@app.route('/video_feed')
def video_feed():
    return Response(gen(Video()), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port='5500', debug=True)
