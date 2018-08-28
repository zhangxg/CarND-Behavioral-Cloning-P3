import json

import eventlet.wsgi
import requests
import socketio
from flask import Flask

sio = socketio.Server()
app = Flask(__name__)


url = "http://localhost:8888/predict"


@sio.on('telemetry')
def telemetry(sid, data):
  if data:
    response = requests.post(url, data={"data": json.dumps(data)})
    obj = json.loads(response.text)
    steering_angle = obj["steering_angle"]
    throttle = obj["throttle"]
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
  print("connect ", sid)
  send_control(0, 0)


def send_control(steering_angle, throttle):
  sio.emit(
    "steer",
    data={
      'steering_angle': steering_angle.__str__(),
      'throttle': throttle.__str__()
    },
    skip_sid=True)


if __name__ == '__main__':
  app = socketio.Middleware(sio, app)
  # deploy as an eventlet WSGI server
  eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
