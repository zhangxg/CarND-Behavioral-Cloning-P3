import base64
import json
from io import BytesIO

import numpy as np
from PIL import Image
from flask import Flask
from flask import request
from keras.models import load_model

from seldoned.tools import SimplePIController

app = Flask(__name__)


controller = SimplePIController(0.1, 0.002)

model = None
model_name = "model.h5"


@app.route("/predict", methods=['POST'])
def hello():
  # return drive.predict(request.values["data"], None)

  # model = load_model(model_name)
  data = request.json
  speed = data["speed"]
  imgString = data["image"]
  image = Image.open(BytesIO(base64.b64decode(imgString)))
  image_array = np.asarray(image)

  steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

  throttle = controller.update(float(speed))
  return json.dumps((steering_angle, throttle))
  # return json.dumps(drive.predict(json.dumps(request.json), None))


if __name__ == "__main__":
  model = load_model(model_name)
  model._make_predict_function()
  # drive = driver.UdaDrive()
  app.run(host='localhost', port=1234)
