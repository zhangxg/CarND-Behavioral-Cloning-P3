import base64
from io import BytesIO

import numpy as np
from PIL import Image
from keras.models import load_model

from seldoned.tools import SimplePIController

model_name = "model.h5"


class UdaDrive(object):

  def __init__(self):
    self.uda_model = load_model(model_name)
    self.controller = SimplePIController(0.1, 0.002)
    self.controller.set_desired(10)

  def predict(self, data, feature_names):
    speed = data["speed"]
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)

    steering_angle = float(self.uda_model.predict(image_array[None, :, :, :], batch_size=1))

    throttle = self.controller.update(float(speed))

    print(steering_angle, throttle)
    return steering_angle, throttle


