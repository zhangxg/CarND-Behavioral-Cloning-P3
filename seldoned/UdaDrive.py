import base64
from io import BytesIO

import numpy as np
from PIL import Image
from keras.models import load_model
import tensorflow as tf

from seldoned.tools import SimplePIController


class UdaDrive(object):

  def __init__(self, model_name):
    self.uda_model = load_model(model_name)
    self.graph = tf.get_default_graph()
    self.controller = SimplePIController(0.1, 0.002)
    self.controller.set_desired(10)

  def predict(self, data):
    speed = data["speed"]
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)

    # fix flask can not predict issue
    # see: https://github.com/keras-team/keras/issues/2397#issuecomment-306687500
    with self.graph.as_default():
      steering_angle = float(self.uda_model.predict(image_array[None, :, :, :], batch_size=1))

    throttle = self.controller.update(float(speed))

    print(steering_angle, throttle)
    return steering_angle, throttle


