import json

from flask import Flask
from flask import request

from seldoned.UdaDrive import UdaDrive

app = Flask(__name__)

driver = None


@app.route("/predict", methods=['POST'])
def predict():
  steering_angle, throttle = driver.predict(request.json, None)
  return json.dumps((steering_angle, throttle))


if __name__ == "__main__":
  driver = UdaDrive()
  app.run(host='localhost', port=1234)


# has error:
# ValueError: Tensor Tensor("dense_30/BiasAdd:0", shape=(?, 1), dtype=float32) is not an element of this graph.
# see: https://github.com/keras-team/keras/issues/2397#issuecomment-212287164
# fix see: https://github.com/keras-team/keras/issues/2397#issuecomment-306687500
