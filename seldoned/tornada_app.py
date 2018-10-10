import json

import tornado.ioloop
import tornado.web

from seldoned.UdaDrive import UdaDrive

driver = None


class MainHandler(tornado.web.RequestHandler):
  def post(self, *args, **kwargs):
    # # post with request
    data = tornado.escape.json_decode(self.request.body_arguments['data'][0])
    # # post with postman
    # data = tornado.escape.json_decode(self.request.body)
    steering_angle, throttle = driver.predict(data)
    self.write(json.dumps({"steering_angle": steering_angle, "throttle": throttle}))


def make_app():
  return tornado.web.Application([
    (r"/predict", MainHandler),
  ])


if __name__ == "__main__":
  model_name = "../model_track1_2018-10-10_10:31:46.h5"
  # model_name = "../model_track2_2018-10-10_11:18:49.h5"
  # model_name = "../model_model_track1_with_counter_clock_2018-10-10_12:16:01.h5.h5"
  print("using model {}".format(model_name))
  driver = UdaDrive(model_name)
  app = make_app()
  app.listen(7654)
  tornado.ioloop.IOLoop.current().start()
