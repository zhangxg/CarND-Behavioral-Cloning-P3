import json

import tornado.ioloop
import tornado.web

from seldoned.UdaDrive import UdaDrive

driver = None


class MainHandler(tornado.web.RequestHandler):
  def post(self, *args, **kwargs):
    # post with request
    data = tornado.escape.json_decode(self.request.body_arguments['data'][0])
    # post with postman
    # data = tornado.escape.json_decode(self.request.body)
    steering_angle, throttle = driver.predict(data, None)
    self.write(json.dumps({"steering_angle": steering_angle, "throttle": throttle}))


def make_app():
  return tornado.web.Application([
    (r"/predict", MainHandler),
  ])


if __name__ == "__main__":
  driver = UdaDrive()
  app = make_app()
  app.listen(8888)
  tornado.ioloop.IOLoop.current().start()
