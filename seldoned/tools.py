class SimplePIController:
  def __init__(self, Kp, Ki):
    self.Kp = Kp
    self.Ki = Ki
    self.set_point = 0.
    self.error = 0.
    self.integral = 0.

  def set_desired(self, desired):
    self.set_point = desired

  def update(self, measurement):
    self.error = self.set_point - measurement

    self.integral += self.error

    return self.Kp * self.error + self.Ki * self.integral