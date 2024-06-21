import numpy as np

class NoFilter():
  def __init__(self):
    pass

  def reset(self, measurement):    
    return measurement[:2]
  
  def update(self, dt, measurement):  
    return measurement[:2]
