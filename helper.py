import numpy as np
import cv2

class Area:
	def __init__(self, x, y, w, h):
		self.x = x
		self.y = y
		self.w = w
		self.h = h

	def getPosition1(self):
		return (self.x, self.y)
	def getPosition2(self):
		return (self.x+self.w, self.y+self.h)
	def getWidth(self):
		return self.w
	def getHeight(self):
		return self.h
	def getSize(self):
		return self.w * self.h
	def draw(self, img, color=(0,255,0), width=1):
		cv2.rectangle(img, self.getPosition1(), self.getPosition2(), color, width)