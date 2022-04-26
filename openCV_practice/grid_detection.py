import numpy as np
import matplotlib.pyplot as plt 
import cv2


flat_chess = cv2.imread("chess_board.jpg")
flat_chess = cv2.resize(flat_chess,(350,350))
flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2RGB)
