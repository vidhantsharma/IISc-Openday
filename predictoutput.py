import cv2
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, model_from_json
from keras.utils import to_categorical
import matplotlib.pyplot as plt

from keras.models import load_model
from tkinter import *
import tkinter as tk
from PIL import ImageGrab, Image

# import rospy
# from one_robot_one_obs.msg import relDist

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

def image_processing():
    image = cv2.imread('./test.jpg')
    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    preprocessed_digits = []
    digits_areas = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        
        # Creating a rectangle around the digit in the original image (for displaying the digits fetched via contours)
        cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
        
        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = thresh[y:y+h, x:x+w]
        
        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (18,18))
        
        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
        # cv2.imshow('processed image',padded_digit)
        im = Image.fromarray(padded_digit,mode = "L")
        im.save('test_processed.jpg')
        # Adding the preprocessed digit to the list of preprocessed digits
        preprocessed_digits.append(padded_digit)
        digits_areas.append(area)
    print("\n\n\n----------------Contoured Image--------------------")
    # plt.imshow(image, cmap="gray")
    return preprocessed_digits, digits_areas
    # plt.show()

def predict_digit(image):
    image.save('test.jpg')
    preprocessed_digits, digits_areas = image_processing()
    # inp = np.array(preprocessed_digits)
    pred_res = []
    max_cnt_idx = np.argmax(digits_areas)
    digit = preprocessed_digits[max_cnt_idx]
    # for digit in preprocessed_digits:
    prediction = model.predict(digit.reshape(1, 28, 28, 1))  
    # print ("\n\n---------------------------------------\n\n")
    # print ("=========PREDICTION============ \n\n")
    # plt.imshow(digit.reshape(28, 28), cmap="gray")
    # plt.show()
    print("\n\nFinal Output: {}".format(np.argmax(prediction)))
    pred_res.append(np.argmax(prediction))
    # print ("\nPrediction (Softmax) from the neural network:\n\n {}".format(prediction)) 
    hard_maxed_prediction = np.zeros(prediction.shape)
    hard_maxed_prediction[0][np.argmax(prediction)] = 1
    # print ("\n\nHard-maxed form of the prediction: \n\n {}".format(hard_maxed_prediction))
    # print ("\n\n---------------------------------------\n\n")
    return pred_res

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=800, height=700, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Recognise", command = self.classify_handwriting) 
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        # self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        # self.pub = rospy.Publisher('target_num',relDist, queue_size=10)
        # self.msg = relDist()
    def clear_all(self):
        self.canvas.delete("all")
        # self.msg.xrel = 0 # random digit
        # self.msg.yrel = 0 # status of vehicle (0 - no move, 1 - move)
        # self.pub.publish(self.msg.data)
    def classify_handwriting(self):
        HWND = self.canvas.winfo_id() # get the handle of the canvas
        x1, y1, x2, y2 = 40, 25, 750, 650
        # print(x1,y1, x2,y2)
        # tk.Canvas.create_rectangle(x1,y1,x2,y2)
        im = ImageGrab.grab((x1+40, y1+40, x2+100, y2+100))
        # im = ImageGrab.grab((x1, y1, x2, y2))
        digits = predict_digit(im)
        digit = digits[0]
        # print(digits[0])
        # self.label.configure(text= str(digit)+', '+ str(int(acc*100))+'%')
        # self.label.configure(text= str(' '.join([str(digit) for digit in digits])))
        self.label.configure(text= str(digit))
        # self.msg.xrel = digit # digit
        # self.msg.yrel = 1 # status of vehicle (0 - no move, 1 - move)
        # # self.pub.publish(self.msg.data)
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')

def task():
    # app.pub.publish(app.msg)
    # print(app.msg)
    app.after(50,task)

if __name__ == "__main__":
    # rospy.init_node('openday - DACAS',anonymous=True)
    app = App()
    app.after(50,task)
    mainloop()
    
    