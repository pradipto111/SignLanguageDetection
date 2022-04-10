#Import the necessary libraries
from logging import NullHandler
import mediapipe as mp 
import numpy as np
import sys
import torch, argparse, os
from model import SignLanguageModel
import cv2
from utils import *

#Definition of arguments for ease of access
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default=1, help = "Path of the model file relative to current directory")

#Label Classes
classes = ['A','B','C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

letter = None

# Main begins here
if __name__ == "__main__":
    cap = cv2.VideoCapture(0) #Start capturing the webcame
    hand = None
    mphands = mp.solutions.hands #Instantiate the Mediapipe hands model
    hands = mphands.Hands()
    args = parser.parse_args()
    model_path = args.model_path
    if not os.path.exists(model_path):
        print('MODEL = ', model_path)
        print('The specified model does not exist')
        sys.exit(0)
    MODEL = SignLanguageModel() #Instantiate the Mediapipe model
    MODEL.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) #Loading the model path
    MODEL.eval() #Set model to evaluation mode
    while True: #run this block for every frame from the webcame
        _, frame = cap.read()
        h,w,c = frame.shape
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb) #process the frame with mediapipe hands pipelnine
        hand_landmarks = result.multi_hand_landmarks
        X1 = None
        Y1 = None
        if hand_landmarks: #locate the hand in the frame and crop it out for further processing my mediapipe
            for handLMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                x_min -= 50
                y_min -=50
                x_max +=50
                y_max +=50
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                hand = frame[y_min:y_max, x_min:x_max]
                X1 = x_max
                Y1 = y_min
                try:
                    hand = cv2.resize(hand, (200,200))
                except:
                    None

        if cv2.waitKey(1) & 0xFF == ord(' '): #calculate predictions if space bar key is pressed
            lms = generate_landmarks(hand, hands).unsqueeze(0) #generate landmarks with respect to the cropped hand image
            s = MODEL(lms) #predict
            _, preds  = torch.max(s, dim=1)
            print(classes[preds[0].item()]) #print the prediction in the terminal/command prompt window
            letter = classes[preds[0].item()]
        
        cv2.putText(frame, letter, (X1, Y1), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow('webcam', frame) #display the webcam frame on the screen
    
    