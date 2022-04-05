# Importing necessary Libraries
import mediapipe as mp 
import numpy as np
import sys
import torch, argparse, os
from model import ASLMobilenet, ASLResnet
import cv2


# Defining arguments for ease of access, Model type and Model path to be used for inferencing
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=['resnet34', 'mobilenet_v2'],help = "Desired type of model: Resnet34 or MobilenetV2")
parser.add_argument("--model_path", type=str, help = "Path of model file relative to current directory")

# Label classes
classes = ['A','B','C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']


letter = None

# main begins here
if __name__ == "__main__":
    hand = None
    mphands = mp.solutions.hands 
    hands = mphands.Hands() #Instance of mediapipe hands model
    cap = cv2.VideoCapture(0) #begins webcam capturing
    args = parser.parse_args()
    model_path = args.model_path
    if not os.path.exists(model_path):
        print('MODEL = ', model_path)
        print('The specified model does not exist')
        sys.exit(0)
    MODEL = None
    # Loads the model
    if args.model == 'resnet34':
        MODEL = ASLResnet()
    else:
        MODEL = ASLMobilenet()
    MODEL.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    MODEL.eval() #Sets the model to evaluation mode
    while True: #run this while block for every frame from the webcame
        _, frame = cap.read()
        h,w,c = frame.shape
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb) #process the frame with mediapipe hands
        hand_landmarks = result.multi_hand_landmarks
        # finds the position of the hand in the frame, and crops it out for prediction by CNN
        X1 = None
        Y1 = None
        if hand_landmarks:
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
                x_min -= 90
                y_min -=90
                x_max +=90
                y_max +=90
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                hand = frame[y_min:y_max, x_min:x_max]
                X1 = x_max
                Y1 = y_min
                try:
                    hand = cv2.resize(hand, (200,200))
                except:
                    print("")
        
        
        if cv2.waitKey(1) & 0xFF == ord(' '): # if space bar is pressed
            img = torch.from_numpy(hand).permute(2,0,1).unsqueeze(0).float()
            s = MODEL(img).detach() #predict class of the image
            _, preds  = torch.max(s, dim=1)
            print(classes[preds[0].item()]) #print prediction
            letter = classes[preds[0].item()]
          
        cv2.putText(frame, letter, (X1, Y1), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow('webcam', frame) #display the webcam frame
        
        
        
    
    