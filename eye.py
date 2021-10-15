#!/usr/bin/env python
# coding: utf-8

# In[45]:


import cv2
import numpy as np
import dlib
import os
from math import hypot
import pyglet
import time


# In[46]:


# Load sounds
datSound = r"C:\Users\vijis\Downloads\sound.wav"
datleft = r"C:\Users\vijis\Downloads\left.M4A"
datright = r"C:\Users\vijis\Downloads\right.M4A"

sound = pyglet.media.load(datSound,streaming = False)
left_sound = pyglet.media.load(datleft,streaming = False)
right_sound = pyglet.media.load(datright,streaming = False)


# In[47]:


cap = cv2.VideoCapture(0)
board = np.zeros((300,1400),np.uint8)
board[:] = 255


# In[48]:


datFile =  r"C:\Users\vijis\Downloads\shape_predictor_68_face_landmarks.dat"


# In[49]:


# we used the detector to detect the frontal face
detector = dlib.get_frontal_face_detector()

# it will dectect the facial landwark points 
predictor = dlib.shape_predictor(datFile)

#Keybi
keyboard = np.zeros((600,1000,3),np.uint8)

key_set_1 = {0: "Q" , 1: "W", 2: "E", 3: "R", 4: "T",
             5: "A" , 6: "S", 7: "D", 8: "F", 9: "G",
             10: "Z" ,11: "X",12: "C",13: "V",14: "B"}


# In[50]:


def letter(letter_index,text,letter_light):
    #Keys
    if letter_index == 0:
        x=0
        y=0
    elif letter_index == 1:
        x=200
        y=0
    elif letter_index == 2:
        x=400
        y=0
    elif letter_index == 3:
        x=600
        y=0
    elif letter_index == 4:
        x=800
        y=0
    elif letter_index == 5:
        x=0
        y=200
    elif letter_index == 6:
        x=200
        y=200
    elif letter_index == 7:
        x=400
        y=200
    elif letter_index == 8:
        x=600
        y=200
    elif letter_index == 9:
        x=800
        y=200
    elif letter_index == 10:
        x=0
        y=400
    elif letter_index == 11:
        x=200
        y=400
    elif letter_index == 12:
        x=400
        y=400
    elif letter_index == 13:
        x=600
        y=400
    elif letter_index == 14:
        x=800
        y=400
        
    width = 200
    height = 200
    th = 3
    
    if letter_light is True:
        cv2.rectangle(keyboard, (x+th,y+th), (x+width-th,y+height-th),(255,255,255),-1)
    else:
        cv2.rectangle(keyboard, (x+th,y+th), (x+width-th,y+height-th),(255,0,0),th)

    #Text Settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th= 4
    text_size = cv2.getTextSize(text,font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0],text_size[1]
    text_x = int((width - width_text)/2) + x
    text_y = int((height + height_text)/2) + y

    cv2.putText(keyboard,text,(text_x, text_y),font_letter, font_scale,(255,0,0),font_th)


# In[51]:


def midpoint(p1,p2):
    return int((p1.x + p2.x)/2) , int((p1.y+p2.y)/2) 

font = cv2.FONT_HERSHEY_SIMPLEX


# In[52]:



def get_blinking_ratio(eye_points, facial_landmarks):
    
    left_point = (facial_landmarks.part(eye_points[0]).x,facial_landmarks.part(eye_points[0]).y)
    right_point =(facial_landmarks.part(eye_points[3]).x,facial_landmarks.part(eye_points[3]).y)
    centre_top = midpoint(facial_landmarks.part(eye_points[1]),facial_landmarks.part(eye_points[2]))
    centre_bottom = midpoint(facial_landmarks.part(eye_points[5]),facial_landmarks.part(eye_points[4]))
        
    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    #ver_line = cv2.line(frame, centre_top, centre_bottom, (0, 255, 0), 2)
        
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((centre_top[0] - centre_bottom[0]), (centre_top[1] - centre_bottom[1]))
        
    ratio = hor_line_length/ver_line_length 
    return ratio


def get_gaze_ratio(eye_points, facial_landmarks):
    
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x,facial_landmarks.part(eye_points[0]).y),
                                    (facial_landmarks.part(eye_points[1]).x,facial_landmarks.part(eye_points[1]).y),
                                    (facial_landmarks.part(eye_points[2]).x,facial_landmarks.part(eye_points[2]).y),
                                    (facial_landmarks.part(eye_points[3]).x,facial_landmarks.part(eye_points[3]).y),
                                    (facial_landmarks.part(eye_points[4]).x,facial_landmarks.part(eye_points[4]).y),
                                    (facial_landmarks.part(eye_points[5]).x,facial_landmarks.part(eye_points[5]).y)], np.int32)
        
        
        #cv2.polylines(frame,[left_eye_region],True,(0,0,255),2)
        

    height, width, _ = frame.shape
    mask = np.zeros((height,width), np.uint8)
        
    cv2.polylines(mask,[left_eye_region],True,255,2)
    cv2.fillPoly(mask,[left_eye_region],255)
        
    eye = cv2.bitwise_and(gray,gray, mask=mask)
        
        
        
    min_x =np.min(left_eye_region[: , 0])
    max_x =np.max(left_eye_region[: , 0])
    min_y =np.min(left_eye_region[: , 1])
    max_y =np.max(left_eye_region[: , 1])
        
    gray_eye = eye[min_y : max_y, min_x : max_x]
    _,threshold_eye= cv2.threshold(gray_eye, 70, 255,cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
        
    left_side_threshold = threshold_eye[0: height, 0: int(width/2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
        
    right_side_threshold = threshold_eye[0: height,int(width/2) : width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    
    if left_side_white ==0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white/right_side_white
    
    return gaze_ratio


# In[53]:


#counters

frames = 0
letter_index = 0
blinking_frames =0
text = ""
keyboard_selected = "left"
last_keyboard_selected = "left"
while True:
    _,frame = cap.read()
    frame = cv2.resize(frame , None, fx=0.5,fy=0.5)
    keyboard[:] = (0,0,0)
    frames +=1
    new_frame = np.zeros((500,500,3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    active_letter = key_set_1[letter_index]
    
    faces = detector(gray)
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        
        landmarks = predictor(gray, face)
        
        #Detect Blink
        left_eye_ratio = get_blinking_ratio([36,37,38,39,40,41],landmarks)
        right_eye_ratio = get_blinking_ratio([42,43,44,45,46,47],landmarks)
        
        blinking_ratio = (left_eye_ratio + right_eye_ratio)/2
        
        if blinking_ratio > 5.4:
            cv2.putText(frame, "BLINKING" , (50,150), font, 3 , (255, 0, 0),thickness= 3)
            blinking_frames +=1
            frames -=1
            
            # Typing Letter
            if blinking_frames == 3:
                text += active_letter 
                sound.play()
                time.sleep(1)
        else:
            blinking_frames =0
            
        
            
        #gaze detection
        
        gaze_ratio_left_eye = get_gaze_ratio([36,37,38,39,40,41],landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42,43,44,45,46,47],landmarks)
        
        gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye)/2
        
       
        #if gaze_ratio <= 0.9:
            #keyboard_selected = "right"
            #if keyboard_selected !=last_keyboard_selected:
                #right_sound.play()
                #time.sleep(1)
                #last_keyboard_selected = keyboard_selected
        #else:
            #keyboard_selected = "left"
            #if keyboard_selected !=last_keyboard_selected:
                #left_sound.play()
                #time.sleep(1)
                #last_keyboard_selected = keyboard_selected

        

        cv2.putText(frame, str(gaze_ratio), (50,100), font, 2, (0,0,255), 3)
    
    if frames == 15:
        letter_index +=1
        frames = 0
    if letter_index == 15:
        letter_index = 0
       
    
    
    for i in range(15):
        if i == letter_index:
            light = True
        else:
            light = False
        letter(i,key_set_1[i],light)

    cv2.putText(board, text, (10,100), font, 4, 0,3)
        
    #cv2.imshow("Frame",frame)
    cv2.imshow("Virtual Keyboard",keyboard)
    cv2.imshow("Board",board)


    
    key = cv2.waitKey(1)
    if key == 27:
        break
cap. release
cv2.destroyAllwindows()


# In[ ]:





# In[ ]:




