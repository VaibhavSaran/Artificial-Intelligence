# 1) Importing Necessary Libraries
import numpy as np
import pandas as pd
import os
import time
import cv2
from sklearn.neighbors import KNeighborsClassifier as KNN

# 2) Loading the Haar Filter
face_cascade = cv2.CascadeClassifier("filter/haarcascade_frontalface_default.xml")
skip=0 # Consider every 3rd frame
logins ={}
face_data = []
label=[]
face_section = []
class_id = 0 # Labels for the given file 
names = {} # map id and name

# 3) Data Preparation
for file in os.listdir(".\saved files"):
    if file.endswith('.npy'):
        names[class_id] = file[:-4] # So that we dont end up taking the extentions as well
        data_item = np.load("./saved files/"+file)
        face_data.append(data_item)
        # Create labels for the class
        target = class_id*np.ones((data_item.shape[0],))
        class_id+=1
        label.append(target)

# 4) Creating Datasets
face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(label,axis=0)#.reshape((-1,1))

# 5) Model Building
while True:
    capture = cv2.VideoCapture(0)# for Linux and mac based OS try -1,-2 if 0 throws error
    check,frame = capture.read()
    if check == False:
        continue
    # Converting to Grey Frame for easier computation
    grey_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Detecting Face using Haar Cascade
    faces = face_cascade.detectMultiScale(grey_frame,1.3,5)

    for face in faces:
        x,y,w,h = face
        # Get the region of interest i.e. the face
        offset = 10 # Padding in pixels
        face_section = grey_frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        # Applying KNN Classifier
        model = KNN(n_neighbors=5)
        model.fit(face_dataset,face_labels)
        # Performing Predictions
        prediction = model.predict(face_section.flatten().reshape(1,-1))

        # Creating Attendance Records
        student_name = []
        student_login_time = []
        student_name.append(names[int(prediction)])
        student_login_time.append(time.ctime(time.time()))

        logins[student_name[0]] = student_login_time[0]

        print(logins)

        # Display on the screen the name and class id/label
        cv2.putText(grey_frame,names[int(prediction)],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        cv2.rectangle(grey_frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.imshow("Faces",grey_frame)
    # Cleaning up OpenCV Windows
    key_pressed = cv2.waitKey(1)
    if key_pressed==ord('q'):
        break

# Saving the data as a Data Frame
df = pd.DataFrame(pd.Series(logins),columns=["Time"])
print(df.columns)
df.to_csv(r"saved files\attendance.csv")
capture.release()
cv2.destroyAllWindows()