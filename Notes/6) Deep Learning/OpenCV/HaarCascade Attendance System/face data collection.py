# 1) Importing Necessary Modules
import cv2
import numpy as np

# 2) Loading files and creating variables
face_cascade = cv2.CascadeClassifier("filter/haarcascade_frontalface_default.xml")
skip = 0
face_data = []
face_section = []

# 3) Collecting Facial Data
while(True):
    # Connecting the webcam and reading the video input
    capture = cv2.VideoCapture(0) # For linux and mac based system if 0 doesn't work try -1,-2
    check,frame = capture.read()

    if check==False: # If the video capture fails try again
        continue

    # Converting color frame to grey frame for easier computation
    grey_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Finding faces using the cascade
    faces = face_cascade.detectMultiScale(frame,1.3,5)

    # Sorting the face area based on largest bounding box area
    faces = sorted(faces,key = lambda f : f[2]*f[3], reverse = True)

    for face in faces:
        
        # Getting coordinates for the detected face
        x,y,w,h = face
        # Drawing rectangle on the frame to highlight detected face
        cv2.rectangle(grey_frame,(x,y),(x+w,y+h),(0,255,0),2)

        # Extracting Face
        offset = 10 # Padding in pixels
        face_section = grey_frame[y-offset:y+h+offset, x-offset:x+w+offset]
        # resizing to a common shape
        face_section = cv2.resize(face_section,(100,100))
        
        # Checking for the 3rd frame
        skip+=1
        if skip%3==0:
            face_data.append(face_section)
            print("Length of face_data: ",len(face_data))
    
    # Displaying the image and the face section in consideration
    cv2.imshow("Grey Frame",grey_frame)
    # cv2.imshow("Face Section",face_section)

    # Cleaning up OpenCV Code
    key_pressed = cv2.waitKey(1)
    if key_pressed==ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

# Converting the face array into numpy array and saving it
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))# No of rows = No of faces
print("Shape of Face Data: ",len(face_data),end="\n\n")

# Saving the data as .npy file as in one file there is only one array of faces of one person
np.save(".\saved files\Vaibhav",face_data)