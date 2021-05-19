# import libraries
import os
from cv2 import cv2 
import numpy as np
import face_recognition
from datetime import datetime

# define variables
path = 'Images_Attendance'
images = []
classNames = []

#list all image names
mylist = os.listdir(path)
print(mylist)

#loop all the image and import/load it
for cls in mylist:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    #get name of the files without extenstions
    classNames.append(os.path.splitext(cls)[0])

print(classNames)


# define a function for encoding process
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# start encoding process
encodeListKnownFace = findEncodings(images)
print("Encoding has Completed")

# start camera for capturing video
cap = cv2.VideoCapture(0)

pTime = 0

while True:
    success, img = cap.read()
    # resize image for proccessing speed
    imgSmall =  cv2.resize(img, (0,0), None, 0.25, 0.25)
    # convert to RGB
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    # find face location
    faceCurrentFrame = face_recognition.face_locations(imgSmall)
    # find the encoding of the webcam
    encodesCurrentFrame = face_recognition.face_encodings(imgSmall, faceCurrentFrame)

    # loop and compare the faces
    for encodeFace, faceLoc in zip(encodesCurrentFrame, faceCurrentFrame):
        # start comparing face with known faces
        matches = face_recognition.compare_faces(encodeListKnownFace, encodeFace)
        # find the distance
        faceDist = face_recognition.face_distance(encodeListKnownFace, encodeFace)
        # print(faceDist)

        # find lowest element in our distance result
        matchIndex = np.argmin(faceDist)
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            #create a bouncing box
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1)
            markAttendance(name)

    cv2.imshow('frame', img)
    cv2.waitKey(20)

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()