import os
from datetime import datetime
import cv2
import numpy as np
import face_recognition

def attendance(name):
    # Debugging: Check if the name is being passed correctly
    print(f"Recording attendance for: {name}")
    
    # Absolute path for the Attendance.csv file
    attendance_file = r"C:\Users\hrite\OneDrive\Desktop\face\Attendance.csv"
    
    # Check if the file exists, if not create it with headers
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w', encoding='utf-8') as f:
            f.writelines('Name,Time,Date\n')  # Add headers if the file does not exist
        print("Attendance file created with headers.")

    # Open the file in read mode to check for existing names
    with open(attendance_file, 'r', encoding='utf-8') as f:  # Open in read mode
        myDataList = f.readlines()
        nameList = [line.split(',')[0].strip() for line in myDataList]  # Extract names and remove extra spaces

    # If the name is not in the list, append the name to the file
    if name.strip() not in nameList:  # Ensure no extra spaces before comparison
        with open(attendance_file, 'a', encoding='utf-8') as f:  # Open in append mode
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'{name},{tStr},{dStr}\n')  # Append new entry
        print(f'Attendance recorded for {name}.')
    else:
        print(f'{name} is already in the attendance list.')


# Path to the images folder
path = 'images'
images = []
personNames = []
myList = os.listdir(path)
print(myList)

# Loop through the images in the 'images' folder
for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personNames.append(os.path.splitext(cu_img)[0])
print(personNames)


def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!')

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame for faster processing
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)  # Record attendance if the face is recognized

    # Display the video feed
    cv2.imshow('Webcam', frame)

    # Break the loop if Enter key is pressed
    if cv2.waitKey(1) & 0xFF == 13:  # 13 is the Enter key
        print("Exiting...")
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
