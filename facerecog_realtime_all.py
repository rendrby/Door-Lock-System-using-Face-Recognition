print("[INFO] Initializing...")
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import face_recognition
import imutils
import pickle
import time
import cv2
import os
import smtplib
import RPi.GPIO as GPIO
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

LOCK=19
GPIO.setmode(GPIO.BCM)
GPIO.setup(LOCK,GPIO.OUT)
GPIO.output(LOCK,False)
print("[INFO] Waiting for sensor to settle...")
time.sleep(0.2)

# Initialize 'currentname' to trigger only when a new person is identified
currentname = "Unknown"
name = "Unknown"

# Load the encoded faces from encodings.pickle file created from training.py
encodingsP = "/home/pi/face_recog/encodings.pickle"
data = pickle.loads(open(encodingsP, "rb").read())
print(data)

# Initialize the mail addresses and password
# Changed for privacy reasons
sender_address = 'a@gmail.com'
sender_pass = 'abcd'
receiver_address = 'b@gmail.com'
# Setup the MIME
message = MIMEMultipart()
message['From'] = sender_address
message['To'] = receiver_address
message['Subject'] = 'WARNING! Unknown face detected.'
mail_content = '''Hello,
We have detected an unknown face from the webcam.
In this mail we are sending an attachment of the unknown face.

Thank You.
'''
message.attach(MIMEText(mail_content, 'plain'))

# Initialize camera
frame_size = (480, 320)
camera = VideoStream(src=0,usePiCamera=True,resolution=frame_size).start()
time.sleep(2.0)
fps = FPS().start()

# Loop over frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to 500px (to speedup processing)
    frame = camera.read()
    frame = imutils.resize(frame, width=300)
    # Detect the fce boxes
    boxes = face_recognition.face_locations(frame)
    # compute the facial embeddings for each face bounding box
    encoding = face_recognition.face_encodings(frame, boxes)
    names = []

    if encoding:
        matches = face_recognition.compare_faces(data["encodings"],encoding[0],tolerance=0.45)
        name = "Unknown" #if face is not recognized, then print Unknown

        face_distances = face_recognition.face_distance(data["encodings"], encoding[0])
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = data["names"][best_match_index]

        names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image - color is in BGR
            cv2.rectangle(frame, (left, top), (right, bottom),(0, 200, 200), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_DUPLEX,.6, (0, 200, 200), 1)
    else:
        name = "Empty"

    frame1 = imutils.resize(frame, width=640)
    # display the image to our screen
    cv2.imshow("Facial Recognition is Running, Esc to Exit", frame1)
    key = cv2.waitKey(1) & 0xFF

    #If someone in your dataset is identified, print their name on the screen
    if currentname != name:
        currentname = name
        # print(currentname)
        if currentname.__eq__("Empty"):
            continue
        elif currentname.__eq__("Unknown"):
            print("[INFO] Access denied!")
            # Code to send email
            cv2.imwrite("Unknown.jpg",frame1)
            with open('Unknown.jpg', 'rb') as fp:
                img = MIMEImage(fp.read())
                img.add_header('Content-Disposition', 'attachment', filename="UnknownFace.jpg")
                message.attach(img)
            
            session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
            session.starttls() #enable security
            session.login(sender_address, sender_pass) #login with mail_id and password
            text = message.as_string()
            session.sendmail(sender_address, receiver_address, text)
            session.quit()
            print('[INFO] Mail Sent!')
            os.remove("Unknown.jpg")
        else:
            print("[INFO] Access granted to",currentname)
            # Code to lock
            GPIO.output(LOCK,True)
            time.sleep(5)
            GPIO.output(LOCK,False)            

        
    
    # quit when 'esc' key is pressed
    if key == 27:
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] Elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approximate FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
camera.stop()
