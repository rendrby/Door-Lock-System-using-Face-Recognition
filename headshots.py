from imutils.video import VideoStream
import cv2
import os

# Initialize camera
# camera = cv2.VideoCapture(0)
frame_size = (640, 480)
camera = VideoStream(src=0,usePiCamera=True,resolution=frame_size).start()


# Input name for the dataset folder
name = input("Enter your name: ")
if not os.path.exists('/home/pi/face_recog/dataset/' + name):
    os.makedirs('/home/pi/face_recog/dataset/' + name)
    print('Directory /dataset/{} created!'.format(name))
else:
    print('Directory /dataset/{} already exists.'.format(name)) 

img_counter = 0

while True:

    frame = camera.read()
    cv2.imshow("Press space to take a photo; Esc to quit", frame)

    k = cv2.waitKey(1)
    if k == 27:
        # ESC pressed, close program
        print("Esc hit, closing...")
        break
    elif k == 32:
        # SPACE pressed, take picture
        img_name = "/home/pi/face_recog/dataset/"+ name +"/image_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

camera.stop()

cv2.destroyAllWindows()