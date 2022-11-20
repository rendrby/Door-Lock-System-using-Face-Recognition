print("[INFO] Initializing...")
from imutils import paths
import face_recognition
import pickle
import cv2
import os

print("[INFO] Start processing faces...")

# Get paths from dataste folder
imagePaths = list(paths.list_images("/home/pi/face_recog/dataset"))

# Initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# Loop over the training for all pictures
for (i, imagePath) in enumerate(imagePaths):
    # Extract the person name from the image path
	print("[INFO] Processing image {}/{}".format(i + 1,len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# Load the input image and convert it from BGR (OpenCV ordering) to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# Detect the (x, y)-coordinates of the bounding boxes corresponding to each face in the input image
	face_loc = face_recognition.face_locations(rgb,model="hog")

	# Encode the detected face
	face_encoding = face_recognition.face_encodings(rgb,face_loc)

	# Loop over the encodings + names
	for encoding in face_encoding:
		# Add each encoding + name to our set of known names and encodings
		knownEncodings.append(encoding)
		knownNames.append(name)

# Save the facial encodings + names to a file
print("[INFO] Serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("/home/pi/face_recog/encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()