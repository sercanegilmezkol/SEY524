import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os
import numpy as np

def get_available_devices():
	devices = []
	for i in range(10):
		cap = cv2.VideoCapture(i)
		if cap.isOpened():
			devices.append(i)
			cap.release()
	return devices

def capture_image(device_id):
	CAMERA_WIDTH = 1920
	CAMERA_HEIGHT = 1080
	#
	cap = cv2.VideoCapture(device_id)
	# Set resolution
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
	#
	ret, frame = cap.read()
	cap.release()
	if not ret:
		print("Failed to capture image from device",device_id)
		return None
	return frame

# Get available video capture devices
devices = get_available_devices()

# Check if any devices are available
if not devices:
	print("No video capture devices found.")
	exit()

# Print the available devices
print("Available devices:")
for device in devices:
	print("Device", device)
    
# Defining cameras
LR=0 #LR_Counter
while True:
	# Ask the user to select a device
	if LR==0 :
		selected_device = int(input("Enter the left camera device number: "))
	else:
		selected_device = int(input("Enter the right camera device number: "))

	# Check if the selected device is valid
	if selected_device not in devices:
		print("Invalid device number.")
		exit()

	# Capture an image from the selected device
	image = capture_image(selected_device)

	# Check if image capture was successful
	if image is not None:
		# Display the captured image		
		#cv2.namedWindow("3D movie",cv2.WINDOW_NORMAL)
		#cv2.resizeWindow("3D movie",960,540)
		plt.rcParams["figure.figsize"] = [9.60, 5.40]
		#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB,aspect='auto'))
		plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		plt.title("Captured Image")
		plt.axis("off")
		plt.show()

	# Ask the user for input to capture the images
	if LR==0 :
		response = input("Is left camera OK? Press 'Y' or 'n' and press Enter.")
		if response == 'Y' or response == 'y':
			device_Number_L = selected_device
			LR=1
	else:
		response = input("Is right camera OK? Press 'Y' or 'n' and press Enter.")
		if response == 'Y' or response == 'y':
			device_Number_R = selected_device
			break



#
# Vize kapsamındaki çalışmalarda bundan sonra sol/sağ kameraların ID'leri elle girilmiştir. 
#
device_Number_L = 0
device_Number_R = 2  
# Set the folder paths for saving the images
left_folder = './data/stereoL'
right_folder = './data/stereoR'
# Create the folders if they don't exist
os.makedirs(left_folder, exist_ok=True)  
os.makedirs(right_folder, exist_ok=True)

# Initialize the camera capture objects
left_camera = cv2.VideoCapture(device_Number_L)
right_camera = cv2.VideoCapture(device_Number_R)
# Set resolution
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
left_camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
left_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
right_camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
right_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Infinite loop to capture images
while True:
	# Ask the user for input to capture the images
	response = input("Do you want to take a shot? (Enter a number and press Enter. Press 'q' to quit): ")

	if response == 'q':
		break

	# Read frames from both cameras
	ret_left, frame_left = left_camera.read()
	ret_right, frame_right = right_camera.read()

	# Check if frames are captured successfully
	if not ret_left or not ret_right:
		print("Failed to capture frames from both cameras.")
		break

	# Save the frames to separate folders
	left_image_path = os.path.join(left_folder, "img" + response + ".jpg")
	right_image_path = os.path.join(right_folder, "img" + response + ".jpg")

	cv2.imwrite(left_image_path, frame_left)
	cv2.imwrite(right_image_path, frame_right)

	print("Images captured and saved successfully.")

	# Display the left and right images
	fig, ax = plt.subplots(1, 2)
	ax[0].imshow(cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB))
	ax[0].set_title("Left Image")
	ax[0].axis("off")
	ax[1].imshow(cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB))
	ax[1].set_title("Right Image")
	ax[1].axis("off")
	plt.show()
	plt.close(fig)
  
# Release the camera capture objects and close windows
left_camera.release()
right_camera.release()
cv2.destroyAllWindows()




# Autonomous Mobile Robots II
# Perception
# Stereo Camera Calibration
# OpenCV example for camera calibration

import glob
from tqdm import tqdm

# Set the path to the images captured by the left and right cameras
pathL = "./data/stereoL/"
pathR = "./data/stereoR/"

# Defining the dimensions of checkerboard
CHECKERBOARD = (6, 9)

# In this case the maximum number of iterations is set to 30 and epsilon = 0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []

# Creating vector to store vectors of 2D points for each checkerboard image
img_ptsL = []
img_ptsR = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1, 2)
print(objp)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
for i in tqdm(range(1,32)):
	img_L = cv2.imread(pathL+"img%d.jpg"%i)
	img_R = cv2.imread(pathR+"img%d.jpg"%i)
	gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
	gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)
	output_L = img_L.copy()
	output_R = img_R.copy()

	# Find the chess board corners
	# If desired number of corners are found in the image then ret = true
	ret_L, corners_L = cv2.findChessboardCorners(output_L, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

	ret_R, corners_R = cv2.findChessboardCorners(output_R, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
	"""
	If desired number of corner are detected,
	we refine the pixel coordinates and display
	them on the images of checker board
	"""
	
	if (ret_L == True) and (ret_R == True):
		objpoints.append(objp)
		# refining pixel coordinates for given 2d points.
		corners_L2 = cv2.cornerSubPix(gray_L, corners_L, (11,11), (-1, -1), criteria)  
		corners_R2 = cv2.cornerSubPix(gray_R, corners_R, (11,11), (-1, -1), criteria)

		img_ptsL.append(corners_L2)
		img_ptsR.append(corners_R2)  

		# Draw and display the corners
		img_chess_L = cv2.drawChessboardCorners(output_L, CHECKERBOARD, corners_L2, ret_L)
		img_chess_R = cv2.drawChessboardCorners(output_R, CHECKERBOARD, corners_R2, ret_R)

		cv2.namedWindow('Corners Left',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('Corners Left',960,540)  
		cv2.imshow('Corners Left', img_chess_L)
		cv2.namedWindow('Corners Right',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('Corners Right',960,540)  
		cv2.imshow('Corners Right', img_chess_R)
		
		cv2.waitKey(0)

cv2.destroyAllWindows() 
  
  

"""
Performing camera calibration by
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the
detected corners (imgpoints)
"""
# Left Camera Calibration
ret_L, mtx_L, dist_L, rvecs_L, tvecs_L = cv2.calibrateCamera(objpoints, img_ptsL, gray_L.shape[::-1], None, None)
# Undistortion
h_L, w_L = gray_L.shape[:2]
newcameramtx_L, roi_L = cv2.getOptimalNewCameraMatrix(mtx_L, dist_L, (w_L,h_L), 1, (w_L,h_L))

# Right Camera Calibration
ret_R, mtx_R, dist_R, rvecs_R, tvecs_R = cv2.calibrateCamera(objpoints, img_ptsR, gray_R.shape[::-1], None, None)
# Undistortion
h_R, w_R = gray_R.shape[:2]
newcameramtx_R, roi_R = cv2.getOptimalNewCameraMatrix(mtx_R, dist_R, (w_R,h_R), 1, (w_R,h_R))

# Step 2: Performing stereo calibration with fixed intrinsic parameters
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same

criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(objpoints, img_ptsL, img_ptsR, newcameramtx_L, dist_L, newcameramtx_R, dist_R, gray_L.shape[::-1], criteria_stereo, flags)

# Step 3: Stereo Rectification
rectify_scale= 1
rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR, gray_L.shape[::-1], Rot, Trns, rectify_scale,(0,0))

# Step 4: Compute the mapping required to obtain the undistorted rectified stereo image pair
Left_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l, gray_L.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r, gray_R.shape[::-1], cv2.CV_16SC2)
print("Saving parameters ......")
cv_file = cv2.FileStorage("improved_params2.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
cv_file.release()