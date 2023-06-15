# Autonomous Mobile Robots II 
# Perception 
# Stereo Camera 
# OpenCV example for Obstacle Avoidance 

import numpy as np 
import cv2 
import math

# Check for left and right camera IDs 
# These values can change depending on the system 
CamL_id = 2 # Camera ID for left camera 
CamR_id = 0 # Camera ID for right camera 

CamL = cv2.VideoCapture(CamL_id) 
CamR = cv2.VideoCapture(CamR_id) 

CamL.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
CamL.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
CamR.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
CamR.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

Cam_file = "data/Nearest5.mp4"
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter(Cam_file, fourcc, 3.0, (1920, 1080))


# Reading the mapping values for stereo image rectification 
cv_file = cv2.FileStorage("improved_params2.xml", cv2.FILE_STORAGE_READ) 
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat() 
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat() 
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat() 
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat() 
cv_file.release() 

disparity = None 
depth_map = None 

# These parameters can vary according to the setup 
max_depth = 190 # maximum distance the setup can measure (in cm) 
min_depth = 50  # minimum distance the setup can measure (in cm) 

# Reading the stored the StereoBM parameters 
cv_file = cv2.FileStorage("./data/depth_estmation_params_py2.xml", cv2.FILE_STORAGE_READ) 
numDisparities = int(cv_file.getNode("numDisparities").real()) 
blockSize = int(cv_file.getNode("blockSize").real()) 
preFilterType = int(cv_file.getNode("preFilterType").real()) 
preFilterSize = int(cv_file.getNode("preFilterSize").real()) 
preFilterCap = int(cv_file.getNode("preFilterCap").real()) 
textureThreshold = int(cv_file.getNode("textureThreshold").real()) 
uniquenessRatio = int(cv_file.getNode("uniquenessRatio").real()) 
speckleRange = int(cv_file.getNode("speckleRange").real()) 
speckleWindowSize = int(cv_file.getNode("speckleWindowSize").real()) 
disp12MaxDiff = int(cv_file.getNode("disp12MaxDiff").real()) 
minDisparity = int(cv_file.getNode("minDisparity").real()) 
M = cv_file.getNode("M").real() 
cv_file.release() 

# mouse callback function 
def mouse_click(event, x, y, flags, param): 
	global Z 
	if event == cv2.EVENT_LBUTTONDBLCLK: 
		print("Disparity= %.4f" % disparity[y, x]) 
		print("Distance = %.4f cm" % depth_map[y, x])
		print(x)
		print(y) 


cv2.namedWindow('disp', cv2.WINDOW_NORMAL) 
cv2.resizeWindow('disp', 600, 600) 
cv2.setMouseCallback('disp', mouse_click)
cv2.namedWindow("output_canvas",cv2.WINDOW_NORMAL)

# Creating an object of StereoBM algorithm 
stereo = cv2.StereoBM_create() 


threshold1 = 77	  # Adjust the region to scan
threshold2 = 170

def yakin_objeler():
	N = 5	  # Set the number of nearest objects to retrieve
	mask = cv2.inRange(depth_map, threshold1, threshold2)

	contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
	contours = sorted(contours, key=cv2.contourArea, reverse=True) 

	# Consider larger 25 contours at most 
	l_contour = len(contours)
	Nc=25
	if Nc > l_contour:
		Nc = l_contour

	ddd=0
	C_M=[]
	for i in range(Nc):
		x, y, w, h = cv2.boundingRect(contours[i])
		depth_value = depth_map[y, x]
			
		row=[]
		if abs(ddd - depth_value) > 4 and depth_value > 0:
			row.append(depth_value)
			row.append(x)
			row.append(y)
			row.append(w)
			row.append(h)
			C_M.append(row)
			ddd=depth_value

	C_M = sorted(C_M, key=lambda x: x[0])
	lCM = len(C_M)
	
	if N > lCM:
		N = lCM

	for i in range(N):
		x=C_M[i][1]
		y=C_M[i][2]
		w = C_M[i][3]
		h = C_M[i][4]	
		# Draw a rectangle around the object & write its distance
		cv2.rectangle(output_canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.putText(output_canvas, "%.2f cm" % C_M[i][0], (x + w//2, y + h//2), 1, 2, (100, 10, 25), 2, 2)
		#cv2.putText(output_canvas, "%.2f cm" % C_M[i][0], (90*i+125, 90*i+25), 1, 2, (100, 10, 25), 2, 2)
		#cv2.putText(output_canvas, "%d" % x, (90*i+10, 90*i+25), 1, 2, (100, 10, 25), 2, 2)
		#cv2.putText(output_canvas, "%d" % y, (90*i+10, 90*i+60), 1, 2, (100, 10, 25), 2, 2)

while True: 
	retR, imgR = CamR.read() 
	retL, imgL = CamL.read() 
	
	if retL and retR: 
		
		output_canvas = imgL.copy() 
		
		imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY) 
		imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY) 
		
		# Applying stereo image rectification on the left image 
		Left_nice = cv2.remap(imgL_gray, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

		# Applying stereo image rectification on the right image 
		Right_nice = cv2.remap(imgR_gray, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0) 
		
		# Setting the updated parameters before computing disparity map 
		stereo.setNumDisparities(numDisparities) 
		stereo.setBlockSize(blockSize) 
		stereo.setPreFilterType(preFilterType) 
		stereo.setPreFilterSize(preFilterSize) 
		stereo.setPreFilterCap(preFilterCap) 
		stereo.setTextureThreshold(textureThreshold) 
		stereo.setUniquenessRatio(uniquenessRatio) 
		stereo.setSpeckleRange(speckleRange) 
		stereo.setSpeckleWindowSize(speckleWindowSize) 
		stereo.setDisp12MaxDiff(disp12MaxDiff) 
		stereo.setMinDisparity(minDisparity) 
		
		# Calculating disparity using the StereoBM algorithm 
		disparity = stereo.compute(Left_nice, Right_nice) 
		# NOTE: compute returns a 16bit signed single channel image, 
		# CV_16S containing a disparity map scaled by 16. Hence it 
		# is essential to convert it to CV_16S and scale it down 16 times. 
		
		# Converting to float32 
		disparity = disparity.astype(np.float32) 
		
		# Normalizing the disparity map 
		disparity = (disparity / 16.0 - minDisparity) / numDisparities 
		
		depth_map = M / (disparity) # for depth in (cm) 
		
		mask_temp = cv2.inRange(depth_map, min_depth, max_depth) 
		depth_map = cv2.bitwise_and(depth_map, depth_map, mask=mask_temp)
		
		yakin_objeler()
		
		cv2.resizeWindow('output_canvas',960,540)
		cv2.imshow('output_canvas', output_canvas)
		writer.write(output_canvas)

		cv2.resizeWindow("disp", 700, 700) 
		cv2.imshow("disp", disparity) 
		
		cv2.waitKey(16)
		
		if cv2.waitKey(1) == 27: 
			break 
			
	else: 
		CamL = cv2.VideoCapture(CamL_id) 
		CamR = cv2.VideoCapture(CamR_id)
		
		
CamL.release()
CamR.release()
writer.release()
cv2.destroyAllWindows()		
		
		



### Diğer denemeler. Possible future works...

#from functools import cmp_to_key
#def compare(doc):
#	return i1-i2




#    from functools import cmp_to_key
#    def compare(i1,i2):
#      return i1-i2
#    events.sort(key=cmp_to_key(compare))



# Example output:
# Object 1: X=120, Y=80, Distance=50
# Object 2: X=180, Y=220, Distance=55
# Object 3: X=60, Y=150, Distance=60
