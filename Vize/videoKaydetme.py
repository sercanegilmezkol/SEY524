import numpy as np
import cv2
import sys

Cam_file = "data/stereo.mp4"

CamL_id = 0
CamR_id = 2

CamL= cv2.VideoCapture(CamL_id)
CamR= cv2.VideoCapture(CamR_id)

print("Reading parameters ......")
cv_file = cv2.FileStorage("improved_params2.xml", cv2.FILE_STORAGE_READ)

Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()

CamL.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
CamL.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
CamR.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
CamR.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#writer = cv2.VideoWriter(Cam2_file, fourcc, 5.0, (1920, 1080))
writer = cv2.VideoWriter(Cam_file, fourcc, 8.0, (1920, 1080))
#writer = cv2.VideoWriter(Cam1_file, fourcc, 12.0, (1920, 1080))

### Aşağıdaki kısım disparity denemesi. Vizeye bir manisi yok bu şekliyle burada bırakılmıştır.
#
# Setting parameters for StereoSGBM algorithm
minDisparity = 0;
numDisparities = 64;
blockSize = 8;
disp12MaxDiff = 1;
uniquenessRatio = 10;
speckleWindowSize = 10;
speckleRange = 8;
###############         disp,0,255,cv2.NORM_MINMAX)
# Creating an object of StereoSGBM algorithm
stereo = cv2.StereoSGBM_create(minDisparity = minDisparity,
        numDisparities = numDisparities,
        blockSize = blockSize,
        disp12MaxDiff = disp12MaxDiff,
        uniquenessRatio = uniquenessRatio,
        speckleWindowSize = speckleWindowSize,
        speckleRange = speckleRange
    )

while True:
	retR, imgR= CamR.read()
	retL, imgL= CamL.read()
    
	# Calculating disparith using the StereoSGBM algorithm
	#### disp = stereo.compute(imgL, imgR).astype(np.float32)
	#### disp = cv2.normalize(disp,0,255,cv2.NORM_MINMAX)
 
	# Displaying the disparity map
	# Displaying the disparity map
	#cv2.namedWindow("3D movie",cv2.WINDOW_NORMAL)
	#cv2.resizeWindow("3D movie",960,540)
	#cv2.imshow("3D movie",disp)
	#cv2.waitKey(0)

	if retL and retR:
		imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
		imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)

		Left_nice= cv2.remap(imgL,Left_Stereo_Map_x,Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
		Right_nice= cv2.remap(imgR,Right_Stereo_Map_x,Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
		
		output = Right_nice.copy()
		output[:,:,0] = Right_nice[:,:,0]
		output[:,:,1] = Right_nice[:,:,1]
		output[:,:,2] = Left_nice[:,:,2]

		output = cv2.resize(output,(1920,1080))
		cv2.namedWindow("3D movie",cv2.WINDOW_NORMAL)
		cv2.resizeWindow("3D movie",960,540)
		cv2.imshow("3D movie",output)

		writer.write(output)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break

CamL.release()
CamR.release()
writer.release()
cv2.destroyAllWindows()