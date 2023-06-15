#Distance: 140 cm | Disparity: 0.51925224
#Distance: 130 cm | Disparity: 0.57393974
#Distance: 120 cm | Disparity: 0.640346
#Distance: 110 cm | Disparity: 0.7103795
#Distance: 100 cm | Disparity: 0.80133927
#Distance: 90 cm | Disparity: 0.90290177
#Distance: 80 cm | Disparity: 0.99553573
#Distance: 70 cm | Disparity: 0.46289062
#Distance: 60 cm | Disparity: 0.99553573
#Distance: 50 cm | Disparity: 0.749163

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

max_dist = 140 
min_dist = 41 
sample_delta = 10 

Value_pairs = []
Value_pairs.append([140, 0.51925224])
Value_pairs.append([130, 0.57393974])
Value_pairs.append([120, 0.640346])
Value_pairs.append([110, 0.7103795])
Value_pairs.append([100, 0.80133927])
Value_pairs.append([90, 0.90290177])
Value_pairs.append([80, 0.99553573])

value_pairs = np.array(Value_pairs)
z = value_pairs[:, 0]
disp = value_pairs[:, 1]
disp_inv = 1 / disp


# Plotting the relation depth and corresponding disparity
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(disp, z, 'o-')
ax1.set(xlabel='Normalized disparity value', ylabel='Depth from camera (cm)', title='Relation between depth \n and corresponding disparity')
ax1.grid()
ax2.plot(disp_inv, z, 'o-')
ax2.set(xlabel='Inverse disparity value (1/disp) ', ylabel='Depth from camera (cm)', title='Relation between depth \n and corresponding inverse disparity')
ax2.grid()
plt.show()

# Solving for M using least square fitting with QR decomposition method
coeff = np.vstack([disp_inv, np.ones(len(disp_inv))]).T
ret, sol = cv2.solve(coeff, z, flags=cv2.DECOMP_QR)
M = sol[0, 0]
C = sol[1, 0]
print("Value of M = ", M)
