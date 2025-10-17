import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load input images
background_path = "plain_billboard.jpg"
overlay_path = "surfexcel.jpg"

background_img = cv2.imread(background_path)
bg_copy = background_img.copy()
overlay_img = cv2.imread(overlay_path)

# Resize overlay to match background dimensions (optional)
overlay_img = cv2.resize(overlay_img, (background_img.shape[1], background_img.shape[0]))

#Store selected coordinates
selected_points = []

#Mouse callback function for selecting 4 points on the background
def get_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append([x, y])
        print(f"Selected point: ({x}, {y})")
        
        # Visual feedback on selection
        cv2.circle(bg_copy, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Target Area", bg_copy)

#Display the background image for point selection
cv2.imshow("Select Target Area", bg_copy)
cv2.setMouseCallback("Select Target Area", get_points)

#Wait until user selects 4 points
while len(selected_points) < 4:
    cv2.waitKey(1)

cv2.destroyAllWindows()

#Prepare corresponding point sets
pts_background = np.array(selected_points, dtype=np.float32)
pts_overlay = np.array([
    [0, 0],
    [overlay_img.shape[1], 0],
    [overlay_img.shape[1], overlay_img.shape[0]],
    [0, overlay_img.shape[0]]
], dtype=np.float32)

#Compute homography matrix
H, _ = cv2.findHomography(pts_overlay, pts_background)

# Warp the overlay image onto the background 
warped_overlay = cv2.warpPerspective(overlay_img, H, (background_img.shape[1], background_img.shape[0]))

#Create a mask for blending region
mask = np.zeros_like(background_img, dtype=np.uint8)
cv2.fillConvexPoly(mask, pts_background.astype(int), (255, 255, 255))

# Blend both images together
final_result = cv2.addWeighted(background_img, 0.6, warped_overlay, 1, -1.5)

#Display the final output
plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Warped and Blended Overlay")
plt.show()

