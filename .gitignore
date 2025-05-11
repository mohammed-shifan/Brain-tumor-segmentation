import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# === Step 1: Load the MRI image ===
image_path = 'main1.jpg'  # Replace with your MRI scan image path
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image from '{image_path}'.")
    sys.exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# === Step 2: Create a mask of the full brain (remove skull) ===
_, brain_thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(brain_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max_contour = max(contours, key=cv2.contourArea)

brain_mask = np.zeros_like(gray)
cv2.drawContours(brain_mask, [max_contour], -1, 255, thickness=cv2.FILLED)

# Apply the brain mask to remove background/skull
brain_only = cv2.bitwise_and(gray, gray, mask=brain_mask)

# === Step 3: Detect white patches (tumor-like regions) across full brain ===
_, white_patch_thresh = cv2.threshold(brain_only, 200, 255, cv2.THRESH_BINARY)

# === Step 4: Find contours of white patches ===
patch_contours, _ = cv2.findContours(white_patch_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# === Step 5: Find largest white patch area ===
max_contour_area = 0
for cnt in patch_contours:
    max_contour_area = max(max_contour_area, cv2.contourArea(cnt))

# Set a threshold to ignore small white regions (noise)
min_tumor_area = 500

# === Step 6: Classify and Report ===
if max_contour_area > min_tumor_area:
    print("Tumor Detected! Large white patch found.")
    brain_area = np.sum(brain_mask == 255)
    tumor_percentage = (max_contour_area / brain_area) * 100 if brain_area > 0 else 0
    print(f"Percentage of brain affected by tumor: {tumor_percentage:.2f}%")
else:
    print("Healthy Brain! No significant white patch detected.")

# === Step 7: Draw tumor outline on original image ===
outlined_image = image.copy()
for cnt in patch_contours:
    if cv2.contourArea(cnt) == max_contour_area:
        cv2.drawContours(outlined_image, [cnt], -1, (0, 255, 0), 2)

# === Step 8: Display Results ===
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.title('Original MRI')
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Brain Only')
plt.imshow(brain_only, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Tumor Outlined')
plt.imshow(cv2.cvtColor(outlined_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
