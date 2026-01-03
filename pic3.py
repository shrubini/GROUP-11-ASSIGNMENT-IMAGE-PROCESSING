# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 15:15:27 2026

@author: W11
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD IMAGE
path = r"C:\Users\W11\Downloads\pls.jpg"
img = cv2.imread(path)

if img is None:
    print("Error: Could not find image. Check your path!")
else:
    # 2. SEGMENTATION: Separate colorful pills from white background
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use Otsu's thresholding to find the pills (darker) against background (lighter)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up small noise and fill internal holes in pills
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 3. WATERSHED: Separating touching pills
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground (centers of pills)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # Adjust 0.4 to a higher value (e.g. 0.6) if pills are still sticking together
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)

    # Find unknown region (the boundaries)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply Watershed
    markers = cv2.watershed(img, markers)

    # 4. FINAL RESULTS & ANALYSIS
    result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pill_count = 0
    
    for label in np.unique(markers):
        if label <= 1: continue # Skip background and unknown
        
        mask = np.uint8(markers == label)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(cnts) > 0:
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 100: # Ignore tiny noise
                pill_count += 1
                
                # Shape regularity (Circularity)
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter**2))
                else:
                    circularity = 0

                # Highlight: Green for circles (>0.8), Red for capsules/others
                color = (0, 255, 0) if circularity > 0.8 else (255, 0, 0)
                cv2.drawContours(result, [cnt], -1, color, 2)
                
                # Center label for counting
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(result, f"{pill_count}", (cX-10, cY), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 5. PRESENTATION
    plt.figure(figsize=(18, 10))
    plt.subplot(2, 2, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title('1. Original Picture')
    plt.subplot(2, 2, 2); plt.imshow(opening, cmap='gray'); plt.title('2. Binary Mask (Otsu)')
    plt.subplot(2, 2, 3); plt.imshow(dist_transform, cmap='jet'); plt.title('3. Distance Map (For Separation)')
    plt.subplot(2, 2, 4); plt.imshow(result); plt.title(f'Detected Objects | Total: {pill_count}')
    plt.tight_layout()
    plt.show()