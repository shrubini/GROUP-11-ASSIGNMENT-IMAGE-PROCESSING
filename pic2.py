# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 13:49:31 2026

@author: W11
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. LOAD IMAGE
path = r"C:\Users\W11\Downloads\pic2.jpg"
img = cv2.imread(path)

if img is None:
    print("Error: Could not find image. Check your path!")
else:
    # 2. SEGMENTATION: Removing background via Color Filtering
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    background_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    pill_mask = cv2.bitwise_not(background_mask)
    
    kernel = np.ones((3,3), np.uint8)
    segmented_mask = cv2.morphologyEx(pill_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. HIGHLIGHTING & SHAPE ANALYSIS
    dist_map = cv2.distanceTransform(segmented_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_map, 0.5 * dist_map.max(), 255, 0)
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(cv2.dilate(segmented_mask, kernel, iterations=3), sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)

    # 4. FINAL RESULTS & REGULARITY CHECK
    result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pill_count = 0
    
    for label in np.unique(markers):
        if label <= 1: continue 
        mask = np.uint8(markers == label)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(cnts) > 0:
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 250: 
                pill_count += 1
                
                # --- SHAPE REGULARITY CHECK ---
                # Circularity = 4 * pi * Area / (Perimeter^2)
                # 1.0 is a perfect circle, lower is more irregular
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * (area / (perimeter**2))
                
                # Highlight: Green for Regular (Circles), Red for Irregular (Capsules)
                color = (0, 255, 0) if circularity > 0.8 else (255, 0, 0)
                cv2.drawContours(result, [cnt], -1, color, 2)
                
                # Labeling
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    cv2.putText(result, f"{pill_count}", (cX-10, cY), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(result, f"R:{circularity:.2f}", (cX-15, cY+12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

    # 5. PRESENTATION
    plt.figure(figsize=(18, 10))
    plt.subplot(2, 2, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title('1. Original Picture')
    plt.subplot(2, 2, 2); plt.imshow(segmented_mask, cmap='gray'); plt.title('2. Segmented Mask (Background Removed)')
    plt.subplot(2, 2, 3); plt.imshow(dist_map, cmap='jet'); plt.title('3. Distance Map (Seed Points)')
    plt.subplot(2, 2, 4); plt.imshow(result); plt.title(f'4. Regularity Check (Green > 0.8) | Total: {pill_count}')
    plt.tight_layout()
    plt.show()