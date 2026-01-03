# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 22:02:01 2026

@author: W11
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. SETUP PATH
path = r"C:\Users\W11\Downloads\pic5.jpg"
img = cv2.imread(path)

if img is None:
    print("ERROR: Could not find image. Check your path!")
else:
    # 2. COLOR SPACE CONVERSION (Better than Grayscale)
    # Convert to HSV to separate Color from Brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv) # We use the 'S' (Saturation) channel
    
    # 3. INTENSE CLEANING
    # Median blur kills the palm lines/skin texture
    s_blurred = cv2.medianBlur(s, 11) 
    
    # Thresholding the saturation channel
    _, thresh = cv2.threshold(s_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. WATERSHED SEGMENTATION
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Distance map is the "heart" of the pill
    dist_map = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_map, 0.4 * dist_map.max(), 255, 0)
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)

    # 5. FINAL RESULT
    result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pill_count = 0
    for label in np.unique(markers):
        if label <= 1: continue
        pill_count += 1
        mask = np.uint8(markers == label)
        # Draw clean contours for a "Master's level" look
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, cnts, -1, (0, 255, 0), 2)
        # Label the number
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            cv2.putText(result, str(pill_count), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # 6. 4-PANEL COMPARISON
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title('1. Original')
    plt.subplot(2, 2, 2); plt.imshow(s, cmap='gray'); plt.title('2. Saturation Channel (Clearer Edges)')
    plt.subplot(2, 2, 3); plt.imshow(dist_map, cmap='jet'); plt.title('3. Distance Map (Seed Points)')
    plt.subplot(2, 2, 4); plt.imshow(result); plt.title(f'4. Final Detection: {pill_count}')
    plt.tight_layout()
    plt.show()
    