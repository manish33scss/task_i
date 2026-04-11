#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 08:36:14 2026

@author: manish
"""

# gamma_correction.py
"""
Automatic gamma correction module.
Uses the mean luminance of the frame to estimate gamma
and applies correction via a lookup table (LUT) — very fast.

Usage (standalone):
    from gamma_correction import GammaCorrector
    gc = GammaCorrector()
    corrected = gc.process(frame)
"""

import cv2
import numpy as np


class GammaCorrector:
    def __init__(self, target_mean: float = 128.0, clip_gamma: tuple = (0.4, 2.5)):
        """
         
            target_mean  : desired mean brightness (0-255). 128 = neutral.
            clip_gamma   : (min, max) gamma clamp to avoid over-correction.
        """
        self.target_mean = target_mean
        self.clip_gamma  = clip_gamma


    def estimate_gamma(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
    
        gamma = mean / self.target_mean
        gamma = np.clip(gamma, self.clip_gamma[0], self.clip_gamma[1])
        return float(gamma)
    
    
    def build_lut(self, gamma):
        lut = np.array([
            ((i / 255.0) ** gamma) * 255
            for i in range(256)
        ], dtype=np.uint8)
        return lut

    def process(self, frame: np.ndarray) -> tuple:
        """
        Apply automatic gamma correction.

        Returns:
            corrected (np.ndarray) : gamma-corrected BGR frame
            gamma     (float)      : gamma value that was applied
        """
        gamma = self.estimate_gamma(frame)
        lut   = self.build_lut(gamma)
        corrected = cv2.LUT(frame, lut)
        return corrected, gamma
    
if __name__ == "__main__":
    gc  = GammaCorrector()
    cap = cv2.VideoCapture("/home/manish/Mee/codes/vayudh_task/video1.avi")
 
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        print("Video file opened successfully!")
     
    while True:
    # Read the first frame to confirm reading
        ret, frame = cap.read()
         
        if not ret:
            break
            # Display the frame using imshow
        gamma_frame, gamma_val = gc.process(frame)

        cv2.imshow("First Frame", gamma_frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Close the window
        #else:
        #    print("Error: Could not read the frame.")
         

