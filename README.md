# CS Privacy Protection
 
1-Demo.m is only for demonstration purposes:
  i) It creates a snapshot from the webcam. You can select a small region to double preserve.
  ii) The selection is made via mouse-clicking, left click to draw the region, right click to finish drawing. The drawing should define the outer edges of the mask. 
  iii) For this demo, the maximum number of pixels to be doubly concealed (mask region) is    
21000. 

2-main.m is the main function to make the analysis in the manuscript. It encrypts the images of YouTubeFaces and recovers them via UserA.m (Algorithm of Type A) and UserB.m (Algorithm of Type B). 

3-In YouTubeFace Folder, only a small subset of the dataset that is used in experiments is uploaded for demonstration purposes. Please see the experimental setup in the manuscript for more detail. 

4-mainFace.py is to make face recognition analysis on both Type A and Type B recovered images 

5- Dlib's pre-trained networks are used for feature extraction from the images.

6- You need Wavelab850 toolbox: https://statweb.stanford.edu/~wavelab/
