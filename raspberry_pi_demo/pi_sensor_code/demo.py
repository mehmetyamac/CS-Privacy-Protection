from measurements.lpnoiselet_xforms import A_noiselet, At_noiselet
from measurements.dct_xforms import A_dct, At_dct
from npsocket import SocketNumpyArray

from imutils.video import VideoStream
from imutils.video import FPS
from scipy.special import erfinv

from numpy.linalg import norm

import imutils
import time
import cv2
import numpy as np

np.random.seed(1)
sock_sender =SocketNumpyArray()
#sock_sender.initialize_sender('10.42.0.1', 5432)
sock_sender.initialize_sender('192.168.43.132', 5432)

class Parameters:
  def __init__(self, mr, N = 2**16, em_power = 0.15, degradation = 'binary', m1 = 7, m2 = 7):
    self.mratio = mr/10.0
    self.N = N
    self.em_power = em_power
    self.degradation = degradation
    self.m1 = m1
    self.m2 = m2
    
    samples = np.random.rand(2**18)
    # transform from uniform to standard normal distribution using inverse cdf
    samples = np.sqrt(2) * erfinv(2 * samples - 1)
    self.matrix = 0.9 + 0.05*np.reshape(samples, [2**9, 2**9], order = 'F')
    
    self.type = 'rgb'
    self.seed1=8
    self.seed2=9
    self.M=15003 #maximum leghts of the bits we can embed
    self.pr = 0.5
    self.redundant = 0


param = Parameters(6)



detector = cv2.CascadeClassifier('/home/pi/demo/haarcascade_frontalface_default.xml')

vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

fps = FPS().start()
frame = vs.read()
frame = cv2.flip(frame, 0)
#frame = cv2.imread("sample.png")
S1_init = frame.shape[0]
S2_init = frame.shape[1]
sfactor = np.sqrt(param.N/(S1_init * S2_init))
counter = 1
while True:
    input("Press Eter to continue..")
    frame = vs.read()
    if frame is None: continue
    frame = cv2.flip(frame, 0)
    #frame = cv2.imread("sample.png")
    frame = cv2.resize(frame, None, fx = sfactor, fy = sfactor)
    frame = frame[:,0:-1,:]
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(0) & 0xFF

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    boxes = [(x, y, x + w, y + h) for (x, y, w, h) in rects]

    # loop over the recognized faces
    for (x, y, x_w, y_h) in boxes:
        cv2.rectangle(frame, (x, y), (x_w, y_h), (0, 255, 0), 2)

    # If no faces do only compression
    if len(boxes) == 0:
        mask = np.zeros(gray.shape);
        MM = (np.random.rand(frame.shape[0], frame.shape[1]) < param.pr)
        MM = np.array(MM, dtype = np.float64)
        MM = MM*2 -1
        D = np.multiply(mask, MM)
        D2 = np.multiply(D, param.matrix[0:frame.shape[0], 0:frame.shape[1]])
        d = np.reshape(D, D.size, order = 'F')
        inx = np.where(np.reshape(mask, mask.size, order = 'F') == 1)[0]
        watermark_infD = np.reshape(D, D.size, order = 'F')

        # Create watermark coordinates
        w_c = np.zeros((4, 1), dtype = np.uint8)
        w_c = np.unpackbits(w_c, axis = 1)
        w_c = np.array(w_c, dtype = np.float64)
        w_c[w_c == 0] = -1
        # Create watermark for mask
        w_m = d[inx]
        w_c = w_c.T
        w_m = np.array(w_m, dtype = np.float64)
        ww = np.concatenate((np.reshape(w_c, w_c.size, order = 'F'), w_m))
        
        
        www = np.zeros((param.M))
        www[0:len(ww)] = ww
        
        www1 = www[0:param.M:3]
        www2 = www[1:param.M:3]
        www3 = www[2:param.M:3]
        
        N = frame.shape[0] * frame.shape[1]
        m = np.round(param.mratio * param.N)
        
        s1 = np.array(frame[:, :, 0], dtype = np.float64)
        s2 = np.array(frame[:, :, 1], dtype = np.float64)
        s3 = np.array(frame[:, :, 2], dtype = np.float64)
        S = np.concatenate((np.reshape(s1, (s1.size, 1), order = 'F'), np.reshape(s2, (s2.size, 1), order = 'F'), np.reshape(s3, (s3.size, 1), order = 'F')), axis = -1)
        smean = np.mean(S, axis = 0)
        S = S - smean
        
        np.random.seed(1)
        temp1 = np.random.permutation(int(N))
        omega = np.array(temp1[0:int(m)], dtype = np.int32) # Pick up measurements randomly

        
        param.redundant = param.N - N
    
        signal1 = np.concatenate((S[:, 0], np.zeros(param.redundant, dtype = np.float64)), axis = 0)
        signal2 = np.concatenate((S[:, 1], np.zeros(param.redundant, dtype = np.float64)), axis = 0)
        signal3 = np.concatenate((S[:, 2], np.zeros(param.redundant, dtype = np.float64)), axis = 0)
        
        y1 = A_noiselet(signal1, omega)
        y2 = A_noiselet(signal2, omega)
        y3 = A_noiselet(signal3, omega)
        y = np.concatenate((y1, y2, y3), axis = 0)
              
             
        y_w = np.concatenate((y1, y2, y3), axis = 0)
        y_w = np.concatenate((y_w, smean), axis = 0)
        y_w = np.append(y_w, 0)
        y_w = np.append(y_w, 0)
        y_w = np.append(y_w, 0)
        y_w = np.append(y_w, 0)
        y_w = np.append(y_w, 0)
        y_w = np.append(y_w, 0)
        y_w = np.append(y_w, 0)
        sock_sender.send_numpy_array(y_w)
        
        counter = counter + 1
        # display the image to our screen
        #cv2.imshow("Frame", frame)
        #key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        #if key == ord("q"):
        #    break

        # update the FPS counter
        fps.update()
        continue
    
    # If there is face, do compression and encription
    mask = np.zeros(gray.shape);
    if len(boxes) > 0:
        (x, y, x_w, y_h) = boxes[0]
        mask[y:y_h + 1, x:x_w + 1] = 1;
        
    if mask.sum() > 15000.0 : continue # Too big
    
    (x_coor, y_coor, x_w_coor, y_h_coor) = boxes[0]
    
    MM = (np.random.rand(frame.shape[0], frame.shape[1]) < param.pr)
    MM = np.array(MM, dtype = np.float64)
    MM = MM*2 -1
    D = np.multiply(mask, MM)
    D2 = np.multiply(D, param.matrix[0:frame.shape[0], 0:frame.shape[1]])
    d = np.reshape(D, D.size, order = 'F')
    inx = np.where(np.reshape(mask, mask.size, order = 'F') == 1)[0]
    watermark_infD = np.reshape(D, D.size, order = 'F')

    # Create watermark coordinates
    w_c = np.zeros((4, 1), dtype = np.uint8)
    w_c[0] = boxes[0][0]
    w_c[1] = boxes[0][2] - boxes[0][0]
    w_c[2] = boxes[0][1]
    w_c[3] = boxes[0][3] - boxes[0][1]
    w_c = np.unpackbits(w_c, axis = 1)
    w_c = np.array(w_c, dtype = np.float64)
    w_c[w_c == 0] = -1
    # Create watermark for mask
    w_m = d[inx]
    w_c = w_c.T
    w_m = np.array(w_m, dtype = np.float64)
    ww = np.concatenate((np.reshape(w_c, w_c.size, order = 'F'), w_m))
    
    
    www = np.zeros((param.M))
    www[0:len(ww)] = ww
    
    www1 = www[0:param.M:3]
    www2 = www[1:param.M:3]
    www3 = www[2:param.M:3]
    
    N = frame.shape[0] * frame.shape[1]
    m = np.round(param.mratio * param.N)
    
    s1 = np.array(frame[:, :, 0], dtype = np.float64)
    s2 = np.array(frame[:, :, 1], dtype = np.float64)
    s3 = np.array(frame[:, :, 2], dtype = np.float64)
    S = np.concatenate((np.reshape(s1, (s1.size, 1), order = 'F'), np.reshape(s2, (s2.size, 1), order = 'F'), np.reshape(s3, (s3.size, 1), order = 'F')), axis = -1)
    smean = np.mean(S, axis = 0)
    S = S - smean
    
    np.random.seed(1)
    temp1 = np.random.permutation(int(N))
    omega = np.array(temp1[0:int(m)], dtype = np.int32) # Pick up measurements randomly
    
    p1 = m - param.M/3
    
    np.random.seed(2)
    temp2 = np.random.permutation(int(m))
    omega2 = np.array(temp2[0:int(p1)], dtype = np.int32) # Pick up measurements randomly
    
    P = 1.0/(param.m1*param.m2)*np.ones([param.m1, param.m2], dtype = np.float64)
    
    outside = (mask - 1) * (-1)
    
    param.redundant = param.N - N
    #S_new = np.concatenate((S, np.zeros([param.redundant, 3], dtype = np.float64)), axis = 0)
    signal1 = np.multiply(np.reshape(outside, outside.size, order = 'F'), S[:, 0]) + np.multiply(np.reshape(D, D.size, order = 'F'), S[:, 0])
    signal2 = np.multiply(np.reshape(outside, outside.size, order = 'F'), S[:, 1]) + np.multiply(np.reshape(D, D.size, order = 'F'), S[:, 1])
    signal3 = np.multiply(np.reshape(outside, outside.size, order = 'F'), S[:, 2]) + np.multiply(np.reshape(D, D.size, order = 'F'), S[:, 2])
    signal1 = np.concatenate((signal1, np.zeros(param.redundant, dtype = np.float64)), axis = 0)
    signal2 = np.concatenate((signal2, np.zeros(param.redundant, dtype = np.float64)), axis = 0)
    signal3 = np.concatenate((signal3, np.zeros(param.redundant, dtype = np.float64)), axis = 0)
    
    y1 = A_noiselet(signal1, omega)
    y2 = A_noiselet(signal2, omega)
    y3 = A_noiselet(signal3, omega)
    y = np.concatenate((y1, y2, y3), axis = 0)
    
    temp2 = np.ones(int(m), dtype = np.float64)
    temp2[omega2] = 0
    inn = np.where(temp2 == 1)[0]
    
    bw1 = At_dct(www1, int(m), inn)
    bw2 = At_dct(www2, int(m), inn)
    bw3 = At_dct(www3, int(m), inn)
    
    alpha1 = norm(y1) * param.em_power
    alpha2 = norm(y2) * param.em_power
    alpha3 = norm(y3) * param.em_power
    
    bw1 = (bw1/norm(bw1)) * alpha1
    bw2 = (bw2/norm(bw2)) * alpha2
    bw3 = (bw3/norm(bw3)) * alpha3
    
    # Compute v
    bw_inn1 = A_dct(bw1, inn)
    bw_inn2 = A_dct(bw2, inn)
    bw_inn3 = A_dct(bw3, inn)
    v1 = np.abs(bw_inn1[0])
    v2 = np.abs(bw_inn2[0])
    v3 = np.abs(bw_inn3[0])
    
    
    y_w = np.concatenate((y1 + bw1, y2 + bw2, y3 + bw3), axis = 0)
    y_w = np.concatenate((y_w, smean), axis = 0)
    y_w = np.append(y_w, v1)
    y_w = np.append(y_w, v2)
    y_w = np.append(y_w, v3)
    y_w = np.append(y_w, x_coor)
    y_w = np.append(y_w, y_coor)
    y_w = np.append(y_w, x_w_coor)
    y_w = np.append(y_w, y_h_coor)
    sock_sender.send_numpy_array(y_w)
    
    counter = counter + 1
    # display the image to our screen
    #cv2.imshow("Frame", frame)
    #key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    #if key == ord("q"):
    #    break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("Elasped time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()