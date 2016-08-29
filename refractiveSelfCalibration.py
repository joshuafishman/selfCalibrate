# -*- coding: utf-8 -*-
"""
Created on Mon ‎Jul ‎18 ‏‎13:14:43 2016

@author: JoshuaF
"""

import os
import sys
import inspect
import glob
import copy
import tiffcapture as tc
import numpy as np
import cv2
import numpy.linalg as lin
import cv2.cv as cv
import math
import itertools
import warnings
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #warning is inaccurate, this is used to plot in 3D



### This code performs the "Belden" method of self-calibration for multiple cameras in a refractive system 



#################################     Class definitions    ##########################################


class parameters(object):    
#basic parameter storage object
    
    def __iter__ (self):
        for name,val in self.__dict__.iteritems():
            yield name,val
    
    def params (self): #get parameter name-value pairs in an instance of class
        return dict(self)

    @classmethod
    def getInfo (cls):
        # get the names of input arguments passed into class and the parameters it holds (not always the same as input)
        Input = inspect.getargspec(cls.__init__)[0][1:] #get arguments passed into __init__
        
        o = object.__new__(cls)  #create a new dummy object of type (there is probably an easier way than this)
        o.__init__(*range(len(Input))) #initialize dummy object 
        Params = [name for name in o.params()] #get the names of parameters it holds
        
        return {"Input": Input, "Params": Params}
        
                                   
class planeData(parameters):
    #object containing parameters of the calibration planes/grids  
    def __init__(self,dX,dY,nX,nY,ncalplanes,znet= None, z0 = None):
        self.dX         = dX
        self.dY         = dY
        self.nX         = int(nX) #these need to be ints
        self.nY         = int(nY)
        self.ncalplanes = int(ncalplanes)
        if z0:
            self.z0     = z0
        elif znet:
            self.z0     = znet/(ncalplanes-1)*np.linspace(0,1,ncalplanes) #calculate origin for n evenly spaced calibration planes
        else:
            raise Exception ("Input either z coordinates for each calibration plane (z0) or a net z traverse (znet).")                

class sceneData(parameters):
    #object containing parameters of the experimental setting
    def __init__(self,n1,n2,n3,tW,zW):
        self.n  = [n1,n2,n3]
        self.tW = tW
        self.zW = zW

class cameraData(parameters):
    #object containing parameters of the calibration images   
    def __init__(self,sX,sY,pitch,so,f,ncams,nframes):
        self.sX       = sX
        self.sY       = sY
        self.shiftx   = sX/2
        self.shifty   = sY/2
        self.mpix     = 1/pitch
        self.f        = f
        self.so       = so
        self.a0       = 1/pitch*f
        self.ncams    = int(ncams) #these need to be ints
        self.nframes  = int(nframes)
        self.pix_phys = None

class refracTol(parameters):
    #object containing error tolerances for solvers
    def __init__(self,tol,fg_tol,maxiter,bi_tol,bi_maxiter,z3_tol,rep_err_tol):
        self.tol         = float(tol)
        self.fg_tol      = float(fg_tol)
        self.maxiter     = int(maxiter)
        self.bi_tol      = float(bi_tol)
        self.bi_maxiter  = int(bi_maxiter)
        self.z3_tol      = float(z3_tol)
        self.rep_err_tol = float(rep_err_tol) 
    
    
    
########################### Multipage Tiff handling functions #####################################

    
def getSpacedFrameInd(tiff, n, nframes = 0) :
    #Generate indices of n evenly spaced frames in a multipage tiff
    #Inputs:
    # tiff       - location of tiff being calibrated
    # n          - number of indices to generate  
    # nframes    - number of frames in each tiff -- if not provided, program will try to determine (slow) 
    #
    #Outputs:
    # frameind   - indices of frames to be used for calibration
    
    # try to get the total number of images in each calibration file -- doesn't always work
    if not nframes:
        nframes  = tc.opentiff(tiff)._count_frames()
        if nframes == 0:
            raise Exception ('Unable to count frames in image ' + tiff)
    
    frameind = np.linspace(0,nframes-1,num=n,endpoint=True,retstep=False)
    frameind = np.uint16(frameind)
    
    return frameind
    
def saveCalibImagesTiff(datapath, exptpath, camNames, ncalplanes, nframes = 0):
    # get correct images for calibration from multipage tiff and save them in a folder called 'calibration'
    #Inputs:
    # datapath   - path to stored images
    # exptpath   - path to output saved data
    # camNames   - names of cameras to which images belong
    # ncalplanes - number of calibration planes in images
    # nframes    - number of frames in each tiff -- if not provided, program will try to determine
    #
    #Outputs:
    # Images saved in folders with the camera names inside a folder called 'calibration'
    # outpath    - path to saved data
    
    ncams = len (camNames)        
    cams  = glob.glob(os.path.join(datapath,'*.tif'))
    
    outpath = os.path.join(exptpath,'calibration')
    
    # find calibration files to use
    img = [y for y in cams for i in range(0, ncams) if camNames[i] in y]
    if not len (img) == ncams:
        raise Exception ("Error parsing files in " +datapath + ". Check the names of .tif files.")
    
    # find indices of frames of tiff to be used for calibration
    frameind = getSpacedFrameInd(img[0], ncalplanes, nframes)
    
    # setup folders by camera
    for i in range(0,ncams):
        seq = tc.opentiff(img[i])
    
        # output file details
        folder = os.path.join(outpath,camNames[i])
        if not os.path.exists(folder):
            os.makedirs(folder)
    
        if not os.listdir(folder):
            print('Copying images')
    
            for j in range(0,ncalplanes):
                try:
                    ical    = seq.find_and_read(frameind[j])
                except Exception as exc:      
                   raise Exception((str(exc)+'. Unable to create image ' +str(j) + ' in ' +img[i] +'. ')), None, sys.exc_info()[2] 
                    
                outfile = os.path.join(folder,str(j).zfill(2)+'.tif')
                cv2.imwrite(outfile,ical)
                
    return outpath

def getCalibImagesTiff(datapath, camNames, ncalplanes, nframes = 0):
    # return correct images for calibration from multipage tiff 
    #Inputs:
    # datapath   - path to stored images
    # exptpath   - path to output saved data
    # camNames   - names of cameras to which images belong
    # ncalplanes - number of calibration planes in images
    # nframes    - number of frames in each tiff -- if not provided, program will try to determine (slow)
    #
    #Outputs:
    # ical       - images to use for calibration ([ncams[nplanes]])
    
    ncams = len (camNames)        
    cams  = glob.glob(os.path.join(datapath,'*.tif'))
    
    # find calibration files to use
    img = [y for y in cams for i in range(0, ncams) if camNames[i] in y]
    if not len (img) == ncams:
        raise Exception ("Error parsing files in " +datapath + ". Check the names of .tif files.")
    
    # find indices of specific images in tiff to be used for calibration
    frameind = getSpacedFrameInd(img[0], ncalplanes, nframes)
    
    print ('\nFinding frames ' +str(frameind)+ ' in images '+str(img)+'\n')
    
    try:
        ical   = [[tc.opentiff(i).find_and_read(frameind[j]) for j in range(ncalplanes)] for i in img]  
    except Exception as exc:     
        try: #whoooah meta
            raise Exception((str(exc)+'. Unable to create image ' +str(j) + ' in ' +i +'. ')), None, sys.exc_info()[2]
             #should work, but it plays a bit fast and loose with scope so:
        except UnboundLocalError:
            raise exc #less specific 
    
    return ical
            
            
            
################################ Corner finding functions #############################################            
            
            
def preprocess(img, t):
    # preprocess image for better corner finding
    #Inputs:
    # img  - image to be preprocessed
    # t    - limit of max of image / mean of image
    #Outputs:
    # img  - processed image 
    # std  - standard deviation of image (grayscale)
    # mu   - mean of image (grayscale)
    # m/mu - max of image / mean of image (grayscale)
    
    img = copy.copy(img)
    std, mu, m = np.std(img), np.mean(img), np.max(img)
    if m/mu > t:
        img[img>mu+std] = mu
    return img, std, mu, m/mu

def fixCorners(points,nX,nY, numMissing):
    # replace missing corners in a grid of found corners and reorder them by ascending x (right), descending y (up), by row first 
    #Inputs:
    # points     - found points
    # nX,nY      - number of points per row, column on grid
    # numMissing - number of points missing 
    #            
    #Outputs:
    # sPoints    - sorted full points array 
    
    tol = .04 
    pi = math.pi 
    
    ##### Helper functions ####
    
    #function for distance between 2 points
    dist =  lambda p1,p2: \
            np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2) 
    
    #function for angle between p1 and p2 (out of 2pi)
    def getAngle (p1,p2): 
        theta = math.atan2((p2[1]-p1[1]),(p2[0]-p1[0])) if not np.array_equal(p1, p2) else 0
        return theta if theta >= 0 else 2*pi+theta
        
    #function for difference between angles t1 and t2 out of 2pi
    angleDiff = lambda t1,t2: \
                abs(t1-t2) if abs(t1-t2)<pi else 2*pi-abs(t1-t2)    
    
    #function for range of angles theta. Polar plot angles if show_angles.    
    def angleRange(theta , show_angles= False):
        theta = [t if t >= 0 else 2*pi+t for t in theta] 
       
        my  = np.mean(np.sin(theta))
        mx  = np.mean(np.cos(theta))
        mid = math.atan2(my,mx) #mean angle in range in -pi to pi
        mid = mid if mid>=0 else 2*pi+mid #convert to 0-2pi scale
        
        if show_angles:
            figP = plt.figure('polar')
            figP.clear()
            ax = plt.subplot(111, projection='polar')
            ax.plot(theta, np.ones(len(theta)),'.')
            ax.plot(mid, 1,'gp')
            plt.show()
        
        #split the angles into 2 even groups
        if mid <= pi:  
            above = [t for t in theta if t > mid and t < mid+pi]
        else:
            above = [t for t in theta if t < mid and t > mid-pi]
        below = [t for t in theta if t not in above]
            
        mAbove = max(angleDiff(t,mid) for t in above) if above else 0
        mBelow = max(angleDiff(t,mid) for t in below) if below else 0
         
        return mAbove + mBelow

    #function to check if a point is on the edge of a grid of points. Polar plot of points if show_angles
    isEdge   = lambda pt, pts, show_angles=False: \
               angleRange( np.array([getAngle(pt, p) for p in pts if not np.array_equal(p,pt)]), show_angles) <= pi*(1+tol)        
    
    #function to check if a point is on the corner of a grid of points. Polar plot of points if show_angles
    isCorner = lambda pt, pts, show_angles= False:  \
               angleRange( np.array([getAngle(pt, p) for p in pts if not np.array_equal(p,pt)]), show_angles) <= 2*pi/3*(1+tol)        
        
    
    #function to find up to num adjacent points and their angles relative to pt for a pt on a grid of pts         
    def adjacent(pt, pts, num):
        angles   = []
        adj      = []
        
        for p in sorted(pts, key = lambda p: dist(pt,p)): 
            angle = getAngle(pt,p)
            
            if any(abs((a-angle)/angle) < tol for a in angles) or np.array_equal(p, pt):
                continue   
            
            adj.append(p)
            angles.append(angle)
            
            if len(adj) == num:
                return np.array(adj),angles
        
        return np.array(adj),angles
    
    #function to find the intersection of 2 lines (format of a line if [m,b] for y = mx+b)
    def intersection (l1, l2):
        m1, b1 = l1[0], l1[1]
        m2, b2 = l2[0], l2[1]
        x = (b2-b1)/(m1-m2)
        y = m1*x+b1 if not m1*x+b1 == 0 else m2*x+b2  #not sure why this happens, but if it's really 0 the other line should give the same answer
          
        return x,y
    
    
    
    ### Actual point replacement ###   
    ret = 0
    #First, replace any of the four corners of the grid if they're missing
    edges   = np.array([p for p in points if isEdge(p,points)])
    corners = np.array([p for p in points if isCorner(p,points)])
    
    cLines = []
    
    # find the lines y=mx + b describing the sides of the grid
    for n,c in enumerate(corners): 
        for i,ang in enumerate(adjacent (c,edges,2)[1]):
            angles = [getAngle(c,e) for e in edges if abs((getAngle(c,e)-ang)) < tol*pi/2]
            m      = np.mean(np.sin(angles)) / np.mean(np.cos(angles)) #slope of the line is the tangent of the angle it makes with the x axis
            b      = c[1] - m*c[0]   #y = mx+b
            if not any (line == [m,b] for line in cLines):            
                cLines.append([m,b])       
    
    #edge lines            
    cLines = sorted (cLines, key = lambda l: abs(l[0])) #order lines by slope; should be 2 pairs of similar slope
    
    # find corners at the intersection of lines [0,3], [0,4], [1,3], [1,4] 
    cornersFound = np.array([intersection(cLines[int(i/2)], cLines[2+i%2]) for i in range (4)])   
    
    # add missing corners to found points array  
    #fPoints - found points array
    fPoints = np.append(points,[c for c in cornersFound if not any(np.array_equal(c,c1) for c1 in corners)], axis=0)
    
    if len(cornersFound) !=4:
        print ('Unable to find corner or missing internal point. Found ' + str(len(cornersFound)) + ' corners.') 
        return ret, fPoints
    
    #reconstruct and sort edges
    edges  = [p for p in fPoints if isEdge(p,fPoints)]
    edges  = [sorted([p for p in edges if abs(2*(p[1]-l[0]*p[0]-l[1])/(p[1]+l[0]*p[0])) < tol], key = lambda p: p[1]) for l in cLines]
    
    #edges should be sorted into 4 groups now -- figure out which is which    
    top    = np.array(edges[1])
    bottom = np.array(edges[0]) 
    if np.mean(bottom[:,1]) > np.mean(top[:,1]):
        bottom,top = top,bottom
        
    left   = np.array(edges[2])
    right  = np.array(edges[3])
    if np.mean(left[:,0]) > np.mean(right[:,0]):
        left,right = right,left
    
    #using the sorted edges, break the grid into rows and columns
    if len(bottom) == nX:
        theta  = math.atan(cLines[3][0])
        columns = [sorted([p for p in fPoints if abs(angleDiff(getAngle(bot, p), theta)) < tol*pi/2 or np.array_equal(p, bot)], key = lambda p: p[1]) for bot in bottom] 
    elif len(top) == nX:
        theta  = math.atan(-cLines[3][0])
        columns = [sorted([p for p in fPoints if abs(angleDiff(getAngle(t, p), theta)) < tol*pi/2 or np.array_equal(p, t)], key = lambda p: p[1]) for t in top] 
    else:
        #future work: match up points on opposite edges
        print("Missing parallel edge points -- unable to reconstruct grid")
        return ret, fPoints
    
    if len(left)   == nY:    
        theta   = math.atan(cLines[0][0])
        rows    = [sorted([p for p in fPoints if abs(angleDiff(getAngle(lef, p), theta)) < tol*pi/2 or np.array_equal(p, lef)], key = lambda p: p[0])  for lef in left]
    elif len(right) == nY:
        theta   = math.atan(-cLines[0][0])
        rows    = [sorted([p for p in fPoints if abs(angleDiff(getAngle(rig, p), theta)) < tol*pi/2 or np.array_equal(p, rig)], key = lambda p: p[0]) for rig in right]
    else:
        print("Missing parallel edge points -- unable to reconstruct grid")
        return ret, fPoints
   
    #using the rows and columns, find the lines defining the grid       
    colLines = []
    for col in columns:
        angles = [getAngle(col[0],p) for p in col[1:]] 
        m      = np.mean(np.sin(angles)) / np.mean(np.cos(angles)) #slope of the line is the tangent of the angle it makes with the x axis
        b      = col[0][1] - m*col[0][0]   #y = mx+b
        colLines.append([m,b])
    if len (colLines) != nX:
        print("Wrong number of columns.")
        return ret, fPoints
    
    rowLines = []    
    for row in rows:
        angles = [getAngle(row[0],p) for p in row[1:]] 
        m      = np.mean(np.sin(angles)) / np.mean(np.cos(angles)) #slope of the line is the tangent of the angle it makes with the x axis
        b      = row[0][1] - m*row[0][0]   #y = mx+b
        rowLines.append([m,b])
    if len (rowLines) != nY:
        print ("Wrong number of rows.")  
        return ret, fPoints
    

    #reconstruct the grid points by placing a point at each grid intersection    
    fPoints = np.array([intersection(colLines[c], rowLines[r]) for c in range (nX) for r in range (nY)])
    
    
    #xRange = np.linspace(min(p[0] for p in fPoints), max(p[0] for p in fPoints),50)
    #yRange = np.linspace(min(p[1] for p in fPoints), max(p[1] for p in fPoints),50)
    #colPoints = np.array([( (y-l[1])/l[0], y) for l in colLines for y in yRange])
    #rowPoints = np.array([(x , l[0]*x+l[1])   for l in rowLines for x in xRange])
    
    #figL = plt.figure('Found Lines')
    #figL.clear() 
    #plt.plot(colPoints[:,0],colPoints[:,1])
    #plt.plot(rowPoints[:,0],rowPoints[:,1])
    #plt.plot(fPoints[:,0],fPoints[:,1],'p')
   
    
    ### Reordering ###
    #by ascending x (right), descending y (up), by row first

    theta  = math.atan(cLines[3][0])
    start  = sorted([p for p in fPoints if isCorner(p,fPoints)], key= lambda p:p[1])[0]
    edges  = np.array([p for p in fPoints if isEdge(p,fPoints)])
    bottom = sorted(edges, key =  lambda p: abs(angleDiff(getAngle(start,p), math.atan(cLines[1][0]))) if not np.array_equal(p, start) else 0)[:nX]
    bottom = sorted(bottom, key = lambda p: p[0])
    #break the grid into vertical lines and find points on each in order
    #sPoints - sorted points array
    sPoints = [sorted([p for p in fPoints if abs(angleDiff(getAngle(bot, p), theta)) < tol*pi/2 or np.array_equal(p, bot)], key = lambda p: p[1]) for bot in bottom]   
    try:    
        sPoints = np.array([sPoints[x][y] for y in range(nY) for x in range(nX)])
        ret = 1 #success!
    except:
        print ('  Unable to find point ' + str([x+1,y+1]))
        sPointsFail = []
        for col in sPoints:
            for point in col:
                sPointsFail.append(point)
        sPoints = np.array(sPointsFail)
    
    return ret, sPoints
    
def findCorners(pData, camnames, datapath = [], imgs =[], exptpath =[], show_imgs = False, debug = False):
    #Find chessboard corners on grid images, either passed in directly or in a given location
    #Inputs:
    # camnames          - names of cameras to which images belong
    # pData:
    #     ncalplanes    - number of planes
    #     nX,nY         - number of grid points per row and column on each plane
    # exptpath          - optional; location to save a file containing corners 
    # datapath          - location to get images from if they aren't being passed in directly
    # OR
    # imgs              - array of images in which to find corners (order should be [camera[image]]).
    # show_imgs         - boolean; plot found corners on images         
    # debug             - boolean; if cornerfinder is failing, find out which points failed in which images          
    #
    #Outputs:
    # cor_all           - 2 by nX*nY*ncalplanes by ncams array of corner coordinates
    # saved corners.dat file containing corner coordinates on exptpath if it's provided
    
    ncams         = len(camnames)
    ncalplanes    = pData.ncalplanes
    nX,nY         = pData.nX, pData.nY
    
    if any(imgs) :
        imgloc = 'imgs' # images were passed directly         
    elif any(datapath) and not any(imgs): 
        imgloc = 'path' # images not passed, so they should be found on the path
        cams   = sorted(glob.glob(datapath)) # folder holding image folders must be named 'calibration'
        if len(cams)<ncams:
            if not cams:
                raise Exception ("Calibration image folder in " +datapath+" not found or empty.")
            else:    
                raise Exception ("Not enough camera folders in "+datapath)
    else:
        raise Exception ("Images not passed to corner finder. Either provide them directly (imgs argument) or provide a path to their location (datapath argument)")
    
    if exptpath:
        f = open(os.path.join(exptpath,'corners.dat'), 'w')
        f.write(str(ncalplanes)+'\n')
        f.write(str(ncams)+'\n')

    cor_all = np.zeros([2,ncalplanes*nX*nY,ncams])
    
    if debug:
        failed = {name:[] for name in camnames} #setup a dictionary for cornerfinder failures
        
    for i in range(0,ncalplanes):
        for j in range(0,ncams):
            try:
                #get image
                if imgloc == 'path':
                    files = sorted(glob.glob(os.path.join(cams[j],'*.tif')))
                    if len(files) < ncalplanes:
                        raise Exception ("Less than " +str(ncalplanes) + " images in " + cams[j] + " folder.")
                    file  = files[i]
                    I     = cv2.imread(file, 0)
                    print ('Finding corners in '+file)
                    
                else:  
                    I = imgs[j][i]
                    print ('Finding corners in image '+str(i+1) +' in camera ' +camnames[j])
                
                    
                # Find corners then refine
                ret, corners = cv2.findChessboardCorners(I,(nX,nY))
                #print  corners.shape
                numfound     = 0 if corners is None else len(corners) #number of corners found
                    
                if not ret or not numfound==nX*nY: #cornerfinder failed                    
                    print (' Cornerfinder failed -- preprocessing image')
                    Inew          = np.array(preprocess (I, 2)[0], dtype = 'uint8') # 2 is a hardcoded value
                    ret, corners  = cv2.findChessboardCorners(Inew,(nX,nY))
                    corners.shape = (len(corners),1,2)
                    numfound      = 0 if corners is None else len(corners) 
                            
                    if not ret or numfound==nX*nY: #preprocessing failed
                        print (' Preprocessing failed. Reconstructing corners.')      
                        # try to recontruct grid
                        corners.shape = (len(corners),2)
                        ret, corners  = fixCorners(corners, nX, nY, nX*nY-numfound)
                        corners       = corners.astype('float32')
                        numfound      = len(corners) 
                    
                        if ret and numfound==nX*nY: #reconstruction successful 
                            figC = plt.figure('Corners found in image ' +str(i+1) + ' in camera ' +camnames[j] +'-some reconstructed')                
                            plt.imshow(I, cmap='gray')
                            plt.plot(corners[:,0],corners[:,1])
                            plt.show()    
                                         
                            corners.shape = (nX*nY,1,2) #reshape corners to shape expected by openCV
    
                        else:    
                            print ('  Reconstruction failed.')
                           
                            if not debug or show_imgs:
                                figC = plt.figure('Corners found in image ' +str(i+1) + ' in camera ' +camnames[j] + ' (failed)')                
                                plt.plot(corners[:,0],corners[:,1],'.')
                                plt.imshow(I, cmap='gray')
                                plt.show() 
                                
                            if debug: #trying to find all failed images  
                                failed[camnames[j]].append([i,numfound]) #add [image, number of points found] to the failed dictionary
                                continue  #skip to next iteration  
                             
                            raise Exception ('Failed: only found ' + str(numfound) + ' corners in image ' +str(i+1) + ' in camera ' + camnames[j] + '.')
                            
                    print (' Processing successful.')    
                                   
                
                cv2.cornerSubPix(I, corners, (10,10), (-1,-1), (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 10, 0.01))

                
                if show_imgs: #show images is on
                    figC = plt.figure('Corners found in image ' +str(i+1) + ' in camera ' +camnames[j])      
                    figC.clear()
                    plt.plot(corners[:,0,0], corners[:,0,1])
                    plt.imshow(I, cmap='gray')
                    plt.show()
            
                Dir = (corners[1]-corners[2])[0][0]         
                N   = len(corners)-1  
                cor = np.empty((nX*nY,1,2))
               
                for k in range(0,nX):                     
                    for l in range(0,nY): 
                            if (Dir>0):
                                cor[k*nY+l] = corners[N - (k + nX*l)]
                            else:
                                cor[k*nY+l] = corners[k + nX*l]                          
                
                # corner order is ascending x (right), ascending y (down), by column first 
                cor = cor.reshape(nX*nY,2)
                
                #if i == 0 and j == 0:
                #    cor_all = cor
                #else:
                #    cor_all = np.concatenate((cor_all,cor), axis=0)
        
                cor_all[:,i*nX*nY:(i+1)*nX*nY,j] = cor.T
                
                if exptpath:
                    for c in cor:
                        f.write(str(c[0])+'\t'+str(c[1])+'\n')
            
            except Exception as exc:      
                raise Exception((str(exc)+' Failed on image ' +str(i+1) + ' in camera ' + camnames[j] +'. ')), None, sys.exc_info()[2]                             

    print '\nDONE finding corners!'
    
    if debug:
        print ('/nFailed:/n'+str(failed))
 
    return cor_all

def getScale (u, nX, nY, dX, imgnum, camnum):
    # get the physical size of a pixel on the grid in an image in a camera from corners
    #Inputs:
    # u      - corners file for all images [2xnX*nYxncams]
    # nX,nY  - points per row, column on the grid
    # dX     - horizontal distance between grid points
    # imgnum - image in which to find scale
    # camnum - camera for which to find scale
    #    
    #Outputs:
    # scale  - physical size of a pixel in image
    
    # get 2 points a whole row apart
    p1 = u[:,nX*nY*(imgnum-1),camnum]
    p2 = u[:,nX*nY*(imgnum-1)+nY*(nX-1),camnum]

    # plot a full grid of points
    figG = plt.figure('Grid Points in Camera ' +str(camnum) +', Image ' +str(imgnum))
    figG.clear()
    plt.axis([0,1280,0,800])
    plt.plot(u[0,nX*nY*(imgnum-1):nX*nY*(imgnum), camnum], u[1,nX*nY*(imgnum-1):nX*nY*(imgnum), camnum])
    plt.plot([p1[0],p2[0]], [p1[1],p2[1]],'r+')
    plt.show()
    
    scale = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)/(dX*nX)    
    return scale
    
    

####################################### Calibration functions #############################################

    
def setupCamera(camData,nparam=7):
    #Generate initial guesses for camera parameters
    #Inputs:
    # ncams      - number of cameras
    # nparam     - number of parameters in model (default is 7, which is what the rest of the functions expect)
    # camData:
    #   a0       - magnification
    #
    #Output:
    # cam_params - nparam x ncams matrix of camera parameters
    
    a0 = camData.a0
    ncams = camData.ncams
    
    cam_params = np.zeros([nparam,ncams])
    cam_params[6]=a0*np.ones([1,ncams])
    
    return cam_params

def setupPlanes(nplanes,z0):
    #Generate initial guesses for plane parameters
    #Inputs:
    # nplanes      - number of planes in calibration images
    # z0           - origin of z axis for each plane
    #
    #Outputs:
    # plane_params - 6xNplanes matrix of plane parameters
    
    plane_params = np.zeros([6,nplanes])
    plane_params[5]=z0
    
    return plane_params
    
def planar_grid_to_world(plane_params,xyzgrid,planeData):
    #Calculate world points based on plane parameters
    #Inputs:
    # plane_params - 6 x nplanes matrix of plane parameters
    # xyzgrid      - 3 x nX*nY matrix containing points on a grid
    # planeData:
    #   nplanes    - number of calibration planes 
    #   nX,nY      - number of grid points per row, column on each plane
    #    
    #Output:
    # X            - World Points (3 x nX*nY*nplanes)
    
    nplanes         = planeData.ncalplanes
    nX,nY           = planeData.nX, planeData.nY
    phi,alpha,theta = plane_params[0:3]
    XG              = plane_params[3:6]
    
    for j in range(0,nplanes):
        Rot = np.array([[math.cos(theta[j])*math.cos(phi[j]) + math.sin(theta[j])*math.sin(alpha[j])*math.sin(phi[j]), math.sin(theta[j])*math.cos(alpha[j]), -math.cos(theta[j])*math.sin(phi[j]) + math.sin(theta[j])*math.sin(alpha[j])*math.cos(phi[j])], \
                        [-math.sin(theta[j])*math.cos(phi[j]) + math.cos(theta[j])*math.sin(alpha[j])*math.sin(phi[j]), math.cos(theta[j])*math.cos(alpha[j]), math.sin(theta[j])*math.sin(phi[j]) + math.cos(theta[j])*math.sin(alpha[j])*math.cos(phi[j])], \
                        [math.cos(alpha[j])*math.sin(phi[j]),-math.sin(alpha[j]), math.cos(alpha[j])*math.cos(phi[j])]])

        xtmp = np.reshape(XG[:,j],(3,1))
        XYZgrid=np.dot(lin.inv(Rot),xyzgrid)+np.repeat(xtmp,nX*nY,axis=1)

        if j == 0:
            X = XYZgrid
        else:
            X = np.concatenate((X,XYZgrid),axis=1)
    return X

def ptnormalize(x):
    # Rescale and shift points to improve conditioning
    #
    #Inputs:
    # x     - points to be normalized (world or image)
    #
    #Outputs:
    # xnorm - normalized points
    # T     - matrix for reversing normalization
    
    M = x.shape[2]

    if x.shape[0]==2:

        xnorm = np.empty_like(x)
        T = np.zeros([3,3,x.shape[2]])

        for j in range (0,M):
            xtemp = x[:,:,j]
            xm = -(np.mean(xtemp,axis=1,keepdims=True))
            xs = np.sum(np.square(xtemp+np.repeat(xm,x.shape[1],axis=1)),axis=0)
            d = np.sqrt(xs)
            s = np.sqrt(2)/(np.sum(d)/len(d))

            T[:,:,j] = ([s,0,s*xm[0]], \
                        [0,s,s*xm[1]], \
                        [0,0,1])

            xpad = np.append(xtemp,np.ones((1,x.shape[1])),axis=0)
            xnormtemp = np.dot(T[:,:,j],xpad)
            xnorm[:,:,j]=xnormtemp[0:2,:]

    elif x.shape[0]==3:

        xnorm = np.empty_like(x)
        T = np.zeros([4,4,x.shape[2]])

        for j in range (0,M):
            xtemp = x[:,:,j]
            xm = -(np.mean(xtemp,axis=1,keepdims=True))
            xs = np.sum(np.square(xtemp+np.repeat(xm,x.shape[1],axis=1)),axis=0)
            d = np.sqrt(xs)
            s = np.sqrt(2)/(np.sum(d)/len(d))

            T[:,:,j] = ([s,0,0,s*xm[0]], \
                        [0,s,0,s*xm[1]], \
                        [0,0,s,s*xm[2]], \
                        [0,0,0,1])
            xpad = np.append(xtemp,np.ones((1,x.shape[1])),axis=0)
            xnormtemp = np.dot(T[:,:,j],xpad)
            xnorm[:,:,j]=xnormtemp[0:3,:]

    else:
        print 'error normalizing'
        print x.shape[0]

    return (xnorm,T)

def P_from_DLT(X,u):
    #Camera matrix from direct linear transform
    #Inputs:
    # X - 3D world points
    # u - 2D image points
    #
    #Outputs:
    # P - 3 x 4 x # of cameras camera matrix
    
    if len(X.shape)>2:
        p = X.shape[2]
    else:
        p = 1
    if len(u.shape)>2:
        M = u.shape[2]
    else:
        M=1

    if p < M:
        Xtmp = np.zeros([X.shape[0],X.shape[1],M])
        for j in range (0,M):
            Xtmp[:,:,j]=X
        X = Xtmp

    #Condition points
    u,T=ptnormalize(u)
    X,U=ptnormalize(X)
    
    P = np.zeros([3,4,M])

    for j in range (0,M):
        utmp = u[:,:,j]
        xtmp = X[:,:,j]
        c = utmp.shape[1]
        zv = np.zeros([1,4])

        for k in range (0,c):
            u1 = np.append(utmp[:,k],1)
            X1 = np.append(xtmp[:,k],1)
            Dtmp = np.zeros([2,12])
            Dtmp[0,0:4]=u1[2]*X1
            Dtmp[0,4:8]=zv
            Dtmp[0,8:12]=-u1[0]*X1
            Dtmp[1,0:4]=zv
            Dtmp[1,4:8]=-u1[2]*X1
            Dtmp[1,8:12]=u1[1]*X1
            if k == 0:
                D=Dtmp
            else:
                D=np.concatenate((D,Dtmp),axis=0)
        
        # SVD
        uS,sS,vS = lin.svd(D, full_matrices=False)
        
        p = vS[11,:]
        p = np.reshape(p,[3,4])
        
        # Decondition
        P[:,:,j]=np.dot(np.dot(lin.inv(T[:,:,j]),p),U[:,:,j])
        P[:,:,j]=P[:,:,j]/P[2,3,j]
        
    return P

def cam_decomp(p):
    #Camera center by SVD
    #Inputs:
    # p - camera matrix
    #Outputs:
    # C - coordinates of camera center
    
    u,s,v=lin.svd(p, full_matrices=True)
    C = v[3,:]/v[3,3]
    C = C[0:3]
    return C

def f_eval_1eq(r1,r2,z1,z2,n1,n2):
    #evaluate ray tracing function, gradients for one refractive equation (snell's law with 2 media)
    #Inputs:
    # n1,2     - indices of refraction
    # r1,2     - length of ray in each refractive medium
    # z1,2     - z coordinate of ray intersection with each medium
    #
    #Outputs:
    # f        - output of ray tracing equation for one refractive interface
    # df_dr1   - derivative of f with regards to r1
    
    r1,r2,z1,z2,n1,n2 = np.array([r1,r2,z1,z2,n1,n2])+0. #convert all values to floats

    f      = (r1)/np.sqrt(r1**2 + z1**2) \
             - (n2/n1)*(r2-r1)/np.sqrt((r2-r1)**2 + z2**2)

    df_dr1 = 1/np.sqrt(r1**2 + z1**2) - r1**2/(r1**2 + z1**2)**(3./2) \
             + (n2/n1)/np.sqrt(z2**2 + (r1-r2)**2) \
             - (n2/n1)*(r1-r2)*(2*r1-2*r2)/(2*((r1-r2)**2 + z2**2)**(3./2))

    return (f,df_dr1)

def f_eval_2eq(r1,r2,r3,z1,z2,z3,n1,n2,n3):
    # evaluate ray tracing function, gradients for 2 refractive equations in (snell's law with 3 media) 
    #Inputs:
    # n1,2,3     - indices of refraction
    # r1,2,3     - length of ray in each refractive medium
    # z1,2,3     - z coordinate of ray intersection with each medium
    #
    #Outputs:
    # f          - first ray tracing equation for 2 refractive interfaces
    # g          - second ray tracing equation for 2 refractive interfaces
    # df_dr1     - derivative of f with regards to r1
    # dg_dr1     - derivative of g with regards to r1
    
    [r1,r2,r3,z1,z2,z3,n1,n2,n3]=np.array([r1,r2,r3,z1,z2,z3,n1,n2,n3])+0. #convert all values to floats
    
    f = (r1)/np.sqrt(r1**2. + z1**2.) \
        - (n2/n1)*(r2-r1)/np.sqrt((r2-r1)**2. + z2**2.)

    df_dr1 = 1./np.sqrt(r1**2. + z1**2.) \
             - r1**2./((r1**2. + z1**2.)**(3./2.)) \
             + (n2/n1)/np.sqrt(z2**2. + (r1-r2)**2.) \
             - (n2/n1)*(r1-r2)*(2.*r1-2.*r2)/(2.*((r1-r2)**2. + z2**2.)**(3./2.))

    df_dr2 = (n2/n1)*(r1-r2)*(2.*r1-2.*r2)/(2.*((r1-r2)**2. + z2**2.)**(3./2.)) \
             - (n2/n1)/np.sqrt(z2**2. + (r1-r2)**2.)


    g      = (r2-r1)/np.sqrt((r2-r1)**2. + z2**2.) \
             - (n3/n2)*(r3-r2)/np.sqrt((r3-r2)**2. + z3**2.)

    dg_dr1 = (r1-r2)*(2*r1-2*r2)/(2.*((r1-r2)**2. + z2**2.)**(3./2.)) \
             - 1./np.sqrt(z2**2. + (r1-r2)**2.)

    dg_dr2 = 1./np.sqrt((r1-r2)**2. + z2**2.) \
             + (n3/n2)/np.sqrt(z3**2 + (r2-r3)**2.) \
             - (r1-r2)*(2.*r1-2.*r2)/(2.*((r1-r2)**2. + z2**2.)**(3./2.)) \
             - (n3/n2)*(r2-r3)*(2.*r2-2.*r3)/(2.*((r2-r3)**2. + z3**2.)**(3./2.))

    return (f,df_dr1,df_dr2,g,dg_dr1,dg_dr2)

def NR_1eq(r1,r2,z1,z2,n1,n2,tol):
    #Newton-Raphson Iteration for 1 equation (only 2 refractive media i.e. neglecting tank wall)
    #Inputs:
    # maxIter    - max # of iterations for solver
    # n1,2       - indices of refraction
    # r1,2       - length of ray in each refractive medium
    # z1,2       - z coordinate of ray intersection with each medium
    # tol        - object containing tolerances for solver
    #
    #Outputs:
    # r1new      - length of ray from source to tank wall
    # Iter       - number of iterations for solution
    # max_err_r1 - maximum error in solution
    
    r1new      = r1
    Iter       = 0
    max_err_r1 = 1000

    while max_err_r1 > tol:
        r1_old     = r1new
        f,df_r1    = f_eval_1eq(r1_old,r2,z1,z2,n1,n2) #evaluate snell's law equation
        r1new      = r1_old - f/df_r1
        err_r1     = abs(r1new-r1_old)
        Iter       = Iter + 1
        max_err_r1 = err_r1.max()
        
    return (r1new,Iter,max_err_r1)

def NR_2eq(r1,r2,r3,z1,z2,z3,n1,n2,n3,tol,maxIter):
    #Newton-Raphson iteration to solve the refractive imaging model for 2 equations (3 refractive media - air-tank-wall)
    #Inputs:
    # maxIter    - max # of iterations for solver
    # n1,2,3     - indices of refraction
    # r1,2,3     - length of ray in each refractive medium
    # z1,2,3     - z coordinate of ray intersection with each medium
    # tol        - max error tolerance
    #
    #Outputs:
    # r1new      - length of ray from source to tank wall
    # r2new      - length of ray in tank wall
    # Iter       - number of iterations for solution
    # max_err_r1 - maximum error in solution for r1
    # max_err_r2 - maximum error in solution for r2
    
    r1new      = r1
    r2new      = r2
    Iter       = 0
    max_err_r1 = 1000
    max_err_r2 = 1000
    
    while (max_err_r1 > tol or max_err_r2 > tol) and Iter < maxIter:
        r1_old = r1new
        r2_old = r2new
        
        #evaluate snell's law equations
        f,df_dr1,df_dr2,g,dg_dr1,dg_dr2 = f_eval_2eq(r1_old,r2_old,r3,z1,z2,z3,n1,n2,n3)

        denom = (df_dr1*dg_dr2 - df_dr2*dg_dr1)
        r1new = r1_old - (f*dg_dr2 - g*df_dr2)/denom
        r2new = r2_old - (g*df_dr1 - f*dg_dr1)/denom
        
        Iter   = Iter + 1
        err_r1 = abs(r1new - r1_old)
        err_r2 = abs(r2new - r2_old)

        max_err_r1 = err_r1.max()
        max_err_r2 = err_r2.max()
  
    return (r1new,r2new,Iter,max_err_r1,max_err_r2)

def f_refrac(r1,r2,r3,z1,z2,n1,n2):
    #model refraction with 3 refractive media (snell's law)
    #Inputs:
    # n1,2       - indices of refraction (assume one index is 1)
    # r1,2,3     - length of ray in each refractive medium
    # z1,2       - z coordinate of ray intersection with each medium
    # 
    #Outputs:
    # f          - output of snell's law equation (ideally 0)
    
    [r1,r2,r3,z1,z2,n1,n2]=np.array([r1,r2,r3,z1,z2,n1,n2])+0. #convert all values to floats
    
    f = (r2-r1) / np.sqrt((r2-r1)**2.0 + z1**2.0)   \
        - (n2/n1) * (r3-r2) / np.sqrt((r3-r2)**2.0 + z2**2.0)

    return f

def bisection(r2L,r2U,r1,r3,z1,z2,n1,n2,tol):
    # Bisection solver for 3 media refractive imaging model for 1 ray length
    #Inputs:
    # n1,2       - indices of refraction (assume one index is 1)
    # r1,2L,2U,3 - lengths of ray in each medium (2 in wall b/c for forwards and reverse ray tracing)
    # z1,2       - z coordinate of ray intersection with each medium
    # tol        - object containing tolerances for solver
    #    
    #Outputs:
    # r2f        - length of ray in second medium
    # Iter       - number of iterations to arrive at solution
    # err_r2f    - maximum error in solution
    
    Iter       = 0
    r2         = np.array([r2L,r2U]) 
    r2f        = np.zeros( len(r2[1,:]) )
    i1         = np.array( range( len(r2[1,:]) ) )
    err        = (abs(r2[0,:]-r2[1,:]))/2
    err_r2f    = np.empty( len(r2f) )
    fL         = f_refrac(r1,r2[0,:],r3,z1,z2,n1,n2)
    fU         = f_refrac(r1,r2[1,:],r3,z1,z2,n1,n2)

    while np.any(np.greater(err,tol)) and np.any(np.less_equal(fL*fU,0)):
        r2o  = r2
        Iter = Iter + 1
        
        r2=np.array([r2o[0,:] , 0.5*(r2o[0,:]+r2o[1,:])])
                
        fL         = f_refrac(r1,r2[0,:],r3,z1,z2,n1,n2) # output of snell's law function for forwards ray tracing (ideally 0)
        fU         = f_refrac(r1,r2[1,:],r3,z1,z2,n1,n2) # output of snell's law function for reverse ray tracing  (ideally 0)

        if np.any(np.less_equal(fL*fU,0)):
            ind        = np.array([x for x in range (len (fL*fU)) if fL[x]*fU[x] < 0])
            r2[:,ind]  = np.array([0.5*(r2o[0,ind]+r2o[1,ind]) , r2o[1,ind]])
            fL         = f_refrac(r1,r2[0,:],r3,z1,z2,n1,n2) # output of snell's law function for forwards ray tracing (ideally 0)
            fU         = f_refrac(r1,r2[1,:],r3,z1,z2,n1,n2) # output of snell's law function for reverse ray tracing  (ideally 0)

        err = (abs(r2[0,:] - r2[1,:])) # forwards and reverse ray tracing should yield same values             
        i2      = [x for x in range (len (err)) if (err)[x] < tol]
        i3      = [x for x in range (len (fL*fU)) if fL[x]*fU[x] > 0]
        i4      = np.union1d(i2,i3).astype(int)

        if np.any(i4):
            ind_rem          = i1[i4]
            r2f[ind_rem]     = (r2[0,i4] + r2[1,i4])/2
            err_r2f[ind_rem] = abs(r2[0,i4] - r2[1,i4])/2
            for l in [r1,r2[0],r2[1],z1,fL,fU,i1]:
                l=np.delete(l,i4)
              
    return r2f,Iter,err_r2f

def refrac_solve_bisec(r10,r20,r3,z1,z2,z3,n1,n2,n3,tol,maxIter):
    # Use repeated bisection to solve 3 media refractive imaging model for 2 ray lengths
    #Inputs:
    # maxIter    - max # of iterations for solver
    # n1,2,3     - indices of refraction
    # r10,20,3   - length of ray in each refractive medium
    # z1,2,3     - z coordinate of ray intersection with each medium
    # tol        - object containing tolerances for solver
    #    
    #Outputs:
    # r1n        - length of ray in air (wall to camera)
    # r2n        - length of ray in wall
    # Iter       - iterations to arrive at solutions
    
    r1     = r10
    r2     = r20
    r1n    = np.zeros(len(r1))
    r2n    = np.zeros(len(r2))
    err_f1_n    = np.zeros(len(r1))
    err_f2_n    = np.zeros(len(r2))
    Iter   = 0
    tol2   = 10*tol
    err_f1 = 1000*np.ones(len(r1))
    err_f2 = err_f1
    i1     = np.array(range(0,len(r1)))
    
    #Iteratively solve for the length of the ray in each medium until output from refractive equation is less than a tolerance
    while np.any(np.greater(err_f1 , tol2)) or np.any(np.greater(err_f2 , tol2)) or Iter > maxIter:
        r1o                 = r1
        rdummy              = np.zeros(len(r1))
        r1                  = bisection(r1o,r2,rdummy,r2,z1,z2,n1,n2,tol)[0] #use bisection to find the length of the ray in air
        r2o                 = r2
        r2                  = bisection(r2o,r3,r1,r3,z2,z3,n2,n3,tol)[0] #use bisection to find the length of the ray in wall

        f1,_,_,g1,_,_ = f_eval_2eq(r1,r2,r3,z1,z2,z3,n1,n2,n3) #get the output from the refractive equations to check error (f,g ideally are 0)
       
        err_f1 = np.absolute(f1)
        err_f2 = np.absolute(g1)
       
        Iter = Iter + 1
        
        i2 = [x for x in range(len(err_f1)) if err_f1[x]<tol2]
        i3 = [x for x in range(len(err_f2)) if err_f2[x]<tol2]
        i4 = np.intersect1d(i2,i3)
        if np.any(i4):                   
            ind_rem = i1[i4]
            r1n[ind_rem]  = r1[i4]
            r2n[ind_rem]  = r2[i4]
            err_f1_n[ind_rem] = err_f1[i4]
            err_f2_n[ind_rem] = err_f2[i4]
            for l in [r1,r2,r3,z1,z2,z3,i1]:
                l=np.delete(l,i4)
     
    if Iter == maxIter:
        warnings.warn('Warning: max # iterations reached in refrac_solve_bisec',stacklevel=2)
    
    return [r1n,r2n,Iter]

def img_refrac(XC,X,spData,rTol):
    # Models refractive imaging of points into camera array, using iterative solvers
    #
    # INPUTS:
    # XC            - 3 x 1 vector containing the coordinates of the camera's center of projection (COP)
    # X             - 3 x N vector containing the coordinates of each point
    # rTol          - object containing tolerances for solver 
    # spData:       - scene data 
    #        Zw     - Z coordinate of wall 
    #        n      - index of refraction of air, glass and water
    #        tW     - wall thickness
    #
    # OUTPUTS:
    # XB            - 3 x N vector containing the coordinates where the ray from each point intersects the air-facing side of the interface (wall)
    # rB            - radial distance of points in XB from the Z axis (length of ray to wall)
    # max_err_rB    - maximum error in solution of RB
    
    Npts     = X.shape[1]
    XC.shape = (3,1) #make sure this is a 3X1 array
    zW       = spData.zW
    t        = spData.tW
    n        = spData.n

    zC = XC[2]
    z1 = (zW-zC)*np.ones(Npts) #distance from camera to wall
    z2 = t*np.ones(Npts)
    z3 = (X[2,:]-(zW+t)*np.ones(Npts)).flatten() #distance from each point to the wall

    n1 = n[0]
    n2 = n[1]
    n3 = n[2]

    XB = np.zeros_like(X)
    XB[2,:]=zW

    rPorig = np.sqrt( (X[0,:]-XC[0]*np.ones(Npts))**2 + (X[1,:]-XC[1]*np.ones(Npts))**2 ).flatten() 
    rP = rPorig #distance from each point to the camera

    rB0 = (z1*rP/(X[2,:]-zC)).flatten() #length of ray from source to tank wall
    rD0 = ((z1+z2)*rP/(X[2,:]-zC)).flatten() #length of ray in tank wall

    fcheck = np.zeros(Npts)
    gcheck = fcheck
    max_err_rB=np.zeros(Npts)  
    
    
    # solve the refractve equations (snell's law) for the length of the ray in each medium
    if t==0: # no wall thickness -> no ray in wall -> only 2 media
        rB = rP
        #indices of out-of-tank and in-tank points
        i1 = np.array([x for x in range (Npts) if z3[x] ==0])
        i2 = np.array([x for x in range (Npts) if z3[x] not in i1])
        
        # use Newton-Raphson iteration to solve the refractive equation for the rays from the wall to the camera
        rB[i2] = NR_1eq(rB0[i2],rP[i2],z1[i2],z3[i2],n1,n3,rTol.tol)[0]

        if np.any(np.isnan(rB)):
            rdummy              = np.zeros(1,len(i1))
            #use bisection to solve the refractive equation for the rays from the wall to the camera
            rB[i2] = bisection(rB0[i2],rP[i2],rdummy,rP[i2],z1[i2],z3[i2],n1,n3,rTol.bi_tol)[0]

        #get the output from the refractive equation to check error (f ideally is 0)
        fcheck[i2] = f_eval_1eq(rB[i2],rP[i2],z1[i2],z3[i2],n1,n3)[0]

        if max(np.absolute(fcheck)) > rTol.fg_tol:
            warnings.warn('Warning: max values of f = ' + str(max(np.absolute(fcheck)))+'. This may be larger than it should be',stacklevel=2)
        
        if np.any(np.isnan(fcheck)):
            warnings.warn('Warning: f has a NaN',stacklevel=2)
            
    elif t > 0:
        #3 media
        rB = rP
        rD = rP

        #indices of out-of-tank and in-tank points
        i1 = np.array([x for x in range (Npts) if z3[x] < rTol.z3_tol])
        i2 = np.array([x for x in range (Npts) if z3[x] >= rTol.z3_tol])

        if not i1.size==0:
            #solve for any on-wall (2 media) points
            rdummy     = np.zeros(len(i1))
            #use bisection to solve the refractive equation for the rays from the wall to the camera
            rB[i1]     = bisection(rB0[i1],rD0[i1],rdummy,rP[i1],z1[i1],z2[i1],n1,n2,rTol.bi_tol)[0]
            #get the output from the refractive equation to check error (f ideally is 0)
            fcheck[i1] = f_eval_1eq(rB[i1],rP[i1],z1[i1],z2[i1],n1,n2)[0]


        # use Newton-Raphson iteration to solve the refractive equation for the rays from the wall to the camera and in the wall 
        rB[i2], rD[i2], Iter, max_err_rB[i2], max_err_rD = NR_2eq(rB0[i2],rD0[i2],rP[i2],z1[i2],z2[i2],z3[i2],n1,n2,n3,rTol.tol,rTol.maxiter)


        # If N-R doesn't converge => use bisection
        if np.any(np.isnan(rB)) or np.any(np.isinf(rB)):         
            
            i1 = np.array([x for x in range (Npts) if z3[x] < rTol.z3_tol])            
            if not i1.size==0:
                rdummy     =  np.zeros(len(i1))    
                
                #use bisection to solve the refractive equation for the rays from the wall to the camera
                rB[i1]     =  bisection(rB0[i1],rD0[i1],rdummy,rP[i1],z1[i1],z2[i1],n1,n2,rTol.bi_tol)[0]   
                
                #get the output from the refractive equation to check error (f ideally is 0)
                fcheck[i1] =  f_eval_1eq(rB[i1],rP[i1],z1[i1],z2[i1],n1,n2)[0]                 
           
            nan_ind  = [x for x in range (len(rB)) if math.isnan(rB[x]) or math.isinf(rB[x])]            
            #use iterative bisection to solve the 2 refractive equations for the rays from the wall to the camera and in the wall
            rB[nan_ind],rD[nan_ind],_ =  refrac_solve_bisec(rB0[nan_ind],rD0[nan_ind],rP[nan_ind],z1[nan_ind],z2[nan_ind],z3[nan_ind],n1,n2,n3,rTol.bi_tol,rTol.bi_maxiter)


        #get the output from the refractive equations to check error (f,g ideally are 0)          
        fcheck[i2],_,_,gcheck[i2],_,_ =  f_eval_2eq(rB[i2],rD[i2],rP[i2],z1[i2],z2[i2],z3[i2],n1,n2,n3)       

        if max(np.absolute(fcheck)) > rTol.fg_tol or max(np.absolute(gcheck)) > rTol.fg_tol:
            warnings.warn('Warning: max values of f = ' + str(max(np.absolute(fcheck))) + ', max values of g = ' + str(max(np.absolute(gcheck))) + '. These may be larger than they should be',stacklevel=2)
        
        if np.any(np.isnan(fcheck)) or np.any(np.isnan(gcheck)):
            warnings.warn('Warning: f or g has a NaN',stacklevel=2)
            
        
    phi     = np.arctan2((X[1,:]-XC[1,:]).flatten(),(X[0,:]-XC[0,:]).flatten())
    XB[0,:] = rB*np.cos(phi) + XC[0,:]
    XB[1,:] = rB*np.sin(phi) + XC[1,:]
    
    return (XB,rB,max_err_rB)

def P_from_params (cam_params,caData):
    # camera matrix from parameters
    #Input:
    # cam_params      - camera parameters
    # caData:         - camera data
    #   shiftx,shifty - difference between origin of image coordinates and center of image plane
    #
    #Output:
    # P               - camera matrix
    
    P = np.zeros([3,4])
    
    #world-frame camera location    
    XC = cam_params[0:3]
    
    #magnification
    a = cam_params[6]
    
    #rotation angles from world to image plane 
    alpha = cam_params[3]
    phi = cam_params[4]
    theta = cam_params[5]
    
    #intrinsic camera matrix
    K = np.zeros([3,3])
    K[0,0]=a
    K[1,1]=a
    K[2,2]=1
    K[0,2]=caData.shiftx
    K[1,2]=caData.shifty
    
    #Rotation matrix
    Rot = np.array([[math.cos(theta)*math.cos(phi) + math.sin(theta)*math.sin(alpha)*math.sin(phi), math.sin(theta)*math.cos(alpha), -math.cos(theta)*math.sin(phi) + math.sin(theta)*math.sin(alpha)*math.cos(phi)], \
                    [-math.sin(theta)*math.cos(phi) + math.cos(theta)*math.sin(alpha)*math.sin(phi), math.cos(theta)*math.cos(alpha), math.sin(theta)*math.sin(phi) + math.cos(theta)*math.sin(alpha)*math.cos(phi)], \
                    [math.cos(alpha)*math.sin(phi),-math.sin(alpha), math.cos(alpha)*math.cos(phi)]])

    P[:,0:3]=Rot
    P[:,3]=np.dot(-Rot,XC)
    P = np.dot(K,P)

    return P
    
def refrac_proj(X,P,spData,rTol, Ind =[]):
    # Given M camera pinhole matrices and the coordinates of a world point,
    # project the world point to image points in each camera.
    #
    #Inputs:
    # P           - 3x4xM matrix of pinhole camera matrices
    # X           - 3xNpts vector containing coordinates of world points
    # SpData:     - imaging system parameters 
    # rTol        - object containing tolerances for solvers
    # Ind         - indices of points on the tank wall (so no refraction)
    #
    #Outputs:
    # u           - 2 x Npts x M matrix of non-homogeneous image points

    ncams = np.shape(P)[2] 
    ind   = [i for i in range(len(X[2])) if i not in Ind] #in-tank points
    
    # Project the points into the cameras.
    u =np.empty([2,len(X[0]),ncams])
    for j in range(ncams):      
        XC = cam_decomp(P[:,:,j])  #find camera centers
        XB = np.zeros_like(X)
        XBtemp    = img_refrac(XC,X[:,ind],spData,rTol)[0]  #length of each ray from the tank wall to the camera (from refractive model)
        XB[:,ind] = XBtemp
        
        xtemp         = np.dot(P[:,:,j],np.vstack((XB,np.ones(len(XB[1])))))
        xtemp[0,:]    = xtemp[0,:]/xtemp[2,:]
        xtemp[1,:]    = xtemp[1,:]/xtemp[2,:]
    
        u[:,:,j]  = xtemp[:2,:]
        
    return u
    
def refrac_proj_onecam(cam_params,X,spData,caData,rTol):
    # This function projects the 3D world points to 2D image coordinates using 7 camera parameters
    #Inputs:
    # cam_params  - camera parameters
    # X           - 4xN matrix of world points,
    # spData:     - imaging system parameters (scene data)
    # caData      - camera data (unpacked later)
    # rTol        - object containing tolerances for solvers
    #
    #Outputs:
    # u           - 2xN matrix of image points
    
    
    P       = P_from_params (cam_params,caData) #camera matrix   
    P.shape = [3,4,1] #shape P for refrac_proj
    u       = refrac_proj(X,P,spData,rTol) [:,:,0]
 
    return u
      
def cam_model_adjust(u,par0,X,sD,cD,rTol,maxFev=1600,maxfunc_dontstop_flag=0,print_err=False,print_data=False):
    # This function finds the best-fit camera model by minimizing the
    # sum of squared differences of the (known) world points projected into
    # cameras and the measured images of these points.  The minimization is
    # done using a non-linear least squares Lev-Marq solver
    #
    #Inputs:
    # u                     - 2 x N x M matrix containing N measured 2D image plane points 
    # par0                  - 3 x 4 x M initial guess for the camera matrix (Later I will change this to be refreaction model)
    # X                     - 3 x N x M matrix containing intial guesses for the 3D world point coordinates
    # sD                    - scene data (unpacked later)
    # cD                    - camera data (unpacked later)
    # rTol                  - object containing tolerances
    # maxFev                - max number of iterations for optimizer
    # maxfunc_dontstop_flag - run until solution is found if 'on'
    # print_err             - boolean; Print pre- and post-optimization error
    # print_data            - boolean; Print initial and final camera parameters   
    #
    #Outputs:
    # Pnew                  - 3 x 4 x M best-fit camera matrix
    # params                - 7xM array of best-fit parameters

    m_eval=maxFev
    
    if len(X.shape)>2:
        p = X.shape[2]
    else:
        p = 1
    if len(u.shape)>2:
        M = u.shape[2]
    else:
        M=1

    if p < M:
        Xtmp = np.zeros([X.shape[0],X.shape[1],M])
        for j in range (0,M):
            Xtmp[:,:,j]=X
        X = Xtmp
    
    Pnew   = np.zeros([3,4,M])
    params = np.empty((7,M)) 
    for j in range (M):
        ind1  = [x for x in range(len(u[0,:,j])) if not math.isnan(u[0,x,j])]
        ind2  = [x for x in range(len(X[0,:])) if not np.any(np.isnan(X[0,x,:]))]
        ind   = np.intersect1d(ind1,ind2)
        
        Xtemp = X[:,ind,j]
        utemp = u[:,ind,j]                
        umeas = np.ravel(utemp)

        stop_flag=1
        while stop_flag:
            stop_flag = 0
            # assume refractive model
            
            #utest = refrac_proj_onecam(par0[:,j],Xtemp,sD,cD,rTol) #test for img_refrac and refrac_proj_onecam                   
            
            try:       
                # project the guessed world points through the camera model and adjust 
                # the camera parameters to fit the output to the measured image points   
                       
                bound=[] #bounds for optimization -- uncomment below to use (bounded optimization can't use the lm method)
                #bound=[[-np.inf for x in range (len(par0[:,j]))],[np.inf for x in range (len(par0[:,j]))]]
                 #set lower, upper bounds for specific parameters below:
                #[bound[0][6], bound[1][6]]= [par0[6,j],par0[6,j]*2]
                
                f = lambda x,*p: np.ravel(refrac_proj_onecam(np.array(p),x,sD,cD,rTol))
                # curve_fit needs a function that takes the independent variable as the first argument and the parameters to fit as separate remaining arguments, which the lambda function provides    
                               
                if any (bound):                    
                    params[:,j]  =   scipy.optimize.curve_fit(f, Xtemp, umeas, par0[:,j], max_nfev=m_eval,verbose=1,bounds=bound)[0]
                else:
                    params[:,j]  =   scipy.optimize.curve_fit(f, Xtemp, umeas, par0[:,j], method = 'lm', maxfev=m_eval)[0]
  
                                   
            except RuntimeError: #the minimizer hit its max eval count without converging
                if maxfunc_dontstop_flag: 
                    # double the max eval count and try again
                    m_eval=2*m_eval
                    stop_flag=1
                else:
                    raise
                    
            Pnew[:,:,j] = P_from_params(params[:,j],cD)
            
            #print in format for test
            if print_err:
                print ('\nAdjusting camera ' +str(j+1)+' parameters:')
                print ('error pre-optimization = '+str(np.mean (refrac_proj_onecam(par0[:,j],Xtemp,sD,cD,rTol)-umeas)))            
                print ('error post-optimization = '+str(np.mean (refrac_proj_onecam(params[:,j],Xtemp,sD,cD,rTol)-umeas)))
                if print_data:                
                    print '\npar0:' 
                    for l in par0[:,j]:
                        print l
                    print '\nparams:'    
                    for l in params[:,j]:
                        print l   

    return (Pnew,params)
   

def planar_grid_adj (planeParams,P,xyzgrid,spData,planeData,rTol):
    # Given M camera pinhole matrices and the coordinates of a world point,
    # project the world point to image points in each camera.
    #
    #Inputs:
    # P             - 3 x 4 x ncams camera parameter matrix
    # xyzgrid       - 3 x N matrix of points on the grid
    # rTol          - object containing tolerances  
    # spData:
    #       Zw      - Z coordinate of wall 
    #       tW      - wall thickness
    # planeData     - basic quantities associated with calibration images (unpacked later)
    #    
    #Outputs:
    # u             - 2xN matrix of non-homogeneous image points
    
    Zw, t = spData.zW, spData.tW
    
    # Calculate the world points based on the plane parameters
    X = planar_grid_to_world(planeParams,xyzgrid,planeData)
    
    Ind = [x for x in range( len(X[2]) ) if X[2,x] < (Zw+t)] #out-of-tank points
    
    # Project the points into the cameras.
    u = refrac_proj(X,P,spData,rTol,Ind)
    
    return u

def planar_grid_triang(umeas_mat,P,xyzgrid,planeParams0,spData,planeData,rTol,print_err=False,print_data=False):
    # This function finds the best-fit world point location by minimizing the
    # sum of squares of the point projected into (known) cameras.  The
    # minimization is done using a non-linear least squares Lev-Marq solver
    #
    #Inputs:
    # umeas_mat           - 2 x Npts x ncams matrix containing N measured image points in M
    #                       cameras.  If any camera doesn't see a point, NaN's should be in place of
    #                       the points.
    # P                   - 3 x 4 x ncams camera matrix for M cameras 
    # xyzgrid             - 3 x Npts matrix of points on a grid
    # plane_params0       - initial guess for plane parameters (6 rows, nPlanes columns)
    # rTol                - object containing tolerances  
    # spData              - basic quantities associated with experiment (unpacked later)
    # planeData           - basic quantities associated with calibration images (unpacked later)
    # print_err           - boolean; Print pre- and post- optimization error
    # print_data          - boolean; Print initial and final data (if print_err is also on)
    #    
    #Outputs:
    # plane_params        - optimized plane parameters
    
    [nPts,ncams]   = umeas_mat[0,:,:].shape
    
    #NaN handling - untested in Python  
    i2             =          [pt for pt in range(nPts) for cam in range(ncams) if math.isnan(umeas_mat[0,pt,cam])]   
    nan_ind1       = np.ravel([[pt+M*nPts for M in range(ncams)] for pt in range (nPts) if pt in i2 and (ncams-i2.count(pt))<3 ])
    nan_ind2       =          [i for i in range(nPts*ncams) if math.isnan(umeas_mat[0,:,:].flat[i])]
    nan_ind        = np.unique(np.append(nan_ind1,nan_ind2))

    umeas_temp       = np.reshape(umeas_mat,[2,nPts*ncams])
    umeas_temp       = np.delete(umeas_temp,nan_ind, axis=0)
    umeas_temp.shape = umeas_mat.shape
        
    #ptest= planar_grid_adj(planeParams0,P,xyzgrid,spData,planeData,rTol)
    
    ngridpts              = len(xyzgrid[0,:])
    nplanes               = len(planeParams0[0,:])
    
    # curve_fit needs a function that accepts parameters individually and 1D arrays for dependent and independent variables, so the lambda function reshapes arguments for/from planar_grid_adj
    f = lambda xyz, *params: planar_grid_adj(np.array([params[p*nplanes:(p+1)*nplanes] for p in range(6)]), P, np.array([xyz[:ngridpts], xyz[ngridpts:2*ngridpts], xyz[2*ngridpts:]]), spData, planeData, rTol).flatten()                                           
    
    # fit projected image points to the measured image points by adjusting plane parameters and projecting resulting world points to image plane
    planeParamsFlat = scipy.optimize.curve_fit(f, np.ravel(xyzgrid), np.ravel(umeas_temp), np.ravel(planeParams0))[0]    
    
    #reshape 1D output     
    planeParams = np.reshape(planeParamsFlat,np.shape(planeParams0))  
    
    #Get world points from adjusted grid parameters
    Xadj = planar_grid_to_world(planeParams,xyzgrid,planeData)
    
    #print in format for tests
    if print_err:
        print ('\nAdjusting plane parameters:')
        print ('error pre-optimization = '+str(np.mean(planar_grid_adj(planeParams0,P,xyzgrid,spData,planeData,rTol)-umeas_temp)))            
        print ('error post-optimization = '+str(np.mean (planar_grid_adj(planeParams,P,xyzgrid,spData,planeData,rTol)-umeas_temp)))   
        if print_data:        
            print '\nplaneParams0:' 
            for l in planeParams0:
                print l
            print '\nplaneParams:'    
            for l in planeParams:
                print l   
          
    return (Xadj,planeParams)


    
def reprojError (umeas,P,xPts,spData,rTol):
    # Calculate pointwise and mean reprojection errors given camera matrices and plane parameters
    #Inputs:
    # umeas         - 3xNptsxNcams array of measured image points
    # P             - 3x4xM matrix of pinhole camera matrices
    # xPts          - 3xNpts array of world points
    # spData        - basic quantities associated with experiment (unpacked later)
    # rTol          - tolerances for solvers
    #    
    #Outputs:
    # rep_err       - NptsxNcams array of reprojection error for each point 
    # rep_epp_mean  - Ncams array of mean reprojection error for each cameras
    
    uReproj      = refrac_proj(xPts,P,spData,rTol)
    # image points reprojected from world points
    
    rep_err      = np.transpose([[np.abs(pX-pU) for pX,pU in itertools.izip(mX,mU)] for mX,mU in itertools.izip(np.transpose(uReproj),np.transpose(umeas))]) 
    # reprojection error for each point in each camera    
    
    rep_err_mean = [np.mean(err) for err in np.transpose(rep_err)]
    # mean reprojection error for each camera    
    
    return (rep_err,rep_err_mean)


def selfCalibrate (umeas, pData, camData, scData, tols):
    # Carry out the autocalibration process    
    #Inputs:
    # Umeas        - 2xNxncams array of image points in each camera
    # pData:       - plane data object
    #   dX,dY      - spacing between points on the plane
    #   nX,nY      - number of points in each row/column on the plane
    #   ncalplanes - number of planes
    #   z0         - origin for position of each calibration plane in millimeters
    # camData:     - camera data object
    #   so         - distance of cameras from tank (mm)
    #   ncams      - number of cameras being calibrated
    # scData:      - scene data object 
    # tols:        - tolerances object 
    #   maxiter    - max number of iterations for optimizers
    #   rep_err_tol- acceptable refraction error
    #
    #Outputs:
    # P            - camera matrix for each camera (3 x 4 x ncams)
    # camParams    - parameters for each camera (7 x ncams)
    # Xworld       - world points
    # planeParams  - parameters for each calibration plane (6 x ncalplanes)
    
    dx         = pData.dX
    dy         = pData.dY
    nx         = pData.nX
    ny         = pData.nY
    ncalplanes = pData.ncalplanes
    z0         = pData.z0
    so         = camData.so  
    ncams      = camData.ncams
   
    # generate initial guesses for parameters 
    camParams   = setupCamera(camData)
    planeParams = setupPlanes(ncalplanes,z0)
    
    # generate locations of the points on each plane
    xvec        = np.arange(-(math.floor(nx/2)),math.floor(nx/2)+1)
    yvec        = np.arange(-(math.floor(ny/2)),math.floor(ny/2)+1)
    xphys       = dx*xvec
    yphys       = dy*yvec
    xmesh,ymesh = np.meshgrid(xphys,yphys)
    xmesh       = np.reshape(xmesh.T,(1,nx*ny))
    ymesh       = np.reshape(ymesh.T,(1,nx*ny))
    xy_phys     = np.concatenate((xmesh,ymesh),axis=0)
    xyz_phys    = np.concatenate((xy_phys,np.zeros([1,nx*ny])),axis=0)
    
    # initial grid to world points
    Xworld = planar_grid_to_world(planeParams,xyz_phys,pData) 
    
    # estimate initial camera matrix by DLT
    P = P_from_DLT(Xworld,umeas)
    
    # get camera parameters based on P estimate
    for aa in range (0,ncams):              
        C                 = cam_decomp(P[:,:,aa]) #calculate camera centers      
        camParams[0:3,aa] = C
        camParams[2,:]    = so
     
    #Calibrate 
    print '\n\nCalibrating:'
    
    rep_err_mean                 = np.empty((tols.maxiter+1,ncams))
    rep_err_diff                 = -1*float('Inf')
    #calculate initial reprojection error (rep_err -> reprojection error)
    rep_err,rep_err_mean[0]      = reprojError(umeas,P,Xworld,scData,tols) 
    

    # AUTOCALIBRATION LOOP 
    Iter= 0 
    while np.any(rep_err_mean[Iter] > tols.rep_err_tol*np.ones(ncams)) and Iter < tols.maxiter and rep_err_diff < -1*tols.rep_err_tol/10 :
         print ('\n\nIteration ' + str(Iter+1)+'\nInitial mean reprojection error in each camera= ' +str(rep_err_mean[Iter]))
            
         #optimize camera and plane parameters in sequence 
         P,camParams          = cam_model_adjust(umeas,camParams, Xworld, scData, camData, tols)
         Xworld,planeParams   = planar_grid_triang(umeas, P, xyz_phys, planeParams, scData, pData, tols)
         
         Iter=Iter+1
         
         #recalculate reprojection error
         rep_err,rep_err_mean[Iter]   = reprojError(umeas,P,Xworld,scData,tols)
         rep_err_diff                 = np.mean(rep_err_mean[Iter]-rep_err_mean[Iter-1])
         
         print ('\n Change in mean reprojection error: '+str(rep_err_diff))
         
         if rep_err_diff > 0: # mean reprojection error should reaally be decreasing each iteration
             warnings.warn('\nReprojection error increasing in iteration ' +str(Iter),stacklevel=3)

    
    print ('\n\n\nCalibration Complete!')
    if not np.any(rep_err_mean[Iter] > tols.rep_err_tol*np.ones(ncams)):
        print ('Reprojection error less than threshold  in all cameras.')
    if Iter >= tols.maxiter:
        print ('Max iterations reached.')
    if not rep_err_diff < -1*tols.rep_err_tol/10:
        print ('Change in mean reprojection error less than threshold.')
    print ('Results: \n Final mean reprojection error in each camera: ' +str(rep_err_mean[Iter]) )  
    
    return P,camParams,Xworld,planeParams,rep_err



############################# Save and display data functions #################################


def showCalibData(ccoords, X, zW): 
    #display calibration results
    #Inputs:
    # ccoords  - xoordinates of camera centers
    # X        - world points
    # zW       - z coordinate of wall
    #Outputs:
    # plots world points and camera locations in 3D, with a sample of the wall
    
    #plot world points and camera locations 
    fig = plt.figure('Camera and Grid Locations',figsize=(8,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.axis('equal')
    #include a sample of the tank wall for context
    wallpts = np.array([[x,y,zW] for y in range (int(min(ccoords[1,:])-100), int(max(ccoords[1,:])+100), 12) for x in range (int(min(ccoords[0,:])-50), int(max(ccoords[0,:])+50), 12)])   
    wall, = ax.plot (wallpts[:,0], wallpts[:,1], wallpts[:,2], 'g.', label='Tank Wall')    
    grid, = ax.plot (X[0,:], X[1,:], X[2,:], 'b.', label='Grid Points') 
    cams, = ax.plot (ccoords[0,:], ccoords[1,:], ccoords[2,:], 'r+', label='Cameras')    
    plt.legend(handles=[grid,cams,wall])
    plt.show()

def saveCalibData(exptpath, camnames, p, cparams, err, scdata, camdata, name):
    #Save calibration data
    #Inputs:
    # exptpath - path on which data should be saved
    # camnames - names of cameras to which data belongs
    # p        - 3 x 4 x number of cameras array of camera matrices 
    # cparams  - parameters from which the matrices were contructed
    # err      - reprojection error in each camera
    # scdata:  - scene data object
    #   zW     - z coordinate of the wall
    #   tW     - wall thickness
    #   n[]    - indices of refraction
    # camdata: - camera data object
    #   sX,sY  - size of image
    #   mpix   - 1/size of a pixel
    # name     - name of file to save data as
    #
    #Outputs:
    # saves data on the experiment path
    # f - file to which data was saved (closed)
    
    
    #save data on the experiment path
    with open(os.path.join(exptpath, name+'.txt'),'w') as f:        
        # Printing dummy timestamp
        f.write('---Dummy Timestamp---\n')
        
        # Average reprojection error
        f.write('{}\n'.format(np.mean(err)))
        
        # Image size and physical to pixel conversion factor
        f.write('{:d}\t{:d}\t{}\n'.format(int(camdata.sX), int(camdata.sY), camdata.pix_phys))
        
        # P matrices
        sizeP = np.shape(p)
        f.write('{:d}\n'.format(int(sizeP[2])))
        for n in range(sizeP[2]):
            
            f.write('{}\n'.format(camnames[n]))
            
            for i in range(3):
                for j in range (4):
                    f.write('{0:.15f}\t'.format(p[i,j,n]))
                f.write('\n')
            
            for i in range(3):
                f.write('{:.15f}\t'.format(cparams[i][n]))
            f.write('\n')
        
        # Refractive flag
        f.write('{}\n'.format('1'))

        # Refractive geometry paramters
        f.write('{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(scdata.zW, scdata.tW, scdata.n[0], scdata.n[1], scdata.n[2]))
        
        #f.write('Camera Matrices\n')
        #for c in range(len(camnames)):
        #    f.write(camnames[c]+'\n')
        #    np.savetxt(f,p[:,:,c],delimiter=', ',fmt='%f')            
        #f.write('\n\nCamera Parameters\n')
        #for c in range(len(camnames)):
        #    f.write(camnames[c]+'\n')
        #    np.savetxt(f,cparams[:,c],delimiter=', ',fmt='%f')
        #f.write('\n\nWorld Points\n')
        #np.savetxt(f,X,delimiter=', ',fmt='%f')
        
        return f  # file will close when with statement terminates



####################################### General functions ##############################################################


def setupObjects(dx,dy,nx,ny,ncalplanes,znet, sx,sy,pix_pitch,so,f,ncams,nframes, n1,n2,n3,tw,zw, tol,fg_tol,maxiter,bi_tol,bi_maxiter,z3_tol,rep_err_tol,):
    # setup experimental parameter storage objects
    #
    #Inputs:
    # n1,2,3      - indices of refraction of 3 media in system    
    # dx,dy       - spacing of grid points on calibration grid    
    # nx,ny       - number of grid points in each row, column on grid
    # pix_pitch   - pixel pitch (size of a pixel)    
    # so          - distance of cameras from tank    
    # f           - focal length of the lens    
    # tW          - tank wall thickness (2nd refractive medium)
    # sx,sy       - dimensions of images being calibrated
    # zw          - z coordinate of wall in coordinate system being used  
    # ncalplanes  - number of planes 
    # znet        - total traverse of calibration plate in millimeters (used to calculate origin of ncalplanes evenly spaced planes)
    # ncams       - number of cameras
    # nframes     - number of frames in calibration video/tiff -- optional, but preferrable to provide
    # tol         - tolerance for Newton-Raphson snell's law solver 
    # fg_tol      - tolerance for output of snell's law function
    # maxiter     - max number of iterations for solvers
    # bi_tol      - tolerance for bifurcation snell's law solver
    # bi_maxiter  - max number of iterations for bifurcation snell's law solver
    # z3_tol      - tolerance for distance of grid points from tank wall
    # rep_Err_Tol - tolerance for final reprojection error 
    #
    #Outputs:
    # planedata   - object containing parameters of the calibration planes/grids
    # cameradata  - object containing parameters of the calibration images
    # scenedata   - object containing parameters of the experimental setting
    # tolerances  - object containing error tolerances for solvers
    
    planedata   = planeData(dx,dy,nx,ny,ncalplanes,znet)
    cameradata  = cameraData(sx,sy,pix_pitch,so,f,ncams,nframes)
    scenedata   = sceneData(n1,n2,n3,tw,zw)
    tolerances  = refracTol(tol,fg_tol,maxiter,bi_tol,bi_maxiter,z3_tol,rep_err_tol) 
    
    return planedata, cameradata, scenedata, tolerances
    
    
def CalibrationTiff(dx,dy,nx,ny,ncalplanes,znet, sx,sy,pix_pitch,so,f,nframes, n1,n2,n3,tw,zw, tol,fg_tol,maxiter,bi_tol,bi_maxiter,z3_tol,rep_err_tol, datapath, exptpath, camids) :
    # Carry out the refractive autocalibration process from beginning to end, using multipage tiffs 
    #
    #Inputs:
    # datapath    - Path to stored images
    # exptpath    - Path for saved output
    # camids      - Names of cameras being calibrated    
    # n1,2,3      - indices of refraction of 3 media in system    
    # dx,dy       - spacing of grid points on calibration grid    
    # nx,ny       - number of grid points in each row, column on grid
    # pix_pitch   - pixel pitch (size of a pixel)    
    # so          - distance of cameras from tank    
    # f           - focal length of the lens    
    # tW          - tank wall thickness (2nd refractive medium)
    # sx,sy       - dimensions of images being calibrated
    # zw          - z coordinate of wall in coordinate system being used    
    # znet        - total traverse of calibration plate in millimeters 
    # ncalplanes  - number of planes 
    # nframes     - number of frames in calibration video/tiff -- optional, but preferrable to provide
    # tol         - tolerance for Newton-Raphson snell's law solver 
    # fg_tol      - tolerance for output of snell's law function
    # maxiter     - max number of iterations for solvers
    # bi_tol      - tolerance for bifurcation snell's law solver
    # bi_maxiter  - max number of iterations for bifurcation snell's law solver
    # z3_tol      - tolerance for distance of grid points from tank wall
    # rep_Err_Tol - tolerance for final reprojection error 
    #
    #Output:
    # plotted world points and camera locations, with a sample of the wall  
    # file containing camera parameters, camera matrices, world points and plane parameters, saved on exptpath 
    
    ncams = len(camids) #get number of cameras
    
    # setup experimental parameter storage objects
    planedata, cameradata, scenedata, tolerances  = setupObjects(dx,dy,nx,ny,ncalplanes,znet, sx,sy,pix_pitch,so,f,ncams,nframes, n1,n2,n3,tw,zw, tol,fg_tol,maxiter,bi_tol,bi_maxiter,z3_tol,rep_err_tol,)

    #find images for calibration in multipage tiff files
    calimages = getCalibImagesTiff(datapath, camids, planedata.ncalplanes, cameradata.nframes)
    
    # call to corner finder to get 2D image plane points
    Umeas = findCorners(planedata, camids, imgs = calimages)
      
    # find correct camera and plane parameters
    p,camparams,xworld,planeparams,reperror = selfCalibrate(Umeas, planedata, cameradata, scenedata, tolerances)
    
    # display data
    showCalibData(camparams[:3], xworld, scenedata.zW)
    
    #get the scale of the images
    cameradata.pix_phys = getScale(Umeas, planedata.nX, planedata.nY, planedata.dX, int(planedata.ncalplanes/2), int(ncams/2)) 
     
    # save data
    print ('\n\nData saved in '+ str(saveCalibData(exptpath, camids, p, camparams, reperror, scenedata, cameradata, 'Calibration_Results')))





    
    
