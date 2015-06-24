#from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import itertools
import pylab
from scipy import spatial
from matplotlib.path import Path

# user defined options
disk = False                # this parameter defines if we look for Poisson-like distribution on a disk (center at 0, radius 1) or in a square (0-1 on x and y)
squareRepeatPattern = False  # this parameter defines if we look for "repeating" pattern so if we should maximize distances also with pattern repetitions
#num_points = 32            # number of points we are looking for
num_iterations = 10         # number of iterations in which we take average minimum squared distances between points and try to maximize them
first_point_zero = disk     # should be first point zero (useful if we already have such sample) or random
iterations_per_point = 30   # iterations per point trying to look for a new point with larger distance
sorting_buckets = 0         # if this option is > 0, then sequence will be optimized for tiled cache locality in n x n tiles (x followed by y)

def random_point_disk():
    alpha = random.random() * math.pi * 2.0
    radius = math.sqrt(random.random())
    x = math.cos(alpha) * radius
    y = math.sin(alpha) * radius
    return np.array([x,y])

def random_point_square():
    x = random.uniform(0,1)
    y = random.uniform(0,1)
    return np.array([x,y])

def first_point():
    if first_point_zero == True:
        return np.array([0,0])
    elif disk == True:
        return random_point_disk()
    else:
        return random_point_square()

# if we only compare it doesn't matter if it's squared
def min_dist_squared_pure(points, point):
    diff = points - np.array([point])
    return np.min(np.einsum('ij,ij->i',diff,diff))

def min_dist_squared_repeat(points, point):
    dist = math.sqrt(2)
    for y in range(-1,2):
        for x in range(-1,2):
            testing_point = np.array([point-[x,y]])
            diff = points-testing_point
            dist = min(np.min(np.einsum('ij,ij->i',diff,diff)),dist)
    return dist

def find_next_point(current_points):
    best_dist = 0
    best_point = []
    for i in range(iterations_per_point):
        new_point = random_point()
        dist = min_dist_squared(current_points, new_point)
        if (dist > best_dist) and (dist > 0.00):
            best_dist = dist
            best_point = new_point
    return best_point

def find_next_point_2(current_points):
    best_dist = 0
    best_point = []
    for i in range(iterations_per_point):
        new_point = getRandomPoint()
        dist = min_dist_squared(current_points, new_point)
        if (dist > best_dist) and (dist > 0.00):
            best_dist = dist
            best_point = new_point
    return best_point

def getRandomPoint():
    x = random.uniform(0,.9)
    y = random.uniform(0,.9)
    return np.array([x,y])

def find_point_set(num_points, num_iter):
    best_point_set = []
    best_dist_avg = num_points*math.sqrt(2.0)
    for i in range(num_iter):
        points = np.array([first_point()])
        for i in range(num_points-1):
            points = np.append(points, np.array(find_next_point(points),ndmin = 2), axis = 0)
        current_set_dist = 0
        for i in range(num_points):
            dist = min_dist_squared(np.delete(points,i,0), points[i])
            current_set_dist += dist
        if current_set_dist < best_dist_avg:
            best_dist_avg = current_set_dist
            best_point_set = points
    return best_point_set

if disk == True:
    random_point = random_point_disk
else:
    random_point = random_point_square

if disk == False and squareRepeatPattern == True:
    min_dist_squared = min_dist_squared_repeat
else:
    min_dist_squared = min_dist_squared_pure

def generatePoissonDisk(num_points):    
    points = find_point_set(num_points,num_iterations)
    
    if sorting_buckets > 0:
        points_discretized = np.floor(points * [sorting_buckets,-sorting_buckets])
        # we multiply in following line by 2 because of -1,1 potential range
        indices_cache_space = np.array(points_discretized[:,1] * sorting_buckets * 2 + points_discretized[:,0])
        points = points[np.argsort(indices_cache_space)]
    
    return points

def generatePDD(num_points, iterations, min_dist,limUp=1,limBot=0,inputPoints=np.array([]), virtualPoints=np.array([])):
    nPoints = inputPoints.size / 2
    vSize = virtualPoints.size / 2
    points = np.zeros((num_points+vSize,2))
    b = .5
    s = int(limUp / (min_dist*b))
    candidatePoints = np.random.uniform(limBot,limUp,(s,s,iterations,2))
    if vSize > 0:
        points[0:vSize,:] = virtualPoints
    if nPoints == 0:
        points[vSize,:] = np.random.uniform(limBot,limUp,(1,2))
        nPoints += 1
    else:
        points[vSize:vSize+nPoints,:] = inputPoints
    
    scale = (limUp - limBot) / float(limUp)
    #andidatePoints = np.random.uniform(limBot,limUp,(s*s*iterations,2))
    xs = np.arange(0,s,dtype=int)
    ys = np.arange(0,s,dtype=int)
    np.random.shuffle(xs)
    np.random.shuffle(ys)
    candidatePoints *= min_dist * b
    for i in range(s):
        i = xs[i]
        for j in range(s):
            j = ys[j]
            candidatePoints[i,j,:,0] += scale * i * min_dist*b
            candidatePoints[i,j,:,1] += scale * j * min_dist*b
            dist = spatial.distance.cdist(candidatePoints[i,j,:,:], points[0:nPoints+vSize])
            d = np.min(dist,axis=1)
            k = np.argmax(d)
            if d[k] > min_dist:
                points[vSize+nPoints,:] = candidatePoints[i,j,k,:]
                nPoints += 1
    
    return points[vSize:,:]

def generatePDDLevels(num_points, iterations, min_dist,limUp,limBot,inputPoints, virtualPoints,inputLevels,virtualLevels):
    nPoints = inputPoints.size / 2
    vSize = virtualPoints.size / 2
    levels = virtualLevels.shape[0]
    points = np.zeros((num_points+vSize,2))
    b = .5
    if vSize > 0:
        points[0:vSize,:] = virtualPoints
    if nPoints == 0:
        points[vSize,:] = np.random.uniform(limBot,limUp,(1,2))
        nPoints += 1
    else:
        points[vSize:vSize+nPoints,:] = inputPoints
    
    limUpNew = limUp - min_dist[0]
    rest= np.zeros((levels,num_points-nPoints),dtype=bool)
    maskLevels = np.concatenate((virtualLevels,inputLevels,rest),axis=1)
    for l in range(0,levels):
        s = int(limUp / (min_dist[l]*b))
        mask = np.sum(maskLevels[0:l+1,:],axis=0) == 1
        candidatePoints = np.random.uniform(limBot,limUpNew,(s,s,iterations,2))
        xs = np.arange(0,s,dtype=int)
        ys = np.arange(0,s,dtype=int)
        np.random.shuffle(xs)
        np.random.shuffle(ys)
        scale = (limUpNew - limBot) / float(limUp)
        candidatePoints *= min_dist[l] * b
        for i in range(s):
            i = xs[i]
            for j in range(s):
                j = ys[j]
                candidatePoints[i,j,:,0] += scale * i * min_dist[l]*b
                candidatePoints[i,j,:,1] += scale * j * min_dist[l]*b
                dist = spatial.distance.cdist(candidatePoints[i,j,:,:], points[0:nPoints+vSize])
                
                score = 0
                m = np.logical_or(mask,maskLevels[l,:])
                for l2 in range(l,levels):
                    if l2 == l:
                        distLevel = dist[:,m[0:nPoints+vSize]]
                    else:
                        distLevel = dist[:,maskLevels[l2,0:nPoints+vSize]]
                    d = np.min(distLevel,axis=1)
                    score += d > min_dist[l2]
                if np.any(score == levels - l):
                    k = np.where(score==levels-l)[0][0] 
#                     print points[vSize+nPoints,:]       
                    points[vSize+nPoints,:] = candidatePoints[i,j,k,:]
#                     print maskLevels[:, vSize+nPoints], l
                    maskLevels[l, vSize+nPoints] = True 
                    nPoints += 1
    
    pointLevels = np.zeros((points[vSize:vSize+nPoints,:].shape[0]))
#     print maskLevels
#     print maskLevels[0,:]
#     print maskLevels[1,:]
#     print np.sum(maskLevels,axis=0)
    for l in range(0,levels):
#         print maskLevels[l,vSize:vSize+nPoints]
        pointLevels[maskLevels[l,vSize:vSize+nPoints]] = l 
    return points[vSize:vSize+nPoints,:],pointLevels

        
def generatePDDwithPoints(num_points, maskPoints, existingPoints):
    s1 = maskPoints.shape[0]
    s2 = existingPoints.shape[0]
    s = s1+s2
    totPoints = np.empty((s,2))
    totPoints[0:s1,:] = maskPoints
    totPoints[s1:s,:] = existingPoints
    #for i in range(num_iter):
    points = totPoints
    for i in range(num_points-s2):
        points = np.append(points, np.array(find_next_point_2(points),ndmin = 2), axis = 0)
    #current_set_dist = 0
    points = points[s1::,:]
    return points

def generatePDD_new(gridResolution, min_dist_array,level,mask,finalMask,minBoundX = 0.0, minBoundY = 0.0, maxBoundX = 1.0, maxBoundY = 1.0, existingPoints=np.array([]),itr=20):
    min_dist = min_dist_array[level]
    totLevels = min_dist_array.shape[0]
    b = min_dist * .5
    nGroups = int(np.ceil(1.0 / (b)))
    nPoints = (nGroups+1)*(nGroups+1) + existingPoints.shape[0]
    points = np.zeros((nPoints,3))
    currentNPoints = 0
        
    if existingPoints.shape[0] == 0:
        rPoint =  np.random.uniform(0,1,(1,2))
        points[currentNPoints,0] = rPoint[0,0] * (maxBoundX - minBoundX) + minBoundX
        points[currentNPoints,1] = rPoint[0,1] * (maxBoundY - minBoundY) + minBoundY
        points[currentNPoints,2] = level
        currentNPoints += 1
    else:
        points[0:existingPoints.shape[0],:] = existingPoints
        currentNPoints += existingPoints.shape[0]
    
    totalItr = nGroups*nGroups
    x,y = np.indices((nGroups,nGroups)) * (1/nGroups)
    x = x.ravel()
    y = y.ravel()
    randomPointsGrid = np.random.uniform(0,1,(totalItr,totalItr,2))
#     randomPointsGrid /= float(nGroups)
#     randomPointsGrid[:,:,0] += x[np.newaxis,:]
#     randomPointsGrid[:,:,1] += y[np.newaxis,:]
    randomPointsGrid[:,:,0] *= (maxBoundX - minBoundX)
    randomPointsGrid[:,:,1] *= (maxBoundY - minBoundY)
    randomPointsGrid[:,:,0] += minBoundX
    randomPointsGrid[:,:,1] += minBoundY
    #randomPoints = randomPointsGrid.reshape(nGroups*nGroups*itr,2)
#    print randomPoints.shape
    #allDistances = spatial.distance.cdist(randomPoints,np.concatenate((randomPoints,points[0:currentNPoints,0:2]),axis=0))
#     plt.figure()
#     plt.scatter(randomPointsGrid[:,:,:,0].ravel(),randomPointsGrid[:,:,:,1].ravel())
#     plt.show()
    t1 = mask.contains_points(randomPointsGrid.reshape(totalItr*totalItr,2))
    withInBounds = t1.reshape(totalItr,totalItr).astype(int)
    for i in xrange(0,totalItr):
        dist = spatial.distance.cdist(randomPointsGrid[i,:,:], points[0:currentNPoints,0:2])
        distances = 0
        for l in range(level,totLevels):
            mask = points[0:currentNPoints,2] <= l
            min_dist = min_dist_array[l]
            distL = dist[:,mask]
            d = np.min(distL,axis=1)
            distances += d
            score = withInBounds[i,:]
            score += (d > min_dist).astype(int)
        inds = np.where(score == (1 + totLevels - level))[0]
        if  score[inds].size > 0:
            minDistID = np.argmin(distances[inds])
            k = inds[minDistID]
            points[currentNPoints,0:2] = randomPointsGrid[i,k]
            points[currentNPoints,2] = level
            currentNPoints += 1
    m = finalMask.contains_points(points[:currentNPoints,0:2])
    return points[m,:]    
