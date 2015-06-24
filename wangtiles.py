from poisson import *
import matplotlib.pyplot as plt
import numpy
from scipy import spatial
from matplotlib.path import Path

def getColor(typeList):
    defColors = numpy.array([[0,0,0],[0,1,0],[1,1,0],[0,1,1],[.4,.0,.4],[1,0,0],[0,0,1],[1,1,1]])
    
    return defColors[typeList]
# x and y coords
def nozero(arr2D):
    return arr2D[numpy.all(arr2D[:,0:2] > 0,axis=1)]

def nozero3D(arr3D):
    return arr3D[numpy.all(arr3D[:,:,0:2] > 0,axis=2)]

def generatePoints_new(gridX,gridY,totalTiles,x,y):
    levels = 1
    dist=numpy.array([.34])
    dist2=numpy.array([.33,.2,.1])
    gridRes = numpy.array([.25,.25,.25])
    b = dist[levels-1] * gridRes[levels-1]
    
    gr = int(np.ceil(1.0 / (b))) + 1
    totPoints = gr * gr
    
    
    borderH = numpy.array([0,1])
    borderV = numpy.array([0,1])
    
    posTiles = borderH.size * borderH.size * borderV.size * borderV.size
    
    # Left, right, bottom, top
    print "Generate Wang Tiling"
    wangTiles = numpy.zeros((posTiles,4),dtype=int)
    cornerTiles = numpy.zeros((borderH.size,borderH.size,borderV.size,borderV.size),dtype=int)
    index = 0
    for i in range(0,borderV.size):
        for j in range(0,borderV.size):
            for m in range(0,borderH.size):
                for n in range(0,borderH.size):
                    wangTiles[index,:] = [borderV[i],borderV[j],borderH[m],borderH[n]]
                    cornerTiles[borderH[m],borderH[n],borderV[j],borderV[i]] = index
                    index += 1
    
    sH = borderH.size * borderV.size * borderV.size
    sV = borderH.size * borderH.size * borderV.size
    sD = borderH.size * borderV.size
    combinationsSH = numpy.zeros((borderH.size,sH))
    combinationsSV = numpy.zeros((borderV.size,sV))
    combinationsD = numpy.zeros((borderH.size,borderV.size,sD))
    
    for i in range(0,borderH.size):
        b = borderH[i]
        m = wangTiles[:,2] == b
        combinationsSH[i,:] = numpy.where(m)[0]+1
    
    for i in range(0,borderV.size):
        b = borderV[i]
        m = wangTiles[:,0] == b
        combinationsSV[i,:] = numpy.where(m)[0]+1
    
    for i in range(0,borderH.size):
        for j in range(0,borderV.size):
            combinationsD[i,j,:] = numpy.intersect1d(combinationsSH[i],combinationsSV[j])    
        
    wangTiling = numpy.zeros((gridX+2,gridY+2),dtype=int)
    rnTiles = numpy.random.randint(1,posTiles+1,(totalTiles))
    ordering = numpy.zeros((totalTiles))
    cornerTilingMask = numpy.zeros((gridX+1,gridY+1),dtype=bool)

    for k in xrange(0,totalTiles):
        i = x[k] + 1
        j = y[k] + 1
        if i == 0 and j == 0:
            wangTiling[i,j] = rnTiles[k]
        elif i == 0:
            botTileID = wangTiling[i,j-1]
            if botTileID == 0:
                wangTiling[i,j] = rnTiles[k]
            else:
                b = wangTiles[botTileID-1,3]
                wangTiling[i,j] = combinationsSH[b,rnTiles[k] % sH]
        elif j == 0:
            leftTileID = wangTiling[x[i]-1,y[i]]
            if leftTileID == 0:
                wangTiling[i,j] = rnTiles[k]
            else:
                b = wangTiles[leftTileID-1,1]
                wangTiling[i,j] = combinationsSV[b,rnTiles[k] % sV]
        else:
            botTileID = wangTiling[i,j-1]
            leftTileID = wangTiling[i-1,j]
            if botTileID == 0 and leftTileID == 0:
                wangTiling[i,j] = rnTiles[k]
            elif botTileID == 0:
                b = wangTiles[leftTileID-1,1]
                wangTiling[i,j] = combinationsSV[b,rnTiles[k] % sV]
            elif leftTileID == 0:
                b = wangTiles[botTileID-1,3]
                wangTiling[i,j] = combinationsSH[b,rnTiles[k] % sH]
            else:
                bH = wangTiles[botTileID-1,3]
                bV = wangTiles[leftTileID-1,1]
                wangTiling[i,j] = combinationsD[bH,bV,rnTiles[k] % sD]
        cornerTilingMask[i-1:i+2-1,j-1:j+2-1] = True
        ordering[k] = wangTiling[i,j]

    xIndices,yIndices = np.indices((gridX+1,gridY+1))
 
    xC = xIndices[cornerTilingMask]
    yC = yIndices[cornerTilingMask]
        
    cornerTiling = numpy.zeros((gridX+1,gridY+1),dtype=int)
    rnCornersH = numpy.random.randint(0,borderH.size,(gridX+1,gridY+1,2))
    rnCornersV = numpy.random.randint(0,borderV.size,(gridX+1,gridY+1,2))
    for k in xrange(0,xC.size):
        i = xC[k]
        j = yC[k]
        
        rtTile = wangTiling[i+1,j+1]-1
        rbTile = wangTiling[i+1,j]-1
        ltTile = wangTiling[i,j+1]-1
        lbTile = wangTiling[i,j]-1
        
        tb = rnCornersV[i,j,0]
        bb = rnCornersV[i,j,1]
        lb = rnCornersH[i,j,0]
        rb = rnCornersH[i,j,1]
        
        if not(rtTile == -1):
            tb = wangTiles[rtTile,0]
            rb = wangTiles[rtTile,2]
        if not(rbTile == -1):
            bb = wangTiles[rbTile,0]
            rb = wangTiles[rbTile,3]
        if not(ltTile == -1):
            tb = wangTiles[ltTile,1]
            lb = wangTiles[ltTile,2]
        if not(lbTile == -1):
            bb = wangTiles[lbTile,1]
            lb = wangTiles[lbTile,3]
        
        cornerTiling[i,j] = cornerTiles[lb,rb,bb,tb]
    
    wangTilesPoints = numpy.zeros((posTiles,totPoints,3))
    
    borderPointsH = numpy.zeros((borderH.size,totPoints,3))
    borderPointsV = numpy.zeros((borderV.size,totPoints,3))
    cornerPoints = numpy.zeros((cornerTiles.size,totPoints / 4,3))
    
    print "Generate border points"
    d = dist[0]
    borderVMaskFinal = Path(np.array([(.5 - d*.5,d),(.5 + d*.5,d),(.5 + d*.5,1-d),(.5 - d*.5,1-d)]))
    d = dist[0] * 1
    borderVMask = Path(np.array([(.5 - d*.5,d),(.5 + d*.5,d),(.5 + d*.5,1-d),(.5 - d*.5,1-d)]))
    for i in range(0,borderV.size):
        existingPoints = numpy.array([],dtype=float).reshape(0,3)
        for l in range(0,levels):
            points = generatePDD_new(gridRes[l],dist,l,borderVMask,borderVMaskFinal,minBoundX=(.5 - d*.5),minBoundY=d,maxBoundX=(.5 + d*.5),maxBoundY=(1 - d),existingPoints=existingPoints)
#             border = numpy.logical_and(numpy.logical_and(p[:,0] > (.5 - d*.5), p[:,0] < (.5 + d*.5)), numpy.logical_and(p[:,1] > d, p[:,1] < (1 - d)))
#             points = p[border,:]
            existingPoints = numpy.concatenate((existingPoints,points),axis=0)
        borderPointsV[i,0:points.shape[0]] = points
#         print points
#         plt.scatter(points[:,0],points[:,1])
#         plt.show()
     
    d = dist[0]
    borderHMask = Path(np.array([(d,.5 + d*.5),(d,.5 - d*.5),(1-d,.5 - d*.5),(1-d,.5 + d*.5)]))
    for i in range(0,borderH.size):
        existingPoints = numpy.array([],dtype=float).reshape(0,3)
        for l in range(0,levels):
            points = generatePDD_new(gridRes[l],dist,l,borderHMask,borderHMask,minBoundX=d,minBoundY=(.5 - d*.5),maxBoundX=(1 - d),maxBoundY=(.5 + d*.5),existingPoints=existingPoints)
            #border = numpy.logical_and(numpy.logical_and(p[:,1] > (.5 - d*.5), p[:,1] < (.5 + d*.5)), numpy.logical_and(p[:,0] > d, p[:,0] < (1 - d)))
            #points = p[border,:]
            existingPoints = numpy.concatenate((existingPoints,points),axis=0)
        borderPointsH[i,0:points.shape[0]] = points
#         print points
#         plt.scatter(points[:,0],points[:,1])
#         plt.show()
    
#     cornerMask = Polygon([(.5-d,.5-d*2),(.5+d,.5-d*2),(.5+d*2,.5-d),(.5+d*2,.5+d),(.5+d,.5+d*2),(.5-d,.5+d*2),(.5-d*2,.5+d),(.5-d*2,.5-d)])
#     t = gp.GeoSeries(cornerMask)
#     pt = Point([(1,1)])
#     t2 = gp.GeoSeries(pt)
#     print t2.values.coords
#     t.plot()
#     plt.show()
    print "Generate corner points"
    index = 0
    d = dist[0] * .5
    cornerMask2 = Path(np.array([(.5-d,.5-d*2),(.5+d,.5-d*2),(.5+d,.5-d),(.5+d*2,.5-d),(.5+d*2,.5+d),(.5+d,.5+d),(.5+d,.5+d*2),(.5-d,.5+d*2),(.5-d,.5+d),(.5-d*2,.5+d),(.5-d*2,.5-d),(.5-d,.5-d)]))
    d = dist[0] * .5
    cornerMask = Path(np.array([(.5-d,.5-d*2),(.5+d,.5-d*2),(.5+d*2,.5-d),(.5+d*2,.5+d),(.5+d,.5+d*2),(.5-d,.5+d*2),(.5-d*2,.5+d),(.5-d*2,.5-d)]))
    d = dist[0] * .5
    cornerMaskFinal = Path(np.array([(.5-d,.5-d*2),(.5+d,.5-d*2),(.5+d*2,.5-d),(.5+d*2,.5+d),(.5+d,.5+d*2),(.5-d,.5+d*2),(.5-d*2,.5+d),(.5-d*2,.5-d)]))
    for i in range(0,borderV.size):
        tb = nozero(borderPointsV[i,:].copy())
        tb[:,1] += .5
        for j in range(0,borderV.size):
            bb = nozero(borderPointsV[j,:].copy())
            bb[:,1] -= .5
            for m in range(0,borderH.size):
                lb = nozero(borderPointsH[m,:].copy())
                lb[:,0] -= .5
                for n in range(0,borderH.size):
                    rb = nozero(borderPointsH[n,:].copy())
                    rb[:,0] += .5
                    borderPoints = numpy.concatenate((tb,bb,lb,rb),axis=0)
                    existingPoints = borderPoints
                    for l in range(0,levels):
                        points = generatePDD_new(gridRes[l],dist,l,cornerMask,cornerMaskFinal,minBoundX=.5-d*2,minBoundY=.5-d*2,maxBoundX=.5+d*2,maxBoundY=.5+d*2,existingPoints=existingPoints)
                        existingPoints = numpy.concatenate((borderPoints,points),axis=0)
                    index = cornerTiles[borderH[m],borderH[n],borderV[j],borderV[i]]
                    cornerPoints[index,0:points.shape[0]] = points
#                     print points
                    #print nozero(cornerPoints[index,0:points.shape[0]])
#                     dist1 = spatial.distance.cdist(points, existingPoints)
#                     print numpy.all(dist1 > .3,axis=1)
#                     plt.figure()
#                     plt.scatter(existingPoints[:,0],existingPoints[:,1],c='b')
#                     plt.scatter(points[:,0],points[:,1],c='r')
#                     plt.show()
    
    d = dist[0] * .5
    tileMask = Path(np.array([(d*2,d),(1-d*2,d),(1-d,d*2),(1-d,1-d*2),(1-d*2,1-d),(d*2,1-d),(d,1-d*2),(d,d*2)]))
    tileMaskFinal = Path(np.array([(d*2,d),(d*2,0),(1-d*2,0),(1-d*2,d),(1-d,d*2),(1,d*2),(1,1-d*2),(1-d,1-d*2),(1-d*2,1-d),(1-d*2,1),(d*2,1),(d*2,1-d),(d,1-d*2),(0,1-d*2),(0,d*2),(d,d*2)]))
#     p1 = Polygon([(d*2,d),(d*2,0),(1-d*2,0),(1-d*2,d),(1-d,d*2),(1,d*2),(1,1-d*2),(1-d,1-d*2),(1-d*2,1-d),(1-d*2,1),(d*2,1),(d*2,1-d),(d,1-d*2),(0,1-d*2),(0,d*2),(d,d*2)])
#     g2 = gp.GeoSeries(p1)
#     g2.plot()
#     plt.show()  
    
    print "Generate wang tile points"
    for i in range(0,posTiles):
        wt = wangTiles[i,:]
        left = nozero(borderPointsV[wt[0]].copy())
        left[:,0] -= .5
        right = nozero(borderPointsV[wt[1]].copy())
        right[:,0] += .5
        bottom = nozero(borderPointsH[wt[2]].copy())
        bottom[:,1] -= .5
        top = nozero(borderPointsH[wt[3]].copy())
        top[:,1] += .5
        
        leftTop = cornerTiles[:,wt[3],wt[0],:].ravel()
        rightTop = cornerTiles[wt[3],:,wt[1],:].ravel()
        leftBot = cornerTiles[:,wt[2],:,wt[0]].ravel()
        rightBot = cornerTiles[wt[2],:,:,wt[1]].ravel()

        ltPoints = nozero3D(cornerPoints[leftTop])
        ltPoints[:,0] -= .5
        ltPoints[:,1] += .5
        rtPoints = nozero3D(cornerPoints[rightTop])
        rtPoints[:,0] += .5
        rtPoints[:,1] += .5
        lbPoints = nozero3D(cornerPoints[leftBot])
        lbPoints[:,0] -= .5
        lbPoints[:,1] -= .5
        rbPoints = nozero3D(cornerPoints[rightBot])
        rbPoints[:,0] += .5
        rbPoints[:,1] -= .5
        
        borderCornerPoints = numpy.concatenate((left,right,bottom,top,ltPoints,rtPoints,lbPoints,rbPoints),axis=0)
        existingPoints = borderCornerPoints
        for l in range(0,levels):
            points = generatePDD_new(gridRes[l],dist,l,tileMask,tileMaskFinal,minBoundX=d,minBoundY=d,maxBoundX=1-d,maxBoundY=1-d,existingPoints=existingPoints)
            
            existingPoints = numpy.concatenate((borderCornerPoints,points),axis=0)

        wangTilesPoints[i,0:points.shape[0],:] = points
         
    allPoints = numpy.zeros((totPoints*totalTiles,4))
    index = 0
    tileID = 0
    
    allCornerPoints = cornerPoints[cornerTiling]
    allCornerPoints[:,:,:,0:2] -= .5
    mask = numpy.all(allCornerPoints[:,:,:,0:2] > -.5,axis=3)
    allCornerPoints[:,:,:,0] += xIndices[:,:,numpy.newaxis]
    allCornerPoints[:,:,:,1] += yIndices[:,:,numpy.newaxis]
    
    print 'Generate Point tiling'
    for k in range(0,totalTiles):
        i = x[k]+1
        j = y[k]+1
        wangID = wangTiling[i,j]-1
        cornerP = allCornerPoints[i-1:i+1,j-1:j+1][mask[i-1:i+1,j-1:j+1]]
        mX = numpy.logical_and(cornerP[:,0] > i-1,cornerP[:,0] < i)
        mY = numpy.logical_and(cornerP[:,1] > j-1,cornerP[:,1] < j)
        m = numpy.logical_and(mX,mY)
        cp = cornerP[m]
        p = nozero(wangTilesPoints[wangID])
        #print p
        totP = p.shape[0]
        totCp = cp.shape[0]
        allPoints[index:index+totP,0] = p[:,0] + i - 1
        allPoints[index:index+totP,1] = p[:,1] + j - 1
        allPoints[index:index+totP,2] = p[:,2]
        allPoints[index+totP:index+totP+totCp,0:3] = cp
        allPoints[index:index+totP+totCp,3] = tileID
        tileID += 1   
        index += totP+totCp
    
    return nozero(allPoints)

def generatePoints_cornerbased(gridX,gridY,totalTiles,x,y,dist,levels,nCorners,visualize=False,bg2=numpy.array([])):
    bg = numpy.ones((gridX,gridY,3))
    if bg2.size == 0:
        bg2 = bg
        
        
    
    from galry import *    
    
    # face colors
#     totVert = position.shape[0]
#     
#     nPolygons = 4
#     vertices = np.zeros((totVert*nPolygons,3))
#     
#     for i in range(0,nPolygons):
#         temp = position.copy()
#         temp[:,0] += x[i]
#         temp[:,1] += y[i]
#         vertices[i*12:i*12+12,:] = temp
    
    gridRes = numpy.array([.25,.25,.25])
    b = dist[levels-1] * gridRes[levels-1]
    
    gr = int(np.ceil(1.0 / (b))) + 1
    totPoints = gr * gr
    
    nWangTiles = nCorners**4
    
    # Left, right, bottom, top
    print "Generate Wang Tiling"
    wangTiling = numpy.zeros((totalTiles),dtype=int)
    wangTilesCorners = numpy.zeros((nWangTiles,4),dtype=int)
    wangTiles = numpy.zeros((nCorners,nCorners,nCorners,nCorners),dtype=int)
    wangTiles += numpy.arange(0,nWangTiles).reshape((nCorners,nCorners,nCorners,nCorners))
    
    index = 0
    for i in range(0,nCorners):
        for j in range(0,nCorners):
            for k in range(0,nCorners):
                for m in range(0,nCorners):
                    wangTilesCorners[index] = [i,j,k,m]
                    index +=1
    
    wangCornerTiling = numpy.random.randint(0,nCorners,(gridX+1,gridY+1))
    
    for k in xrange(0,totalTiles):
        i = x[k]
        j = y[k]
        corners = wangCornerTiling[i:i+2,j:j+2].ravel()
        wangTiling[k] = wangTiles[corners[0],corners[1],corners[2],corners[3]]
                  
    wangTilesPoints = numpy.zeros((nWangTiles,totPoints,3))
    
    borderPointsH = numpy.zeros((nCorners,nCorners,totPoints,3))
    borderPointsV = numpy.zeros((nCorners,nCorners,totPoints,3))
    cornerPoints = numpy.zeros((nCorners,totPoints / 4,3))
    
    print "Generate corner points"
    d = dist[0] * .5
    cornerPolygon = np.array([(.5-d,.5-d*2),(.5+d,.5-d*2),(.5+d,.5+d*2),
                              (.5+d,.5+d*2),(.5-d,.5+d*2),(.5-d,.5-d*2),
                              (.5-d*2,.5-d),(.5-d*2,.5+d),(.5+d*2,.5+d),
                              (.5+d*2,.5+d),(.5+d*2,.5-d),(.5-d*2,.5-d)])
    
    cornerMask = Path(np.array([(.5-d,.5-d*2),(.5+d,.5-d*2),(.5+d,.5-d),(.5+d*2,.5-d),(.5+d*2,.5+d),(.5+d,.5+d),(.5+d,.5+d*2),(.5-d,.5+d*2),(.5-d,.5+d),(.5-d*2,.5+d),(.5-d*2,.5-d),(.5-d,.5-d)]))
    #cornerMask = Path(np.array([(.5-d,.5-d*2),(.5+d,.5-d*2),(.5+d*2,.5-d),(.5+d*2,.5+d),(.5+d,.5+d*2),(.5-d,.5+d*2),(.5-d*2,.5+d),(.5-d*2,.5-d)]))
    cornerMaskFinal = Path(np.array([(.5-d,.5-d*2),(.5+d,.5-d*2),(.5+d*2,.5-d),(.5+d*2,.5+d),(.5+d,.5+d*2),(.5-d,.5+d*2),(.5-d*2,.5+d),(.5-d*2,.5-d)]))
    for i in range(0,nCorners):
        existingPoints = numpy.array([],dtype=float).reshape(0,3)
        for l in range(0,levels):
            points = generatePDD_new(gridRes[l],dist,l,cornerMask,cornerMaskFinal,minBoundX=.5-d*2,minBoundY=.5-d*2,maxBoundX=.5+d*2,maxBoundY=.5+d*2,existingPoints=existingPoints,itr=50)
            existingPoints = numpy.concatenate((existingPoints,points),axis=0)
        cornerPoints[i,0:points.shape[0]] = points
        
        if visualize:
            cp = numpy.zeros((points.shape[0]),dtype=int)
            cp += points[:,2]
            pc = getColor(cp+5)
            colors = numpy.ones((cornerPolygon.shape[0]),dtype=int)
            colors += i
            colors = getColor(colors)
            
            figure()
            imshow(bg)
            plot(cornerPolygon[:,0],cornerPolygon[:,1],color=colors,primitive_type='TRIANGLES')
            plot(points[:,0],points[:,1],color=pc,primitive_type='POINTS',marker='.',marker_size=5)
            show() 
     
    print "Generate border points"
    d = dist[0]
    borderVMask = Path(np.array([(.5 - d*.5,d),(.5 + d*.5,d),(.5 + d*.5,1-d),(.5 - d*.5,1-d)])) 
    borderVPolygon = np.array([(.5 - d*.5,d),(.5 + d*.5,d),(.5 + d*.5,1-d),
                               (.5 + d*.5,1-d),(.5 - d*.5,1-d),(.5 - d*.5,d)]) 
    for i in range(0,nCorners):
        ct = nozero(cornerPoints[i,:])
        ct[:,1] += .5
        for j in range(0,nCorners):
            cb = nozero(cornerPoints[j,:])
            cb[:,1] -= .5
            borderPoints = numpy.concatenate((ct,cb),axis=0)
            existingPoints = borderPoints
            for l in range(0,levels):
                points = generatePDD_new(gridRes[l],dist,l,borderVMask,borderVMask,minBoundX=(.5 - d*.5),minBoundY=d,maxBoundX=(.5 + d*.5),maxBoundY=(1 - d),existingPoints=existingPoints,itr=50)
                existingPoints = numpy.concatenate((borderPoints,points),axis=0)
            borderPointsV[i,j,0:points.shape[0]] = points
       
    d = dist[0]
    borderHMask = Path(np.array([(d,.5 + d*.5),(d,.5 - d*.5),(1-d,.5 - d*.5),(1-d,.5 + d*.5)]))
    borderHPolygon = np.array([(d,.5 + d*.5),(d,.5 - d*.5),(1-d,.5 - d*.5),
                               (1-d,.5 - d*.5),(1-d,.5 + d*.5),(d,.5 + d*.5)])
  
    for i in range(0,nCorners):
        cr = nozero(cornerPoints[i,:])
        cr[:,0] += .5
        for j in range(0,nCorners):
            cl = nozero(cornerPoints[j,:])
            cl[:,0] -= .5
            
            borderPoints = numpy.concatenate((cr,cl),axis=0)
            existingPoints = borderPoints
            for l in range(0,levels):
                points = generatePDD_new(gridRes[l],dist,l,borderHMask,borderHMask,minBoundX=d,minBoundY=(.5 - d*.5),maxBoundX=(1 - d),maxBoundY=(.5 + d*.5),existingPoints=existingPoints,itr=50)
                existingPoints = numpy.concatenate((borderPoints,points),axis=0)
            borderPointsH[i,j,0:points.shape[0]] = points
            
            if visualize:
                cp = numpy.zeros((existingPoints.shape[0]),dtype=int)
                cp += existingPoints[:,2]
                pc = getColor(cp+5)
                
                clp = cornerPolygon.copy()
                crp = cornerPolygon.copy()
                bo = borderHPolygon.copy()
                clp[:,0] -= .5
                crp[:,0] += .5
                cps = numpy.concatenate((clp,crp))
                
                colors = numpy.ones((clp.shape[0]+crp.shape[0]),dtype=int)
                colors[0:clp.shape[0]] += j
                colors[clp.shape[0]:clp.shape[0]+crp.shape[0]] += i
                colors = getColor(colors)
                colorsb = numpy.ones((bo.shape[0],3))
                colorsb *= .75
                
                figure()
                imshow(bg)
                plot(cps[:,0],cps[:,1],color=colors,primitive_type='TRIANGLES')
                plot(bo[:,0],bo[:,1],color=colorsb,primitive_type='TRIANGLES')
                plot(existingPoints[:,0],existingPoints[:,1],color=pc,primitive_type='POINTS',marker='.',marker_size=5)
                show() 
      
    d = dist[0] * .5

    tileMask = Path(np.array([(d*2,d),(1-d*2,d),(1-d,d*2),(1-d,1-d*2),(1-d*2,1-d),(d*2,1-d),(d,1-d*2),(d,d*2)]))
    tileMask = Path(np.array([(d,d),(1-d,d),(1-d,1-d),(d,1-d)])) 
#     tileMaskFinal = Path(np.array([(d*2,d),(d*2,0),(1-d*2,0),(1-d*2,d),(1-d,d*2),(1,d*2),(1,1-d*2),(1-d,1-d*2),(1-d*2,1-d),(1-d*2,1),(d*2,1),(d*2,1-d),(d,1-d*2),(0,1-d*2),(0,d*2),(d,d*2)])) 
    tileMaskFinal = Path(np.array([(0,0),(1,0),(1,1),(0,1)])) 
     
    print "Generate wang tile points"
    
    dist *= .9
    for i in range(0,nWangTiles):
        wt = wangTilesCorners[i,:]
        leftTop = wt[1]
        rightTop = wt[3]
        leftBot = wt[0]
        rightBot = wt[2]
        left = nozero(borderPointsV[leftTop,leftBot])
        left[:,0] -= .5
        right = nozero(borderPointsV[rightTop,rightBot])
        right[:,0] += .5
        bottom = nozero(borderPointsH[rightBot,leftBot])
        bottom[:,1] -= .5
        top = nozero(borderPointsH[rightTop,leftTop])
        top[:,1] += .5
  
        ltPoints = nozero(cornerPoints[leftTop])
        ltPoints[:,0] -= .5
        ltPoints[:,1] += .5
        rtPoints = nozero(cornerPoints[rightTop])
        rtPoints[:,0] += .5
        rtPoints[:,1] += .5
        lbPoints = nozero(cornerPoints[leftBot])
        lbPoints[:,0] -= .5
        lbPoints[:,1] -= .5
        rbPoints = nozero(cornerPoints[rightBot])
        rbPoints[:,0] += .5
        rbPoints[:,1] -= .5
          
        borderCornerPoints = numpy.concatenate((left,right,bottom,top,ltPoints,rtPoints,lbPoints,rbPoints),axis=0)
        existingPoints = borderCornerPoints

        for l in range(0,levels):
            points = generatePDD_new(gridRes[l],dist,l,tileMask,tileMaskFinal,minBoundX=d,minBoundY=d,maxBoundX=1-d,maxBoundY=1-d,existingPoints=existingPoints,itr=50)
            existingPoints = numpy.concatenate((borderCornerPoints,points),axis=0)
#         print points
#         plt.figure()
#         plt.scatter(existingPoints[:,0],existingPoints[:,1],c='b')
#         plt.scatter(borderCornerPoints[:,0],borderCornerPoints[:,1],c='r')
#         plt.show()
        wangTilesPoints[i,0:points.shape[0],:] = points
    
    cornerPolygon -= .5
    vertCorner = cornerPolygon.shape[0]
    vertBorder = borderHPolygon.shape[0]
    totCorners = (gridX+1)*(gridY+1)
    totBorderH = gridX*(gridY+1)
    totBorderV = (gridX+1)*gridY
    vertices = numpy.zeros((vertCorner*totCorners+totBorderH*vertBorder+totBorderV*vertBorder,2))
    colors = numpy.zeros((vertCorner*totCorners+totBorderH*vertBorder+totBorderV*vertBorder,3))
    cornerTypes = wangCornerTiling.ravel()
    xci,yci = numpy.indices((gridX+1,gridY+1))
    xci = xci.ravel()
    yci = yci.ravel()
    xbhi,ybhi = numpy.indices((gridX,gridY+1))
    xbhi = xbhi.ravel()
    ybhi = ybhi.ravel()
    xbvi,ybvi = numpy.indices((gridX+1,gridY))
    xbvi = xbvi.ravel()
    ybvi = ybvi.ravel()
    for i in range(0,int(totCorners)):
        temp = cornerPolygon.copy()
        xc = xci[i]
        yc = yci[i]
        temp[:,0] += xc
        temp[:,1] += yc
        vertices[i*vertCorner:i*vertCorner+vertCorner,:] = temp
        colors[i*vertCorner:i*vertCorner+vertCorner] = getColor(int(cornerTypes[i]+1))
    
    borderHPolygon[:,1] -= .5
    borderVPolygon[:,0] -= .5
    start = totCorners*vertCorner
    gray = numpy.ones((vertBorder,3))
    gray *= .75
    for i in range(0,int(totBorderH)):
        tempH = borderHPolygon.copy()
        xbh = xbhi[i]
        ybh = ybhi[i]
        tempH[:,0] += xbh
        tempH[:,1] += ybh
        vertices[start+i*vertBorder:start+i*vertBorder+vertBorder,:] = tempH
        colors[start+i*vertBorder:start+i*vertBorder+vertBorder] = gray
    
    start = totCorners*vertCorner+totBorderH*vertBorder
    for i in range(0,int(totBorderV)):
        tempV = borderVPolygon.copy()
        xbv = xbvi[i]
        ybv = ybvi[i]
        tempV[:,0] += xbv
        tempV[:,1] += ybv
        vertices[start+i*vertBorder:start+i*vertBorder+vertBorder,:] = tempV
        colors[start+i*vertBorder:start+i*vertBorder+vertBorder] = gray
        
    print "Fill Wang Tiling"       
    tileIDs = numpy.arange(0,totalTiles)
    allPoints = wangTilesPoints[wangTiling]
    mask = numpy.all(allPoints[:,:,0:2] > 0,axis=2)
    allPoints[:,:,0] += x[:,numpy.newaxis]
    allPoints[:,:,1] += y[:,numpy.newaxis]
    test = numpy.zeros((allPoints.shape[0],allPoints.shape[1])) 
    test += tileIDs[:,numpy.newaxis]
    test = test[mask]
    allPoints = allPoints[mask]
    pointDistribution = numpy.zeros((allPoints.shape[0],4))
    pointDistribution[:,0:3] = allPoints
    pointDistribution[:,3] = test
    
    if visualize:
        pc = getColor(pointDistribution[:,2].astype(int)+5)
        figure()
        imshow(bg2)
        plot(vertices[:,0],vertices[:,1],color=colors,primitive_type='TRIANGLES')
        show() 
        figure()
        imshow(bg2)
        plot(vertices[:,0],vertices[:,1],color=colors,primitive_type='TRIANGLES')
        plot(pointDistribution[:,0],pointDistribution[:,1],color=pc,primitive_type='POINTS',marker='.',marker_size=2)
        show() 
        figure()
        imshow(bg2)
        plot(pointDistribution[:,0],pointDistribution[:,1],color=pc,primitive_type='POINTS',marker='.',marker_size=2)   
        show()     
    return pointDistribution

def generatePoints(gridX, gridY, totalTiles, x, y, tileSize):
    print gridX,gridY, gridX*gridY,totalTiles
    levels = 1
    dist=numpy.array([0.4,.2,.1])
    itr = 10
    t = int(tileSize / dist[levels-1])
    totPoints = 20 * t
    
    borderH = numpy.array([0,1,2])
    borderV = numpy.array([0,1,2])
    
    posTiles = borderH.size * borderH.size * borderV.size * borderV.size
    
    # Left, right, bottom, top
    wangTiles = numpy.zeros((posTiles,4))
    index = 0
    for i in range(0,borderV.size):
        for j in range(0,borderV.size):
            for m in range(0,borderH.size):
                for n in range(0,borderH.size):
                    wangTiles[index,:] = [borderV[i],borderV[j],borderH[m],borderH[n]]
                    index += 1
    
    sH = borderH.size * borderV.size * borderV.size
    sV = borderH.size * borderH.size * borderV.size
    sD = borderH.size * borderV.size
    combinationsSH = numpy.zeros((borderH.size,sH))
    combinationsSV = numpy.zeros((borderV.size,sV))
    combinationsD = numpy.zeros((borderH.size,borderV.size,sD))
    
    for i in range(0,borderH.size):
        b = borderH[i]
        m = wangTiles[:,2] == b
        combinationsSH[i,:] = numpy.where(m)[0]+1
    
    for i in range(0,borderV.size):
        b = borderV[i]
        m = wangTiles[:,0] == b
        combinationsSV[i,:] = numpy.where(m)[0]+1
    
    for i in range(0,borderH.size):
        for j in range(0,borderV.size):
            combinationsD[i,j,:] = numpy.intersect1d(combinationsSH[i],combinationsSV[j])    
        
    wangTiling = numpy.zeros((gridX,gridY))
    rn = numpy.random.randint(1,posTiles+1,(totalTiles))
    for i in range(0,totalTiles):
        if x[i] == 0 and y[i] == 0:
            wangTiling[x[i],y[i]] = rn[i]
        elif x[i] == 0:
            upTile = wangTiling[x[i],y[i]-1]
            if upTile == 0:
                wangTiling[x[i],y[i]] = rn[i]
            else:
                b = wangTiles[upTile-1,3]
                wangTiling[x[i],y[i]] = combinationsSH[b,rn[i] % sH]
        elif y[i] == 0:
            leftTile = wangTiling[x[i]-1,y[i]]
            if leftTile == 0:
                wangTiling[x[i],y[i]] = rn[i]
            else:
                b = wangTiles[leftTile-1,1]
                wangTiling[x[i],y[i]] = combinationsSV[b,rn[i] % sV]
        else:
            upTile = wangTiling[x[i],y[i]-1]
            leftTile = wangTiling[x[i]-1,y[i]]
            if upTile == 0 and leftTile == 0:
                wangTiling[x[i],y[i]] = rn[i]
            elif upTile == 0:
                b = wangTiles[leftTile-1,1]
                wangTiling[x[i],y[i]] = combinationsSV[b,rn[i] % sV]
            elif leftTile == 0:
                b = wangTiles[upTile-1,3]
                wangTiling[x[i],y[i]] = combinationsSH[b,rn[i] % sH]
            else:
                bH = wangTiles[upTile-1,3]
                bV = wangTiles[leftTile-1,1]
                wangTiling[x[i],y[i]] = combinationsD[bH,bV,rn[i] % sD]
                
    pointBorders = numpy.zeros((totalTiles * 4))
    pointCoordinates = numpy.zeros((totalTiles * 4,2))
    
    borderPointsLevelH = numpy.zeros((levels,borderH.size,totPoints,2))
    borderPointsLevelV = numpy.zeros((levels,borderV.size,totPoints,2))
    wangTilesPoints = numpy.zeros((posTiles,totPoints,2))
    wangTilesLevels = numpy.zeros((posTiles,totPoints))
    
    borderPointsH = numpy.zeros((borderH.size,totPoints,2))
    borderPointsV = numpy.zeros((borderV.size,totPoints,2))
    borderHMaskLevels = numpy.zeros((levels,borderH.size,totPoints),dtype=bool)
    borderVMaskLevels = numpy.zeros((levels,borderV.size,totPoints),dtype=bool)
    
    for i in range(0,borderV.size):
        existingPoints = numpy.array([],dtype=float).reshape(0,2)
        for l in range(0,levels):
            points = generatePDD(totPoints,itr,dist[l],limUp=tileSize,inputPoints=existingPoints)
            border = points[:,0] > tileSize-dist[0]*1
            p = points[border,:]
            borderPointsV[i,0:p.shape[0]] = p
            borderPointsLevelV[l,i,0:p.shape[0]] = p
            borderVMaskLevels[l,i,existingPoints.shape[0]:p.shape[0]] = True
            existingPoints = numpy.concatenate((existingPoints,p),axis=0)
     
    for i in range(0,borderH.size):
        existingPoints = numpy.array([],dtype=float).reshape(0,2)
        for l in range(0,levels):
            points = generatePDD(totPoints,itr,dist[l],limUp=tileSize,inputPoints=existingPoints)
            border = points[:,1] > tileSize-dist[0]*1
            p = points[border,:]
            borderPointsH[i,0:p.shape[0]] = p
            borderPointsLevelH[l,i,0:p.shape[0]] = p
            borderHMaskLevels[l,i,existingPoints.shape[0]:p.shape[0]] = True
            existingPoints = numpy.concatenate((existingPoints,p),axis=0)
    
    for i in range(0,posTiles):
        wt = wangTiles[i,:]
        testBorder = numpy.concatenate((borderPointsH[wt[3],:,:],borderPointsV[wt[1],:,:]),axis=0)
        m = numpy.all(testBorder > 0,axis=1)
        masks = numpy.concatenate((borderHMaskLevels[:,wt[3],:],borderVMaskLevels[:,wt[1],:]),axis=1)
        masks = masks[:,m]
        tempBorder = testBorder[m]
        
        d = spatial.distance.cdist(tempBorder,tempBorder)
        d[d == 0] = numpy.Inf
        tempMask = numpy.zeros((masks.shape[1]),dtype=bool)
        for l in range(0,levels):
            indices = numpy.where(masks[l,:])[0]
            tempMask += masks[l,:]
            dl = d.copy()
            dl = dl[:,tempMask]
            dl = dl[masks[l,:],:]
            while(numpy.any(dl < dist[l])):
                ind = numpy.argmax(numpy.sum(dl < dist[l],axis=1))
                dl[ind,:] = numpy.Inf
                dl[:,ind] = numpy.Inf
                
                d[indices[ind],:] = numpy.Inf
                d[:,indices[ind]] = numpy.Inf
                masks[l,indices[ind]] = False
        
        for l in range(0,levels):
            indices = numpy.where(masks[l,:])[0]
            masks[l,indices[tempBorder[masks[l,:],0] < dist[l] * .75]] = False
            masks[l,indices[tempBorder[masks[l,:],1] < dist[l] * .75]] = False
        
        pm = numpy.sum(masks,axis=0) == 1
        p0 = numpy.sum(masks,axis=0) == 0
        testBorder = tempBorder[pm]
        masks = masks[:,pm]
        vLeft = borderPointsV[wt[0],:,:]
        vLeftLevels = borderVMaskLevels[:,wt[0],:]
        m = numpy.all(vLeft > 0,axis=1)
        vLeft = vLeft[m]
        vLeftLevels = vLeftLevels[:,m]
        vLeft[:,0] -= tileSize
        
        vBot = borderPointsH[wt[1],:,:]
        vBotLevels = borderHMaskLevels[:,wt[1],:]
        m = numpy.all(vBot > 0,axis=1)
        vBot = vBot[m]
        vBotLevels = vBotLevels[:,m]
        vBot[:,1] -= tileSize
        vPoints = numpy.concatenate((vBot,vLeft),axis=0)
        maskVirtual = numpy.concatenate((vBotLevels,vLeftLevels),axis=1)
          
        p,pl = generatePDDLevels(totPoints,itr,dist,tileSize,0,inputPoints=testBorder,virtualPoints=vPoints,inputLevels=masks,virtualLevels=maskVirtual)
        p[numpy.logical_and(p[:,0] < dist[l] * .75, p[:,1] < dist[l] * .75 ),:] = 0
    #     for l in range(0,levels):
    #         indices = numpy.where(masks[l,:])[0]
    #         masks[l,indices[tempBorder[masks[l,:],0] < dist[l] * .75]] = False
    #         masks[l,indices[tempBorder[masks[l,:],1] < dist[l] * .75]] = False
        wangTilesPoints[i,0:p.shape[0],:] = p
        wangTilesLevels[i,0:p.shape[0]] = pl+1
        
    allPoints = numpy.zeros((totPoints*gridX*gridY,4))
    index = 0
    tileID = 0
    for k in range(0,totalTiles):
        i = x[k]
        j = y[k]
        wangID = wangTiling[i,j]-1
        p = wangTilesPoints[wangID]
        pl = wangTilesLevels[wangID]
        m = numpy.all(p > 0,axis=1)
        allPoints[index:index+p[m,0].size,0] = p[m,0] + i * tileSize
        allPoints[index:index+p[m,1].size,1] = p[m,1] + j * tileSize
        allPoints[index:index+p[m,1].size,2] = tileID
        allPoints[index:index+p[m,1].size,3] = pl[m]
        tileID += 1   
        index += p[m,0].size
    
    m = numpy.all(allPoints[:,0:2] > 0,axis=1)
    allPoints = allPoints[m,:]
    
    return allPoints
    
    