from PIL import Image
from galry import *
import numpy
from scipy import interpolate
from scipy import spatial
from scipy import misc
import math
import fractalnoise
import wangtiles
from skimage.util.shape import view_as_blocks

def getHeightMatrix(area):
    if area == "NDVI":
        height = numpy.array([0,150,160,170,180,190,200,210,220,230,240,250,260,270,280,350])
    elif area == "MODEL":
        height = numpy.array([0,10,20,30,40,50,60,70,80,90,100,200])
    return height

def getCoverageMatrix(area):
    if area == "NDVI":
        #Elymus, Spartina, Atriplex, Aster, Limonium, Artemisia, Salicornia
        coverage = numpy.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,60,80,100,100],
                                [99,99,99,99,95,90,85,80,50,20, 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,20,45,40,20, 0, 0],
                                [ 0, 0, 0, 0, 5,10,15,20,10,10,10,10, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0,15,30,25,15, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0,25,40,45,30, 0, 0, 0, 0],
                                [ 7, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0]])
    if area == "LCC":
        #Elymus, Spartina, Atriplex, Aster, Limonium, Artemisia, Salicornia
        coverage = numpy.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,60,60,80,100,100],
                                [99,99,99,99,95,90,85,80,80,80,70,60, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,20,45,40,20, 0, 0],
                                [ 0, 0, 0, 0, 5,10,15,20,10,10,10,10, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0,15,30,25,15, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0, 0, 0,25,40,45,30, 0, 0, 0, 0],
                                [ 7, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0]])
    elif area == "MODEL":
            coverage = numpy.array([[99,99,99,99,99,99,99,99,95,90,90,90],
                                    [ 0, 0, 0, 0, 0, 0, 0, 0, 5,10,10,10],
                                    [ 0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 0, 0]])
    return coverage

def getHurstMatrix(area):
    if area == "NDVI":
        #Elymus, Spartina, Atriplex, Aster, Limonium, Artemisia, Salicornia
        hurst = numpy.array([[.80,.80,.80,.80,.80,.80,.80,.80,.80,.80,.80,.80,.83,.88,.90,.90],
                             [.70,.70,.70,.70,.70,.70,.70,.70,.70,.70,.70,.70,.80,.80,.80,.80],
                             [.65,.65,.65,.65,.65,.65,.65,.65,.65,.65,.65,.75,.80,.85,.85,.85],
                             [.10,.10,.10,.10,.10,.10,.10,.10,.10,.10,.10,.10,.10,.10,.10,.10],
                             [.45,.45,.45,.45,.45,.45,.45,.45,.35,.35,.40,.45,.45,.45,.45,.45],
                             [.56,.56,.56,.56,.56,.56,.56,.56,.65,.70,.72,.80,.72,.56,.56,.56],
                             [.10,.10,.10,.10,.10,.10,.10,.10,.10,.10,.10,.10,.10,.10,.10,.10]])
    elif area == "MODEL":
        hurst = numpy.array([[.80,.80,.80,.80,.80,.80,.80,.80,.80,.80,.80,.80],
                             [.10,.10,.10,.10,.10,.10,.10,.10,.10,.10,.10,.10],
                             [.10,.10,.10,.10,.10,.10,.10,.10,.10,.10,.10,.10]])
    return hurst

def calculateCovNDVI(area,i,ndvi):
    if area == "NDVI":
        if i != 6:
            return ndvi >= .08 #.04
        else:
            return ndvi < .08 #.04
    if area == "MODEL":
        if i != 2:
            return ndvi >= .01
        else:
            return ndvi < .01 

def calculateCovLCC(i,lccValues):
    #Elymus, Spartina, Atriplex, Aster, Limonium, Artemisia, Salicornia
    if i == 0:
        mask = lccValues == 4
    elif i == 1:
        mask = numpy.logical_and(lccValues > 0, lccValues < 3)
    elif i == 2:
        mask = lccValues == 3
    elif i == 3:
        mask = numpy.logical_and(lccValues > 1,lccValues < 3) 
    elif i == 4:
        mask = lccValues == 3
    elif i == 5:
        mask = lccValues == 3
    elif i == 6:
        mask = lccValues == 1

    return mask.astype(float)

def getLCCTypes(i):
    if i == 0:
        return numpy.array([])
    elif i == 1:
        return numpy.array([1,6])
    elif i == 2:
        return numpy.array([1,3])
    elif i == 3:
        return numpy.array([2,4,5])
    elif i == 4:
        return numpy.array([0])

def getFractalNoise_new(points,nTypes,hurst,frequencies):
    totalPoints = points.shape[0]  
    frequencies = 1 - frequencies
    cNoise = numpy.zeros((nTypes,totalPoints),dtype=float)
    for i in range(0,nTypes):
        print "type",i
        # Calculate number of octaves
        h = hurst[i]
        f = frequencies[i]
        lac = 1.87
        f **= 2.2#spectralComp #like this?
        freq = numpy.min(f)
        offset = 0.0
        
        octaves = numpy.log(1/freq) / numpy.log(lac) + 1
        print octaves
        octavesInt = numpy.floor(octaves).astype(int)
        print octavesInt
        octavesInd = numpy.arange(0,octavesInt+1)
        exponent_array = numpy.power(lac, numpy.outer(octavesInd,-h))
        freq_array = freq * numpy.power(lac,octavesInd)
        weights = f[numpy.newaxis,:] / freq_array[:,numpy.newaxis]
        weights[weights > 1] **= -.5 
        weights **= 2.0
        
        w =  numpy.max(weights / numpy.sum(weights,axis=0),axis=1)
        e = numpy.max(exponent_array / numpy.sum(exponent_array,axis=0),axis=1)
        mW = w < .01
        mE = e < .01
        m = numpy.logical_and(mW,mE)
        
        if octavesInd[m].size != 0:
            octavesInt = octavesInd[m][0]
            print octavesInt
        
        print "Generate Simplex noise"
        octaveNoise = numpy.zeros((octavesInt,totalPoints),dtype=float)
        for j in range(0,octavesInt):
            sNoise = ((getSimplexNoise_new(points, i, freq_array[j]) + offset) * exponent_array[j])
            octaveNoise[0:j+1,:] += sNoise
         
        cumsumExponent = numpy.cumsum(exponent_array[0:octavesInt,:][::-1,:],axis=0)[::-1,:]
        
        remainder = octaves - octavesInt 
        if (remainder > 0):
            octaveNoise += (getSimplexNoise_new(points,i, freq_array[octavesInt])+offset) * exponent_array[octavesInt]**remainder   
            cumsumExponent += exponent_array[octavesInt]**remainder
        octaveNoise /= cumsumExponent
        
        totWeights = numpy.sum(weights[0:octavesInt],axis=0)
        cNoise[i,:] += numpy.sum(octaveNoise[0:octavesInt] * weights[0:octavesInt],axis=0)
        cNoise[i,:] /= totWeights
        
    return cNoise

def getSimplexNoise_new(points, pType, freq):
    simplexNoise = fractalnoise.raw_noise_2d(points*freq, pType)
    simplexNoise /= numpy.sum(simplexNoise)
    simplexNoise *= simplexNoise.shape[0]
    return simplexNoise

def getColor(typeList):
    defColors = numpy.array([[0,0,0],[0,.5,0],[1,1,0],[1,0,0],[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    
    return defColors[typeList]

def classifyPoints(sortedFractalNoise, fractalNoise, coveragePoints, pointIndices, pointType, points, plantTypes, visualize):
    nTypes = coveragePoints.shape[0]
    totalPoints = pointIndices.shape[0]
    thresholdIndex = (coveragePoints * (totalPoints-1)).astype(int)
    
    selectedPositions = numpy.zeros((nTypes,totalPoints),dtype=bool)
    selectedValues = numpy.zeros((nTypes,totalPoints))
    for i in range(0,nTypes):
        thresholdValues = sortedFractalNoise[i,thresholdIndex[i,:]]
        thresholdValues[coveragePoints[i,:] == 0] = numpy.Inf
        selectedPositions[i,:] = fractalNoise[i,:] >= thresholdValues
        selectedValues[i,selectedPositions[i]] = fractalNoise[i,selectedPositions[i]]
    
    nSelections = numpy.sum(selectedPositions,axis=0)
    selectedPoints = nSelections > 0
    ind = pointIndices[selectedPoints]
    pointType[ind] = plantTypes[numpy.argmax(selectedValues[:,selectedPoints],axis=0).astype(int)]
    if visualize:
        for i in range(0, nTypes):
            pointColors = getColor(selectedPositions[i,:] * plantTypes[i])
            figure(constrain_navigation=True,constrain_ratio=True,antialiasing=True)
            plot(points[pointIndices,0],points[pointIndices,1],primitive_type='POINTS',color=pointColors,marker='.',marker_size=6)
            show()
        
        pointColors = getColor(pointType)
        pointColors[nSelections > 1] = [1,0,0]
        figure(constrain_navigation=True,constrain_ratio=True,antialiasing=True)
        plot(points[:,0],points[:,1],primitive_type='POINTS',color=pointColors,marker='.',marker_size=6)
        show()
        
        pointColors = getColor(pointType)
        figure(constrain_navigation=True,constrain_ratio=True,antialiasing=True)
        plot(points[:,0],points[:,1],primitive_type='POINTS',color=pointColors,marker='.',marker_size=6)
        show()
        
    return pointType    

def classify_final(fractalNoise,coveragePoints, heightPoints,ndviPoints,points,heightCoverages,noCov,unit,order,plantTypeLevels,plantInfluences=numpy.array([])):
    levels = numpy.unique(points[:,2]).size
    visualize = True
    tree = spatial.cKDTree(points[:,0:2])
    nTypes = fractalNoise.shape[0]
    totalPoints = fractalNoise.shape[1]
    pointType = numpy.zeros((totalPoints),dtype=int)
    plantInfluencesPoints = numpy.zeros((nTypes,totalPoints))
    pointIndices = numpy.arange(0,points.shape[0])
    sortedIndices = numpy.argsort(fractalNoise,axis=1)[:,::-1]
    fractalNoiseSorted = fractalNoise[numpy.arange(nTypes)[:, None],sortedIndices]
    
    heightBins = numpy.zeros((2,points.shape[0]))
    heightBins[:,:] = (heightPoints / unit)[numpy.newaxis,:]
    heightBins = numpy.trunc(heightBins)
    heightBins[1,:] += 1
    heightBins *= unit
    print heightBins
    weights = (unit - numpy.abs(heightPoints[numpy.newaxis,:] - heightBins)).ravel()
    uq,ui = numpy.unique(heightBins,return_inverse=True)
    heightRange = uq.size

    for l in range(0,levels):
#         nocovFunction = interpolate.interp1d(uq,noCov, kind='slinear',bounds_error=False,fill_value=0)
        nocoverage = noCov
        m1 = points[:,2] <= l
        pTypes = numpy.where(plantTypeLevels == l)[0]
        nTypes = pTypes.size
        orderLevel = pTypes.copy()
        for o in range(0,orderLevel.shape[0]):
            orderLevel[o] = numpy.where(pTypes[o] == order)[0]
        #orderLevel = pTypes.copy()
        totCov = numpy.sum(coveragePoints[pTypes],axis=0)
        if nTypes == 0:
            break
        m2 = pointType == 0
        m = numpy.logical_and(m1,m2)
        pInd = pointIndices[m]
        fns = fractalNoiseSorted[m[sortedIndices[pTypes,:]]].reshape(nTypes,pInd.shape[0])
        fn = fractalNoise[pTypes[:,np.newaxis],pInd[np.newaxis,:]]
        cov = coveragePoints[pTypes[:,np.newaxis],pInd[np.newaxis,:]]
#         pointType = classifyPoints(fractalNoiseSorted, fractalNoise, coveragePoints, pointIndices, pointType, points,plantTypes, visualize)
        pointType = classifyPoints(fns, fn, cov, pInd, pointType, points,pTypes+1, visualize)
        
        print "Update coverage values"
        ml = numpy.tile(m,2)
        uil = ui[ml]
        weightsl = weights[ml]
        totCovPerHeight = numpy.bincount(uil,weights=weightsl,minlength=heightRange)
        totCovPerHeight[totCovPerHeight == 0] = 1
        for i in range(0,nTypes):
            i = pTypes[i]
            maskPT = pointType == i+1
            maskPTd = numpy.tile(maskPT,2)
            totCovPerHeightType = numpy.bincount(ui[maskPTd],weights=weights[maskPTd],minlength=heightRange)
            heightCoveragesNew = heightCoverages[i,:] - totCovPerHeightType / totCovPerHeight
            covFunction = interpolate.interp1d(uq,heightCoveragesNew, kind='slinear',bounds_error=False,fill_value=0)
            rest = covFunction(heightPoints)
            rest = numpy.minimum(rest,ndviPoints[i])
            # update step - TOBE further investigated with MASKPT
            covLowerThan0 = rest < 0
            covHigherThan0 = rest > 0
            totMinCov = numpy.sum(rest[covLowerThan0])
            totPlusCov = numpy.sum(rest[covHigherThan0])
            totMinCov /= float(totalPoints)
            totPlusCov /= float(totalPoints)
#             rest += (totMinCov + totPlusCov) * rest
            rest += plantInfluencesPoints[i,:]
            rest[rest < 0] = 0
            rest[rest > 1] = 1
            coveragePoints[i,:] = rest
            
        totCov = numpy.sum(coveragePoints[pTypes,:],axis=0)
        totCov[totCov == 0] = 100.0
        totCov += nocoverage
        nocoverage /= totCov
        coveragePoints[pTypes,:] /= totCov
        
        print "Classify remaining points"
        iTypes = pTypes.copy()
        for i in range(0,nTypes):
            i = orderLevel[i]
            pt = numpy.take(iTypes,[i])
            i = iTypes[i]
            remainingPoints = numpy.logical_and(m,pointType == 0)
            pInd = pointIndices[remainingPoints]
            if pInd.size == 0:
                break
              
            nPoints = pInd.shape[0]
            fns = fractalNoiseSorted[i,remainingPoints[sortedIndices[i,:]]].reshape(1,nPoints)
            fn = fractalNoise[i,pInd].reshape(1,nPoints)
            cov = coveragePoints[i,pInd].reshape(1,nPoints)
  
            pointType = classifyPoints(fns, fn, cov, pInd, pointType, points,pt+1,visualize)
            
            mask = pTypes != i
            pTypes = pTypes[mask]
            if pTypes.size != 0: 
                totCov = numpy.sum(coveragePoints[pTypes,:],axis=0) + nocoverage
                totCov[totCov == 0] = 100.0
                coveragePoints[pTypes,:] /= totCov
                nocoverage /= totCov
               
        if plantInfluences.size != 0 or numpy.sum(plantInfluences) != 0:
            print "Apply plant influences"
            for i in range(0,iTypes.size):
                if numpy.sum(plantInfluences[iTypes[i]]) != 0:
                    pos = pointType == iTypes[i]+1
                    influencedPoints = tree.query_ball_point(points[pos,0:2],.5)
                    for k in range(0,numpy.sum(pos)):
                        for j in range(0,plantInfluences.shape[1]):
                            plantInfluencesPoints[j,influencedPoints[k]] += plantInfluences[iTypes[i],j]
            coveragePoints += plantInfluencesPoints
            coveragePoints[coveragePoints < 0] = 0 

    return pointType
            
def main():
    print "Load data"
    
    sourceHeight = 'height_masked_final.asc'
    sourceBiomass = 'ndvi_masked_final.asc'

    sourceCoverageModel = 'testCov.txt'
    sourceHeightModel = 'heightModel.txt'
    sourceLCC = "LCC.asc"

    heightGrid = numpy.loadtxt(sourceHeight, skiprows=6)
    heightGrid = heightGrid[500:1050,300:750]
#    heightGrid = heightGrid[640:685,455:500]
#    heightGrid = heightGrid[860:890,600:700]
    rgb = (heightGrid - numpy.min(heightGrid)) / (numpy.max(heightGrid) - numpy.min(heightGrid))
    rgb *= 255
    heightRGBA = numpy.zeros((heightGrid.shape[0],heightGrid.shape[1],3),dtype=numpy.uint8)
    heightRGBA[:,:,0:3] = rgb[:,:,numpy.newaxis]
    #misc.imsave('heightMap_Paulinapolder.png',heightRGBA)
    ndviGrid = numpy.loadtxt(sourceBiomass, skiprows=6)
    ndviGrid = ndviGrid[500:1050,300:750]
#    ndviGrid = ndviGrid[640:685,455:500]
#    ndviGrid = ndviGrid[860:890,600:700]
    rgb = (ndviGrid - numpy.min(ndviGrid)) / (numpy.max(ndviGrid) - numpy.min(ndviGrid))
    rgb *= 255
    ndviRGBA = numpy.zeros((ndviGrid.shape[0],ndviGrid.shape[1],3),dtype=numpy.uint8)
    ndviRGBA[:,:,0:3] = rgb[:,:,numpy.newaxis]
    misc.imsave('ndviMap_Paulinapolder.png',ndviRGBA)
    lccGrid = numpy.loadtxt(sourceLCC, skiprows=6)
    lccGrid = lccGrid[500:1050,300:750]
    heightModelGrid = numpy.loadtxt(sourceHeightModel)
    coverageModelGrid = numpy.loadtxt(sourceCoverageModel)
    PaulinaPolder = False
    NDVI = True
    LCC = False
    if PaulinaPolder:
        if NDVI and LCC:
            vegetationMask = ndviGrid > 0
        elif NDVI:
            vegetationMask = ndviGrid > 0.08 #0.02 demo
#             figure()
#             tempveg = numpy.zeros((ndviGrid.shape[0],ndviGrid.shape[1],3))
#             tempveg[:,:,:] = (ndviGrid[:,:,numpy.newaxis]+1) / 2.0;
#             imshow(tempveg)
#             show()
#             figure()
#             tempveg = numpy.zeros((vegetationMask.shape[0],vegetationMask.shape[1],3))
#             tempveg[:,:,:] = vegetationMask[:,:,numpy.newaxis];
#             imshow(tempveg)
#             show()
#             figure()
#             tempveg = numpy.zeros((vegetationMask.shape[0],vegetationMask.shape[1],3))
#             tempveg[:,:,:] = heightGrid[:,:,numpy.newaxis] / numpy.max(heightGrid);
#             imshow(tempveg)
#             show()
        elif LCC:
            vegetationMask = lccGrid > 0
        heightValues = heightGrid[vegetationMask]
        baseValues = ndviGrid[vegetationMask]
        lccValues = lccGrid[vegetationMask]
        lengthX, lengthY = heightGrid.shape
        nTypes = 7
        area = "NDVI"
        
        lXTemp = lengthX
        lYTemp = lengthY
        if lengthX % 2 == 1:
            lXTemp += 1
        if lengthY % 2 == 1:
            lYTemp += 1
        vegetationMaskExtended = np.zeros((lXTemp,lYTemp),dtype=bool)        
        vegetationMaskExtended[0:lengthX,0:lengthY] = vegetationMask
        res = 2.0  # 2.0
        wangGridLengthX = np.ceil(lengthX / res)
        wangGridLengthY = np.ceil(lengthY / res)
        xWangIndices,yWangIndices = numpy.indices((wangGridLengthX,wangGridLengthY))
        blocks = view_as_blocks(vegetationMaskExtended, block_shape=(int(res),int(res)))
        blocks = blocks.reshape(wangGridLengthX,wangGridLengthY,res*res)
        blocks_summed = np.sum(blocks,axis=2)
        wangVegetationMask = blocks_summed > 0
        print "wvm",wangVegetationMask.shape
        print "vm", vegetationMask.shape
    else:
        vegetationMask = coverageModelGrid > 0
        heightValues = heightModelGrid[vegetationMask] * 1
        baseValues = coverageModelGrid[vegetationMask] * .01
        lengthX, lengthY = heightModelGrid.shape
        nTypes = 3
        heightValues += numpy.fabs(numpy.min(heightValues))
        minHeight = numpy.min(heightValues)
        maxHeight = numpy.max(heightValues)
        heightValues = (heightValues - minHeight) / (maxHeight - minHeight)
        rgb = (heightModelGrid - numpy.min(heightModelGrid)) / (numpy.max(heightModelGrid) - numpy.min(heightModelGrid))
        rgb *= 255
        heightRGBA = numpy.zeros((heightModelGrid.shape[0],heightModelGrid.shape[1],3),dtype=numpy.uint8)
        heightRGBA[:,:,0:3] = rgb[:,:,numpy.newaxis]
        #misc.imsave('heightMap_Ecomodel.png',heightRGBA)
        
        rgb = (coverageModelGrid - numpy.min(coverageModelGrid)) / (numpy.max(coverageModelGrid) - numpy.min(coverageModelGrid))
        rgb *= 255
        coverageRGBA = numpy.zeros((coverageModelGrid.shape[0],coverageModelGrid.shape[1],3),dtype=numpy.uint8)
        coverageRGBA[:,:,0:3] = rgb[:,:,numpy.newaxis]
        #misc.imsave('coverageMap_Ecomodel.png',coverageRGBA)
        
        heightValues *= 100
        area = "MODEL"
        
        tileIDs = numpy.arange(0,heightValues.shape[0])
        tileIDsGrid = numpy.zeros((lengthX,lengthY))
        tileIDsGrid[vegetationMask] = tileIDs
        
        wangRes = 1
        res = 1
        wangLengthX = lengthX * res
        wangLengthY = lengthY * res
        
        wangVegetationMask = numpy.zeros((wangLengthX,wangLengthY),dtype=bool)
        xWangIndices,yWangIndices = numpy.indices((wangLengthX,wangLengthY))
        xw = numpy.trunc(xWangIndices / res).astype(int)
        yw = numpy.trunc(yWangIndices / res).astype(int)
        wangVegetationMask[xWangIndices,yWangIndices] = vegetationMask[xw,yw]
        print wangVegetationMask.shape
    
    nTiles = heightValues.shape[0]
    plantTypeLevels = numpy.zeros((nTypes))
    
    if PaulinaPolder:
        if LCC: 
            area2 = "LCC"
        elif NDVI:
            area2 = "NDVI"
        heightMatrix = getHeightMatrix("NDVI")
        coverageMatrix = getCoverageMatrix(area2) * .01
        hurstMatrix = getHurstMatrix("NDVI")
    else:
        heightMatrix = getHeightMatrix("MODEL")
        coverageMatrix = getCoverageMatrix("MODEL") * .01
        hurstMatrix = getHurstMatrix("MODEL")
    
    unit = 1.0
    heightBins = numpy.zeros((2,heightValues.shape[0]))
    heightBins[:,:] = (heightValues / unit)[numpy.newaxis,:]
    heightBins = numpy.trunc(heightBins)
    heightBins[1,:] += 1
    heightBins *= unit
    heightBins = numpy.unique(heightBins)
#     print heightBins
    heightCoverage = numpy.zeros((nTypes,heightBins.size))
    heightHurst = numpy.zeros((nTypes,heightBins.size))
    
    for i in range(0,nTypes):
        fc = interpolate.interp1d(heightMatrix,coverageMatrix[i], kind='slinear',bounds_error=False,fill_value=0)
        fh = interpolate.interp1d(heightMatrix,hurstMatrix[i], kind='slinear',bounds_error=False,fill_value=0)
        heightCoverage[i] = fc(heightBins)
        heightHurst[i] = fh(heightBins)
    
    print "Declare data"
    coveragePerTile = numpy.zeros((nTypes,nTiles),dtype=float)
    hurstPerTile = numpy.zeros((nTypes,nTiles),dtype=float)    
    constraintsPerTile = numpy.zeros((nTypes,nTiles),dtype=float)
    compositionPerTile = numpy.zeros((nTypes,nTiles),dtype=float)
    ndviPerTile = numpy.ones((nTypes,nTiles),dtype=float)
    
    print "Get data"
    for i in range(0,nTypes):
        fc = interpolate.interp1d(heightBins,heightCoverage[i], kind='slinear',bounds_error=False,fill_value=0)
        fh = interpolate.interp1d(heightBins,heightHurst[i], kind='slinear',bounds_error=False,fill_value=0)
        coveragePerTile[i] = fc(heightValues)
        hurstPerTile[i] = fh(heightValues)
        if NDVI and LCC:
            ndviPerTile[i] = calculateCovNDVI(area,i,baseValues)
            lccTiles = calculateCovLCC(i,lccValues)
            constraintsPerTile[i] = numpy.minimum(ndviPerTile[i],lccTiles)
            compositionPerTile[i] = numpy.minimum(coveragePerTile[i],constraintsPerTile[i])
        elif NDVI:
            ndviPerTile[i] = calculateCovNDVI(area,i,baseValues)
            constraintsPerTile[i] = ndviPerTile[i]
            compositionPerTile[i] = numpy.minimum(coveragePerTile[i],constraintsPerTile[i])
        elif LCC:
            constraintsPerTile[i] = calculateCovLCC(i,lccValues)
            compositionPerTile[i] = numpy.minimum(coveragePerTile[i],constraintsPerTile[i])
    
    if LCC:
        if NDVI:
            tempCov = numpy.minimum(ndviPerTile,coveragePerTile)
        else:
            tempCov = numpy.minimum(ndviPerTile,coveragePerTile)#coveragePerTile.copy()#numpy.ones((nTypes,nTiles))
        lccGroups = 4
        noCoveragePerTile = 1 - numpy.sum(tempCov,axis=0)
        noCoveragePerTile[noCoveragePerTile < 0] = 0
        for i in range(0,lccGroups):
            lccMask = numpy.where(lccValues == i+1)[0]
            plantsLCC = getLCCTypes(i+1)
            tcon = constraintsPerTile[:,lccMask]
            tcon = tcon[plantsLCC,:]
            totCovlcc = numpy.sum(tcon,axis=0)
            for j in range(0,plantsLCC.shape[0]):
                compositionPerTile[plantsLCC[j],lccMask] /= (totCovlcc+noCoveragePerTile[lccMask])
    elif NDVI:
        totCov = numpy.sum(compositionPerTile,axis=0)
        noCoveragePerTile = 1 - totCov
        noCoveragePerTile[noCoveragePerTile < 0] = 0
        compositionPerTile /= totCov+noCoveragePerTile
    
    xWang = xWangIndices[wangVegetationMask]
    yWang = yWangIndices[wangVegetationMask]
    nWangTiles = xWang.shape[0]  
    print "Start Point Generation"
    if PaulinaPolder:
        tileIDs = numpy.arange(0,nTiles)
        tileIDsGrid = numpy.zeros((lengthX,lengthY))
        tileIDsGrid[vegetationMask] = tileIDs
        dist = numpy.array([.4]) / res # .4 standard # .8 medium# 1.6 low
        points = wangtiles.generatePoints_cornerbased(wangGridLengthX,wangGridLengthY, nWangTiles, xWang, yWang,dist,1,4)
        points[:,0:2] *= res
        trunckedPoints = np.trunc(points[:,0:2]).astype(int)
        cull = vegetationMaskExtended[trunckedPoints[:,0],trunckedPoints[:,1]]
        points = points[cull,:]
        trunckedPoints = trunckedPoints[cull,:]
        points[:,3] = tileIDsGrid[trunckedPoints[:,0],trunckedPoints[:,1]]
    else:
        dist = numpy.array([.33]) / wangRes
        points = wangtiles.generatePoints_cornerbased(wangLengthX,wangLengthY, nWangTiles, xWang, yWang,dist,1,4)
        points[:,0:2] *= wangRes
        trunckedPoints = np.trunc(points[:,0:2] / (res*wangRes)).astype(int)
        points[:,3] = tileIDsGrid[trunckedPoints[:,0],trunckedPoints[:,1]]
    
    figure(constrain_navigation=True,constrain_ratio=True,antialiasing=True)
    #imshow(numpy.rot90(tempveg,1))
    plot(points[:,0],points[:,1],color='w',primitive_type='POINTS',marker='.',marker_size=6)
    show()
    totalPoints = points.shape[0]
    print totalPoints
    print totalPoints / (nTiles * 1.0); 
    
    hurstPoints = hurstPerTile[:,points[:,3].astype(int)]
    coveragePoints = compositionPerTile[:,points[:,3].astype(int)]
    heightPoints = heightValues[points[:,3].astype(int)]
    nocoveragePoints = noCoveragePerTile[points[:,3].astype(int)]
    basePoints = constraintsPerTile[:,points[:,3].astype(int)]
    
        
    print "Generate MultiFractal Noise"        
    scaleFactor = math.sqrt(totalPoints / (nTiles * 1.0)) * 0.5
    cNoisePoints = getFractalNoise_new(points[:,0:2]*scaleFactor,nTypes,hurstPoints,hurstPoints)

    avgFreqT = numpy.average(hurstPoints, axis=1)
    meanFreq = numpy.average(avgFreqT)
    sd = numpy.fabs(avgFreqT - meanFreq)
    order = numpy.argsort(sd)[::-1]
    
    cNoisePoints2 = cNoisePoints.copy()
    maxHurst = numpy.max(cNoisePoints2,axis=1)
    minHurst = numpy.min(cNoisePoints2,axis=1)
    noiseTypes = (cNoisePoints2 - minHurst[:,numpy.newaxis]) / (maxHurst[:,numpy.newaxis]- minHurst[:,numpy.newaxis])
    for i in range(0,nTypes):
        noiseColors = numpy.zeros((points.shape[0],3))
        noiseColors += noiseTypes[i,:,numpy.newaxis]
        figure(constrain_navigation=True,constrain_ratio=True,antialiasing=True)
        plot(points[:,0],points[:,1],primitive_type='POINTS',color=noiseColors,marker='.',marker_size=6)
        show() 
        
    print "Start classification process"
    pointType = classify_final(cNoisePoints,coveragePoints.copy(), heightPoints, basePoints,points,heightCoverage,nocoveragePoints.copy(),unit,order,plantTypeLevels)

#     pointType = classify_final(cNoisePoints,coveragePoints, heightPoints,diffHeight,points,False)

    print "visualize and writing"
    resultMask = pointType > 0;
    fractalValues = cNoisePoints.copy();
    resultTotPoints =  numpy.sum(resultMask);
    maxfract = numpy.max(fractalValues,axis=1);
    fractalValues /= maxfract[:,numpy.newaxis];
    print fractalValues.shape
    selectedFractalValues = fractalValues[:,(pointType-1)];
    print pointType
    print pointType.shape
    print selectedFractalValues.shape;
    print resultTotPoints;
    print numpy.unique(pointType[resultMask]);
    
    results = numpy.zeros((resultTotPoints),dtype=[('x',numpy.float64),('y',numpy.float64),('t',numpy.int64)])
    results['x'] = points[resultMask,0]
    results['y'] = points[resultMask,1]
    results['t'] = pointType[resultMask]
#    results['f'] = selectedFractalValues[resultMask];
    
    pointColors = getColor(pointType[resultMask])

    #numpy.savetxt('locations_ecomodel.txt',results, delimiter=" ", fmt="%s")
    tot2 = numpy.sum(coveragePoints,axis=0) + nocoveragePoints
    tot2[tot2 == 0] = 1
    coveragePoints /= tot2
    tot = float(points.shape[0])
    print numpy.sum(nocoveragePoints) / tot,numpy.sum(pointType == 0) / tot
    print numpy.sum(coveragePoints[0,:]) / tot, numpy.sum(pointType == 1) / tot
    print numpy.sum(coveragePoints[1,:]) / tot, numpy.sum(pointType == 2) / tot
    print numpy.sum(coveragePoints[2,:]) / tot, numpy.sum(pointType == 3) / tot
    
    if PaulinaPolder:
        print numpy.sum(coveragePoints[3,:]) / tot, numpy.sum(pointType == 4) / tot
        print numpy.sum(coveragePoints[4,:]) / tot, numpy.sum(pointType == 5) / tot
        print numpy.sum(coveragePoints[5,:]) / tot, numpy.sum(pointType == 6) / tot
        print numpy.sum(coveragePoints[6,:]) / tot, numpy.sum(pointType == 7) / tot
    
    if PaulinaPolder:
        interval = int(20)
        bins = 31
    else:
        interval = int(5)
        bins = 21
    coverageStatistics = numpy.zeros((nTypes,2,bins))
    for t in range(0,nTypes):
        for i in range(0,bins):
            b1 = i*interval
            b2 = i*interval+interval
            m1 = heightPoints >= b1
            m2 = heightPoints < b2
            m = numpy.logical_and(m1,m2)
            tot = float(numpy.sum(m))
            if tot == 0:
                tot = 1
            coverageStatistics[t,0,i] = numpy.sum(coveragePoints[t,m]) / tot
            coverageStatistics[t,1,i] = numpy.sum(pointType[m] == t+1) / tot
        print coverageStatistics[t,:]
    
    if PaulinaPolder:
        figure(constrain_navigation=True,constrain_ratio=True,antialiasing=True)
        plot(results['x'],results['y'],primitive_type='POINTS',color=pointColors,marker='.',marker_size=6)
        show()
    else:
        figure(constrain_navigation=True,constrain_ratio=False,antialiasing=True)
        plot(results['x'],results['y'],primitive_type='POINTS',color=pointColors,marker='.',marker_size=5)
        show()
main()
