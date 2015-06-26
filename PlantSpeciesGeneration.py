from galry import *
import numpy
from scipy import interpolate
from scipy import spatial

def getColor(typeList):
    defColors = numpy.array([[0, 0, 0], [0, .5, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

    return defColors[typeList]

def classifyPoints(sortedFractalNoise, fractalNoise, coveragePoints, pointIndices, pointType, points, plantTypes, visualize):
    nTypes = coveragePoints.shape[0]
    totalPoints = pointIndices.shape[0]
    thresholdIndex = (coveragePoints * (totalPoints - 1)).astype(int)

    selectedPositions = numpy.zeros((nTypes, totalPoints), dtype=bool)
    selectedValues = numpy.zeros((nTypes, totalPoints))
    for i in range(0, nTypes):
        thresholdValues = sortedFractalNoise[i, thresholdIndex[i, :]]
        thresholdValues[coveragePoints[i, :] == 0] = numpy.Inf
        selectedPositions[i, :] = fractalNoise[i, :] >= thresholdValues
        selectedValues[i, selectedPositions[i]] = fractalNoise[i, selectedPositions[i]]

    nSelections = numpy.sum(selectedPositions, axis=0)
    selectedPoints = nSelections > 0
    ind = pointIndices[selectedPoints]
    pointType[ind] = plantTypes[numpy.argmax(selectedValues[:, selectedPoints], axis=0).astype(int)]
    if visualize:
        for i in range(0, nTypes):
            pointColors = getColor(selectedPositions[i, :] * plantTypes[i])
            figure(constrain_navigation=True, constrain_ratio=True, antialiasing=True)
            plot(points[pointIndices, 0], points[pointIndices, 1], primitive_type='POINTS', color=pointColors,
                 marker='.', marker_size=6)
            show()

        pointColors = getColor(pointType)
        pointColors[nSelections > 1] = [1, 0, 0]
        figure(constrain_navigation=True, constrain_ratio=True, antialiasing=True)
        plot(points[:, 0], points[:, 1], primitive_type='POINTS', color=pointColors, marker='.', marker_size=6)
        show()

        pointColors = getColor(pointType)
        figure(constrain_navigation=True, constrain_ratio=True, antialiasing=True)
        plot(points[:, 0], points[:, 1], primitive_type='POINTS', color=pointColors, marker='.', marker_size=6)
        show()

    return pointType


def speciesGeneration(fractalNoise, coveragePoints, heightPoints, ndviPoints, points, heightCoverages, noCov, unit, order,
                   plantTypeLevels, plantInfluences=numpy.array([])):
    levels = numpy.unique(points[:, 2]).size
    visualize = True
    tree = spatial.cKDTree(points[:, 0:2])
    nTypes = fractalNoise.shape[0]
    totalPoints = fractalNoise.shape[1]
    pointType = numpy.zeros((totalPoints), dtype=int)
    plantInfluencesPoints = numpy.zeros((nTypes, totalPoints))
    pointIndices = numpy.arange(0, points.shape[0])
    sortedIndices = numpy.argsort(fractalNoise, axis=1)[:, ::-1]
    fractalNoiseSorted = fractalNoise[numpy.arange(nTypes)[:, None], sortedIndices]

    heightBins = numpy.zeros((2, points.shape[0]))
    heightBins[:, :] = (heightPoints / unit)[numpy.newaxis, :]
    heightBins = numpy.trunc(heightBins)
    heightBins[1, :] += 1
    heightBins *= unit
    print heightBins
    weights = (unit - numpy.abs(heightPoints[numpy.newaxis, :] - heightBins)).ravel()
    uq, ui = numpy.unique(heightBins, return_inverse=True)
    heightRange = uq.size

    for l in range(0, levels):
        nocoverage = noCov
        m1 = points[:, 2] <= l
        pTypes = numpy.where(plantTypeLevels == l)[0]
        nTypes = pTypes.size
        orderLevel = pTypes.copy()
        for o in range(0, orderLevel.shape[0]):
            orderLevel[o] = numpy.where(pTypes[o] == order)[0]
        totCov = numpy.sum(coveragePoints[pTypes], axis=0)
        if nTypes == 0:
            break
        m2 = pointType == 0
        m = numpy.logical_and(m1, m2)
        pInd = pointIndices[m]
        fns = fractalNoiseSorted[m[sortedIndices[pTypes, :]]].reshape(nTypes, pInd.shape[0])
        fn = fractalNoise[pTypes[:, np.newaxis], pInd[np.newaxis, :]]
        cov = coveragePoints[pTypes[:, np.newaxis], pInd[np.newaxis, :]]
        pointType = classifyPoints(fns, fn, cov, pInd, pointType, points, pTypes + 1, visualize)

        print "Update coverage values"
        ml = numpy.tile(m, 2)
        uil = ui[ml]
        weightsl = weights[ml]
        totCovPerHeight = numpy.bincount(uil, weights=weightsl, minlength=heightRange)
        totCovPerHeight[totCovPerHeight == 0] = 1
        for i in range(0, nTypes):
            i = pTypes[i]
            maskPT = pointType == i + 1
            maskPTd = numpy.tile(maskPT, 2)
            totCovPerHeightType = numpy.bincount(ui[maskPTd], weights=weights[maskPTd], minlength=heightRange)
            heightCoveragesNew = heightCoverages[i, :] - totCovPerHeightType / totCovPerHeight
            covFunction = interpolate.interp1d(uq, heightCoveragesNew, kind='slinear', bounds_error=False, fill_value=0)
            rest = covFunction(heightPoints)
            rest = numpy.minimum(rest, ndviPoints[i])

            # update step
            covLowerThan0 = rest < 0
            covHigherThan0 = rest > 0
            totMinCov = numpy.sum(rest[covLowerThan0])
            totPlusCov = numpy.sum(rest[covHigherThan0])
            totMinCov /= float(totalPoints)
            totPlusCov /= float(totalPoints)
            rest += plantInfluencesPoints[i, :]

            # Clamp the result between 0 and 1
            rest[rest < 0] = 0
            rest[rest > 1] = 1
            coveragePoints[i, :] = rest

        totCov = numpy.sum(coveragePoints[pTypes, :], axis=0)
        totCov[totCov == 0] = 100.0
        totCov += nocoverage
        nocoverage /= totCov
        coveragePoints[pTypes, :] /= totCov

        print "Classify remaining points"
        iTypes = pTypes.copy()
        for i in range(0, nTypes):
            i = orderLevel[i]
            pt = numpy.take(iTypes, [i])
            i = iTypes[i]
            remainingPoints = numpy.logical_and(m, pointType == 0)
            pInd = pointIndices[remainingPoints]
            if pInd.size == 0:
                break

            nPoints = pInd.shape[0]
            fns = fractalNoiseSorted[i, remainingPoints[sortedIndices[i, :]]].reshape(1, nPoints)
            fn = fractalNoise[i, pInd].reshape(1, nPoints)
            cov = coveragePoints[i, pInd].reshape(1, nPoints)

            pointType = classifyPoints(fns, fn, cov, pInd, pointType, points, pt + 1, visualize)

            mask = pTypes != i
            pTypes = pTypes[mask]
            if pTypes.size != 0:
                totCov = numpy.sum(coveragePoints[pTypes, :], axis=0) + nocoverage
                totCov[totCov == 0] = 100.0
                coveragePoints[pTypes, :] /= totCov
                nocoverage /= totCov

        if plantInfluences.size != 0 or numpy.sum(plantInfluences) != 0:
            print "Apply plant influences"
            for i in range(0, iTypes.size):
                if numpy.sum(plantInfluences[iTypes[i]]) != 0:
                    pos = pointType == iTypes[i] + 1
                    influencedPoints = tree.query_ball_point(points[pos, 0:2], .5)
                    for k in range(0, numpy.sum(pos)):
                        for j in range(0, plantInfluences.shape[1]):
                            plantInfluencesPoints[j, influencedPoints[k]] += plantInfluences[iTypes[i], j]
            coveragePoints += plantInfluencesPoints
            coveragePoints[coveragePoints < 0] = 0

    return pointType
