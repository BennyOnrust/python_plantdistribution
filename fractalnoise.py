import math
import numpy
from numpy.random import RandomState

def generateFractalNoise(points, nTypes, hurst, frequencies):
    totalPoints = points.shape[0]
    frequencies = 1 - frequencies
    cNoise = numpy.zeros((nTypes, totalPoints), dtype=float)
    for i in range(0, nTypes):
        print "type", i
        # Calculate number of octaves
        h = hurst[i];
        f = frequencies[i];
        lac = 1.87;
        f **= 2.2;  # spectralComp
        freq = numpy.min(f);
        offset = 0.0;

        octaves = numpy.log(1 / freq) / numpy.log(lac) + 1
        print octaves
        octavesInt = numpy.floor(octaves).astype(int)
        print octavesInt
        octavesInd = numpy.arange(0, octavesInt + 1)
        exponent_array = numpy.power(lac, numpy.outer(octavesInd, -h))
        freq_array = freq * numpy.power(lac, octavesInd)
        weights = f[numpy.newaxis, :] / freq_array[:, numpy.newaxis]
        weights[weights > 1] **= -.5
        weights **= 2.0

        w = numpy.max(weights / numpy.sum(weights, axis=0), axis=1)
        e = numpy.max(exponent_array / numpy.sum(exponent_array, axis=0), axis=1)
        mW = w < .01
        mE = e < .01
        m = numpy.logical_and(mW, mE)

        if octavesInd[m].size != 0:
            octavesInt = octavesInd[m][0]
            print octavesInt

        print "Generate Simplex noise"
        octaveNoise = numpy.zeros((octavesInt, totalPoints), dtype=float)
        for j in range(0, octavesInt):
            sNoise = ((generateSimplexNoise(points, i, freq_array[j]) + offset) * exponent_array[j])
            octaveNoise[0:j + 1, :] += sNoise

        cumsumExponent = numpy.cumsum(exponent_array[0:octavesInt, :][::-1, :], axis=0)[::-1, :]

        remainder = octaves - octavesInt
        if (remainder > 0):
            octaveNoise += (generateSimplexNoise(points, i, freq_array[octavesInt]) + offset) * exponent_array[
                                                                                                   octavesInt] ** remainder
            cumsumExponent += exponent_array[octavesInt] ** remainder
        octaveNoise /= cumsumExponent

        totWeights = numpy.sum(weights[0:octavesInt], axis=0)
        cNoise[i, :] += numpy.sum(octaveNoise[0:octavesInt] * weights[0:octavesInt], axis=0)
        cNoise[i, :] /= totWeights

    return cNoise

def generateSimplexNoise(points, pType, freq):
    simplexNoise = raw_noise_2d(points * freq, pType)
    simplexNoise /= numpy.sum(simplexNoise)
    simplexNoise *= simplexNoise.shape[0]
    return simplexNoise

def raw_noise_2d(positions, pType):
    seq = numpy.arange(0, 256, dtype=int)
    prng = RandomState(pType)
    prng.shuffle(seq)
    perm = numpy.zeros(512, dtype=int)
    perm[0:256] = seq
    perm[256::] = seq

    grad3 = numpy.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
                         [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
                         [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]])

    nValues = positions.shape[0]

    """2D Raw Simplex noise."""
    # Noise contributions from the three corners
    corners = numpy.zeros((3, 2, nValues))

    # Skew the input space to determine which simplex cell we're in
    F2 = 0.5 * (math.sqrt(3.0) - 1.0)

    # Hairy skew factor for 2D
    s = numpy.sum(positions, axis=1) * F2
    ij = (positions.T + s).astype(int)
    G2 = (3.0 - math.sqrt(3.0)) / 6.0
    t0 = numpy.sum(ij, axis=0) * G2
    # Unskew the cell origin back to (x,y) space
    XY = ij - t0
    # The x,y distances from the cell origin
    corners[0, :, :] = positions.T - XY

    i1 = corners[0, 0, :] > corners[0, 1, :]
    j1 = ~i1
    # A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
    # a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
    corners[1, 0, :] = corners[0, 0, :] - i1 + G2  # Offsets for middle corner in (x,y) unskewed coords
    corners[1, 1, :] = corners[0, 1, :] - j1 + G2
    corners[2, :, :] = corners[0, :, :] - 1.0 + 2.0 * G2  # Offsets for last corner in (x,y) unskewed coords
    # Work out the hashed gradient indices of the three simplex corners
    ij &= 255

    gi = numpy.zeros((3, nValues), dtype=int)
    gi[0, :] = perm[ij[0, :] + perm[ij[1, :]]] % 12
    gi[1, :] = perm[ij[0, :] + i1 + perm[ij[1, :] + j1]] % 12
    gi[2, :] = perm[ij[0, :] + 1 + perm[ij[1, :] + 1]] % 12

    n = numpy.zeros((3, nValues), dtype=float)
    # Calculate the contribution from the three corners    
    temp = corners * corners
    t = .5 - temp[:, 0, :] - temp[:, 1, :]
    m = t >= 0
    t *= t
    t *= t
    grad = grad3[gi]
    # print t
    n[0, m[0, :]] = t[0, m[0, :]] * dot2d(grad[0, m[0, :], :], corners[0, :, m[0, :]])
    n[1, m[1, :]] = t[1, m[1, :]] * dot2d(grad[1, m[1, :], :], corners[1, :, m[1, :]])
    n[2, m[2, :]] = t[2, m[2, :]] * dot2d(grad[2, m[2, :], :], corners[2, :, m[2, :]])

    # Add contributions from each corner to get the final noise value.
    # The result is scaled to return values in the interval [-1,1].
    result = (70.0 * numpy.sum(n, axis=0))
    return (1 + result) * .5


def dot2d(g, c):
    return g[:, 0] * c[:, 0] + g[:, 1] * c[:, 1]


def dot3d(g, c):
    return g[:, 0] * c[:, 0] + g[:, 1] * c[:, 1] + g[:, 2] * c[:, 2]
