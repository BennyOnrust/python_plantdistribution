import math
import numpy
from numpy.random import RandomState

def raw_noise_3d_optimized(x_list,y_list,z_list,pType):
    seq = numpy.arange(0,256,dtype=int)
    prng = RandomState(pType)
    prng.shuffle(seq)
    perm = numpy.zeros((512),dtype=int)
    perm[0:256] = seq
    perm[256::] = seq
#     perm = numpy.array([
#         151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
#         8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
#         35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
#         134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
#         55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
#         18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
#         250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
#         189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
#         172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
#         228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
#         107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
#         138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
#       
#         151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
#         8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
#         35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
#         134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
#         55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
#         18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
#         250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
#         189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
#         172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
#         228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
#         107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
#         138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
#     ])
    
    grad3 = numpy.array([[1,1,0], [-1,1,0], [1,-1,0], [-1,-1,0],
    [1,0,1], [-1,0,1], [1,0,-1], [-1,0,-1],
    [0,1,1], [0,-1,1], [0,1,-1], [0,-1,-1]])
    nValues = x_list.size
    
    # Noise contributions from the five corners
    #n0, n1, n2, n3, n4 = 0.0, 0.0, 0.0, 0.0, 0.0
    coord = numpy.zeros((4,3,nValues))
    
    # The skewing and unskewing factors are hairy again for the 4D case
    F3 = 1.0/3.0 #(math.sqrt(5.0)-1.0) / 4.0
    # Skew the (x,y,z,w) space to determine which cell of 24 simplices we're in
    s_list = (x_list + y_list + z_list) * F3 
    #s = (x + y + z + w) * F4
    # how to apply integer convergence in numpy??!
    i_list = (x_list + s_list).astype(int)
    j_list = (y_list + s_list).astype(int)
    k_list = (z_list + s_list).astype(int)
#     i = int(x + s)
#     j = int(y + s)
#     k = int(z + s)
#     l = int(w + s)

    G3 = 1.0/6.0#(5.0-math.sqrt(5.0)) / 20.0
    t_list = (i_list + j_list + k_list) * G3
    
    #t = (i + j + k + l) * G4
    # Unskew the cell origin back to (x,y,z,w) space
    X0_list = i_list - t_list
    Y0_list = j_list - t_list
    Z0_list = k_list - t_list

#     X0 = i - t
#     Y0 = j - t
#     Z0 = k - t
#     W0 = l - t
    # The x,y,z,w distances from the cell origin
    coord[0,0,:] = x_list - X0_list
    coord[0,1,:] = y_list - Y0_list
    coord[0,2,:] = z_list - Z0_list
    
    # For the 4D case, the simplex is a 4D shape I won't even try to describe.
    # To find out which of the 24 possible simplices we're in, we need to
    # determine the magnitude ordering of x0, y0, z0 and w0.
    # The method below is a good way of finding the ordering of x,y,z,w and
    # then find the correct traversal order for the simplex we're in.
    # First, six pair-wise comparisons are performed between each possible pair
    # of the four coordinates, and the results are used to add up binary bits
    # for an integer index.

        
    # simplex[c] is a 4-vector with the numbers 0, 1, 2 and 3 in some order.
    # Many values of c will never occur, since e.g. x>y>z>w makes x<z, y<w and x<w
    # impossible. Only the 24 indices which have non-zero entries make any sense.
    # We use a thresholding to set the coordinates in turn from the largest magnitude.
    # The number 3 in the "simplex" array is at the position of the largest coordinate.    
    offsets = numpy.zeros((2,3,nValues),dtype=int)
      
    xLy = coord[0,0,:] >= coord[0,1,:]
    xLz = coord[0,0,:] >= coord[0,2,:]
    yLz = coord[0,1,:] >= coord[0,2,:]
    nxLy = numpy.logical_not(xLy)
    nxLz = numpy.logical_not(xLz)
    nyLz = numpy.logical_not(yLz)
    
    mask = numpy.logical_and(xLy,xLz)
    offsets[0,0,mask] = 1
    
#   not(xLy) and yLz
    mask = numpy.logical_and(nxLy,yLz)
    offsets[0,1,mask] = 1 
     
#   not(xLz) and not(yLz)
    mask = numpy.logical_and(nxLz,nyLz)
    offsets[0,2,mask] = 1
    
    #xLz or xLy
    mask = numpy.logical_or(xLz,xLy)
    offsets[1,0,mask] = 1

    mask = numpy.logical_or(nxLy,yLz)    
    #not(xLy) or yLz
    offsets[1,1,mask] = 1

    mask = numpy.logical_or(nxLz,nyLz)    
    #not(xLz) or not(yLz)
    offsets[1,2,mask] = 1
    
    # The fifth corner has all coordinate offsets = 1, so no need to look that up.
    coord[1,:,:] = coord[0,:,:] - offsets[0,:,:] + G3
    coord[2,:,:] = coord[0,:,:] - offsets[1,:,:] + 2.0 * G3
    coord[3,:,:] = coord[0,:,:] - 1.0 + 3.0 * G3

    # Work out the hashed gradient indices of the five simplex corners
    ii_list = i_list.astype(int) & 255
    jj_list = j_list.astype(int) & 255
    kk_list = k_list.astype(int) & 255

    gi0_list = perm[ii_list+perm[jj_list+perm[kk_list]]] % 12
    gi1_list = perm[ii_list+offsets[0,0,:]+perm[jj_list+offsets[0,1,:]+perm[kk_list+offsets[0,2,:]]]] % 12
    gi2_list = perm[ii_list+offsets[1,0,:]+perm[jj_list+offsets[1,1,:]+perm[kk_list+offsets[1,2,:]]]] % 12
    gi3_list = perm[ii_list+1+perm[jj_list+1+perm[kk_list+1]]] % 12
    
#    print gi0_list
    n = numpy.zeros((4,nValues),dtype=float)
    #t = numpy.zeros((100,5))
    # Calculate the contribution from the five corners
    t = 0.6 - coord[:,0,:] * coord[:,0,:] - coord[:,1,:] * coord[:,1,:] - coord[:,2,:] * coord[:,2,:]
    m = t >= 0
    t0 = t[0,m[0,:]]
    t1 = t[1,m[1,:]]
    t2 = t[2,m[2,:]]
    t3 = t[3,m[3,:]]
    n[0,m[0,:]] = t0 * t0 * t0 * t0 * dot3d(grad3[gi0_list[m[0,:]]],coord[0,:,m[0,:]])
    n[1,m[1,:]] = t1 * t1 * t1 * t1 * dot3d(grad3[gi1_list[m[1,:]]],coord[1,:,m[1,:]])
    n[2,m[2,:]] = t2 * t2 * t2 * t2 * dot3d(grad3[gi2_list[m[2,:]]],coord[2,:,m[2,:]])
    n[3,m[3,:]] = t3 * t3 * t3 * t3 * dot3d(grad3[gi3_list[m[3,:]]],coord[3,:,m[3,:]])
    # Sum up and scale the result to cover the range [-1,1]
    return (1 + 32.0 * (n[0,:] + n[1,:] + n[2,:] + n[3,:]))*0.5
    #return numpy.fabs(32.0 * (n[0,:] + n[1,:] + n[2,:] + n[3,:]))

def raw_noise_2d(positions, pType):
    seq = numpy.arange(0,256,dtype=int)
    prng = RandomState(pType)
    prng.shuffle(seq)
    perm = numpy.zeros((512),dtype=int)
    perm[0:256] = seq
    perm[256::] = seq
    
    grad3 = numpy.array([[1,1,0], [-1,1,0], [1,-1,0], [-1,-1,0],
    [1,0,1], [-1,0,1], [1,0,-1], [-1,0,-1],
    [0,1,1], [0,-1,1], [0,1,-1], [0,-1,-1]])
    
    nValues = positions.shape[0]

    """2D Raw Simplex noise."""
    # Noise contributions from the three corners
    corners = numpy.zeros((3,2,nValues))
    
    # Skew the input space to determine which simplex cell we're in
    F2 = 0.5 * (math.sqrt(3.0) - 1.0)
    
    # Hairy skew factor for 2D
    s = numpy.sum(positions,axis=1) * F2
    ij = (positions.T + s).astype(int)
    G2 = (3.0 - math.sqrt(3.0)) / 6.0
    t0 = numpy.sum(ij,axis=0) * G2
    # Unskew the cell origin back to (x,y) space
    
    XY = ij - t0

    # The x,y distances from the cell origin
    corners[0,:,:] = positions.T - XY
    # For the 2D case, the simplex shape is an equilateral triangle.
    # Determine which simplex we are in.
        
    i1 = corners[0,0,:] > corners[0,1,:]
    j1 = ~i1 
    # A step of (1,0) in (i,j) means a step of (1-c,-c) in (x,y), and
    # a step of (0,1) in (i,j) means a step of (-c,1-c) in (x,y), where
    # c = (3-sqrt(3))/6
    corners[1,0,:] = corners[0,0,:] - i1 + G2       # Offsets for middle corner in (x,y) unskewed coords
    corners[1,1,:] = corners[0,1,:] - j1 + G2
    corners[2,:,:] = corners[0,:,:] - 1.0 + 2.0 * G2  # Offsets for last corner in (x,y) unskewed coords
    # Work out the hashed gradient indices of the three simplex corners
    ij &= 255
    
    gi = numpy.zeros((3,nValues),dtype=int)
    gi[0,:] = perm[ij[0,:]+perm[ij[1,:]]] % 12
    gi[1,:] = perm[ij[0,:]+i1+perm[ij[1,:]+j1]] % 12
    gi[2,:] = perm[ij[0,:]+1+perm[ij[1,:]+1]] % 12
    
    n = numpy.zeros((3,nValues),dtype=float)
    # Calculate the contribution from the three corners    
    temp = corners * corners
    t = .5 - temp[:,0,:] - temp[:,1,:]
    m = t >= 0
    t *= t
    t *= t
    grad = grad3[gi]
    #print t
    n[0,m[0,:]] = t[0,m[0,:]] * dot2d(grad[0,m[0,:],:],corners[0,:,m[0,:]])
    n[1,m[1,:]] = t[1,m[1,:]] * dot2d(grad[1,m[1,:],:],corners[1,:,m[1,:]])
    n[2,m[2,:]] = t[2,m[2,:]] * dot2d(grad[2,m[2,:],:],corners[2,:,m[2,:]])
    
#     t0 = 0.5 - x0*x0 - y0*y0
#     if t0 < 0:
#         n0 = 0.0
#     else:
#         t0 *= t0
#         n0 = t0 * t0 * dot2d(grad3[gi0], x0, y0)
# 
#     t1 = 0.5 - x1*x1 - y1*y1
#     if t1 < 0:
#         n1 = 0.0
#     else:
#         t1 *= t1
#         n1 = t1 * t1 * dot2d(grad3[gi1], x1, y1)
# 
#     t2 = 0.5 - x2*x2-y2*y2
#     if t2 < 0:
#         n2 = 0.0
#     else:
#         t2 *= t2
#         n2 = t2 * t2 * dot2d(grad3[gi2], x2, y2)

    # Add contributions from each corner to get the final noise value.
    # The result is scaled to return values in the interval [-1,1].
    result = (70.0 * numpy.sum(n,axis=0))
    return (1+result) * .5

def dot2d(g, c):
    return g[:,0]*c[:,0] + g[:,1]*c[:,1]

def dot3d(g, c):
    return g[:,0]*c[:,0] + g[:,1]*c[:,1] + g[:,2]*c[:,2]