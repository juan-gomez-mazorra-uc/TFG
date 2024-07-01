import numpy as np



########################################################
####################Global definitions##################
########################################################


# Number of states in the individual Hamiltonian
N = 4

# I store in a dictionary the equivalence between the
# total state and the occupancy notation
states = dict()
states[0] = [0, 0, 0, 0]
states[1] = [1, 0, 0, 0]
states[2] = [0, 1, 0, 0]
states[3] = [0, 0, 1, 0]
states[4] = [0, 0, 0, 1]
states[5] = [1, 1, 0, 0]
states[6] = [1, 0, 1, 0]
states[7] = [1, 0, 0, 1]
states[8] = [0, 1, 1, 0]
states[9] = [0, 1, 0, 1]
states[10] = [0, 0, 1, 1]
states[11] = [1, 1, 1, 0]
states[12] = [1, 1, 0, 1]
states[13] = [1, 0, 1, 1]
states[14] = [0, 1, 1, 1]
states[15] = [1, 1, 1, 1]

########################################################
########################################################



########################################################
####################Helper functions####################
########################################################

# This function gives you the state number for a given 
# occupancy state. For example: getState([1,1,0,0]) 
# returns 5
def getState(a):
    for s, o in states.items():
        sameState = True
        for i, j in enumerate(a):
            if j != o[i]:
                sameState= False
        if sameState:
            return s

# This function returns a vector associated to the i
# state. For example: getStateGlobalBasis(1) =
# [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def getStateGlobalBasis(i):

    s = np.zeros(N**2)
    s[i] = 1
    return s

########################################################
########################################################


########################################################
#################Matrix calculations####################
########################################################


# Applys the operator A+j on the state 'state'
def applyAp(j, state):

    newstatevector = states[state].copy()
    #Estimate the sign which is the sum of the numbers of 1 before j
    sumones = 0
    for i in range(0, j):
        sumones = sumones + states[state][i]
    sign = 1
    if sumones % 2 != 0:
        sign = -1

    if states[state][j] == 0:
        newstatevector[j] = 1
        newstate = getState(newstatevector)
    else:
        sign = 0
        newstate = 0
    vector = np.zeros(N**2, dtype=np.int32)
    if sign == 0:
        return vector
    else:
        vector[newstate] = 1
        return sign * vector
    return vector


# Applys the operator Aj on the state 'state'
def applyA(j, state):

    newstatevector = states[state].copy()
    #Estimate the sign which is the sum of the numbers of 1 before j
    sumones = 0
    for i in range(0, j):
        sumones = sumones + states[state][i]
    sign = 1
    if sumones % 2 != 0:
        sign = -1

    if states[state][j] == 1:
        newstatevector[j] = 0
        newstate = getState(newstatevector)
    else:
        sign = 0
        newstate = 0
    vector = np.zeros(N**2, dtype=np.int32)
    if sign == 0:
        return vector
    else:
        vector[newstate] = 1
        return sign * vector
    return vector



# Builds the full A+j matrix
def makeMatrixAp(j):

    mat = applyAp(j, 0)
    for i in range(1, N**2):
        mat = np.vstack((mat, applyAp(j, i)))
    return mat.T



# Builds the full A+j matrix
def makeMatrixA(j):

    mat = applyA(j, 0)
    for i in range(1, N**2):
        mat = np.vstack((mat, applyA(j, i)))
    return mat.T


# Builds the full single particle Hamiltonian
def buildSingleParticleH(h, A, Ap):

    H = np.zeros((N*N, N*N))
    for i in range(0, N):
        for j in range(0, N):
            H = H + h[i, j] * np.matmul(Ap[i], A[j])

    return H


# Builds the single particle Hamiltonian restricted to
# the interesting states given as a vector of states
def buildSingleParticleHRestricted(H, interestingStates):

    n = len(interestingStates)
    
    Hres = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            statei = interestingStates[i]
            statej = interestingStates[j]
            si = getStateGlobalBasis(statei)
            sj = getStateGlobalBasis(statej)
            Hres[i, j] = np.matmul(si, np.matmul(H, sj))

    return Hres


# Creates a vector with the A and A+ operators
def createOperators():

    A = []
    Ap = []
    for i in range(0, N):
        A.append(makeMatrixA(i))
        Ap.append(makeMatrixAp(i))
    return A, Ap


########################################################
########################################################




if __name__=='__main__':


    #We create the operators 
    A, Ap = createOperators()
   
    #We create the hpq matrix
    hpq = np.zeros((N,N))
    hpq[0, 0] = 1.
    hpq[1, 1] = 1.
    hpq[2, 2] = 5.
    hpq[3, 3] = 5.


    #We create the hpqrs matrix
    hpqrs = np.zeros((N,N,N,N))
    #hpqrs[0, 0, 0, 0] = ...
    #.....
    #.....


    #We build the full single particle Hamiltonian
    Hsingle = buildSingleParticleH(hpq, A, Ap)
    
    #We build the full double particle Hamiltonian
    #Hdouble = buildDoubleParticleH(hpqrs, A, Ap)

    #H = Hsingle + Hdouble
    H = Hsingle

    #We restrict to the states in which we are interested in
    interestingStates = [5, 6, 7, 8, 9, 10]
    Hres = buildSingleParticleHRestricted(H, interestingStates)

    print(Hres)





















