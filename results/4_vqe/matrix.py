import sys
import math
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


# I store in a dictionary the equivalence between the
# total state and the occupancy notation in the low dimension
# derivation
states2 = dict()
states2[0] = [1, 1, 1, 1]
states2[1] = [1, 1, 1, 0]
states2[2] = [1, 1, 0, 1]
states2[3] = [1, 1, 0, 0]
states2[4] = [1, 0, 1, 1]
states2[5] = [1, 0, 1, 0]
states2[6] = [1, 0, 0, 1]
states2[7] = [1, 0, 0, 0]
states2[8] = [0, 1, 1, 1]
states2[9] = [0, 1, 1, 0]
states2[10] = [0, 1, 0, 1]
states2[11] = [0, 1, 0, 0]
states2[12] = [0, 0, 1, 1]
states2[13] = [0, 0, 1, 0]
states2[14] = [0, 0, 0, 1]
states2[15] = [0, 0, 0, 0]

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

# This function gives you the state number for a given 
# occupancy state. For example: getState2([1,1,0,0]) 
# returns 3
def getState2(a):
    for s, o in states2.items():
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

# Builds the full double particle Hamiltonian
def buildDoubleParticleH(h, A, Ap):

    H = np.zeros((int(math.pow(2,N)), int(math.pow(2,N))))
    for i in range(0, N):
        for j in range(0, N):
            for k in range(0, N):
                for l in range(0, N):
                    H = H + h[i, j, k, l] * np.matmul(Ap[i], np.matmul(Ap[j], np.matmul(A[k], A[l])))

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



########################################################
#################Matrix calculations####################
######## with lower dimension representation ###########
########################################################

# Builds the full A+j matrix
def LmakeMatrixAp(j):

    Apbase = np.asarray([[0,1],[0,0]])
    Zbase = np.asarray([[1,0],[0, -1]])
    Ibase = np.eye(2)

    Ap = Zbase.copy()
    if j == 0:
        Ap = Apbase.copy()
    
    for i in range(1, N):
        if i < j:
            Ap = np.kron(Ap, Zbase)
        elif i == j:
            Ap = np.kron(Ap, Apbase)
        else:
            Ap = np.kron(Ap, Ibase)
    return Ap


# Builds the full A+j matrix
def LmakeMatrixA(j):

    Abase = np.asarray([[0,0],[1,0]])
    Zbase = np.asarray([[1,0],[0, -1]])
    Ibase = np.eye(2)

    A = Zbase.copy()
    if j == 0:
        A = Abase.copy()
    
    for i in range(1, N):
        if i < j:
            A = np.kron(A, Zbase)
        elif i == j:
            A = np.kron(A, Abase)
        else:
            A = np.kron(A, Ibase)
    return A



# Builds the full single particle Hamiltonian
def LbuildSingleParticleH(h, A, Ap):

    H = np.zeros((int(math.pow(2,N)), int(math.pow(2,N))))
    for i in range(0, N):
        for j in range(0, N):
            H = H + h[i, j] * np.matmul(Ap[i], A[j])

    return H


# Builds the full double particle Hamiltonian
def LbuildDoubleParticleH(h, A, Ap):

    H = np.zeros((int(math.pow(2,N)), int(math.pow(2,N))))
    for i in range(0, N):
        for j in range(0, N):
            for k in range(0, N):
                for l in range(0, N):
                    matrix = np.matmul(Ap[i], np.matmul(Ap[j], np.matmul(A[k], A[l])))
                    H = H + h[i, j, k, l] * np.matmul(Ap[i], np.matmul(Ap[j], np.matmul(A[k], A[l])))

    return H


# Builds the single particle Hamiltonian restricted to
# the interesting states given as a vector of states
def LbuildSingleParticleHRestricted(H, interestingStates):

    n = len(interestingStates)
    Hres = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            statei = np.asarray(interestingStates[i])
            statej = np.asarray(interestingStates[j])
            sis = getState2(statei)
            sjs = getState2(statej)
            si = getStateGlobalBasis(sis)
            sj = getStateGlobalBasis(sjs)
            Hres[i, j] = np.matmul(si, np.matmul(H, sj))
    return Hres


# Creates a vector with the A and A+ operators
def LcreateOperators():

    A = []
    Ap = []
    for i in range(0, N):
        A.append(LmakeMatrixA(i))
        Ap.append(LmakeMatrixAp(i))
    return A, Ap


########################################################
########################################################


########################################################
#                   Build hpqrs                        #
########################################################

def ZeroSpinProduct(phi1, phi2):

    ######
    #psi0 = n1 x |+>
    #psi1 = n1 x |->
    #psi2 = n2 x |+>
    #psi3 = n2 x |->

    if phi1 == phi2 or phi1 == (phi2 + 2)%4:
        return False
    else:
        return True


def makeIntegralHPQ(L, n1, n2):

    if n1 != n2:
        c = L * ((n1 * np.cos(np.pi * n1) * np.sin(np.pi * n2) - np.sin(np.pi * n1) * n2 * np.cos(np.pi * n2)) * np.pi * (n2 - n1) * (n2 + n1)) /\
            (np.pi * (n2 - n1) * (n2 + n1))
        
    if n1 == n2:
        n = n1
        c = - L * (np.sin(2 * np.pi * n) - 2 * np.pi * n) / (4 * np.pi * n)

    return 1/L * (n2 * np.pi / L)**2 * c


def makeIntegralGPQRS(V0, L, n1, n2, n3, n4):

    c = 0
    if n1 - n2 + n3 - n4 == 0:
        c += 1
    if n1 - n2 - n3 + n4 == 0:
        c += 1
    if n1 - n2 + n3 + n4 == 0:
        c -= 1
    if n1 - n2 - n3 - n4 == 0:
        c -= 1
    if n1 + n2 + n3 - n4 == 0:
        c -= 1
    if n1 + n2 - n3 + n4 == 0:
        c -= 1
    if n1 + n2 + n3 + n4 == 0:
        c += 1
    if n1 + n2 - n3 - n4 == 0:
        c += 1
    
    return V0 * 4.0/(L*L) * c/8.0


def returnk(n):

    if n == 0 or n == 1:
        return 1
    else:
        return 2


def buildHPQ(L):

    hpq = np.zeros((N, N))
    for phi1 in range(0, N):
        for phi2 in range(0, N):
                if ZeroSpinProduct(phi1, phi2):
                    continue
                hpq[phi1, phi2] = makeIntegralHPQ(L, returnk(phi1), returnk(phi2))
    return hpq


def buildGPQRS(V0, L):

    gpqrs = np.zeros((N, N, N, N))
    for phi1 in range(0, N):
        for phi2 in range(0, N):
            for phi3 in range(0, N):
                for phi4 in range(0, N):
                    if ZeroSpinProduct(phi1, phi4) or ZeroSpinProduct(phi2, phi3):
                        continue
                    gpqrs[phi1, phi2, phi3, phi4] = makeIntegralGPQRS(V0, L, returnk(phi1), returnk(phi2), returnk(phi3), returnk(phi4))
    return gpqrs



if __name__ == "__main__":

    # Potential well length
    L = 1

    # We create the hpq matrix
    hpq = buildHPQ(L)

    # We create the hpqrs matrix
    hpqrs = buildGPQRS(1, L)

    #################################################
    # We work in the high dimension representation  #
    #################################################
        
    # We create the operators 
    A, Ap = createOperators()
    
        
    # We build the full single particle Hamiltonian
    Hsingle = buildSingleParticleH(hpq, A, Ap)
    
    Hdouble = buildDoubleParticleH(hpqrs, A, Ap)

    H = Hsingle + 1/2 * Hdouble

    # We restrict to the states in which we are interested in
    interestingStates = [5, 6, 7, 8, 9, 10]
    Hres = buildSingleParticleHRestricted(H, interestingStates)
    print('Htotal using the high dimension representation')
    print(H)
    print('Hrestricted using the high dimension representation')
    print(Hres)

    print('========================================================')
    #################################################
    # We work in the low dimension representation   #
    #################################################
        
    # We create the operators 
    A, Ap = LcreateOperators()

    # We build the full single particle Hamiltonian
    Hsingle = LbuildSingleParticleH(hpq, A, Ap)
        
    # We build the full double particle Hamiltonian
    Hdouble = LbuildDoubleParticleH(hpqrs, A, Ap)

    H = Hsingle + 1/2 * Hdouble
        
    # We restrict to the states in which we are interested in
    interestingStates = [states[5], states[6], states[7], states[8], states[9], states[10]]
    Hres = LbuildSingleParticleHRestricted(H, interestingStates)

    print('Htotal using the low dimension representation')
    print(H)
    print('Hrestricted using the low dimension representation')
    print(Hres)
