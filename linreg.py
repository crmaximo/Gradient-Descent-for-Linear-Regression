import sys
import numpy as np

def main():
    assert len(sys.argv) <= 5 #up to third degree polynomial only
    
    lnt = len(sys.argv) - 1
    D = float(sys.argv[lnt])
    if lnt < 2:
        C = 0
    else:
        C = float(sys.argv[lnt-1])
    if lnt < 3:
        B = 0
    else:
        B = float(sys.argv[lnt-2])
    if lnt < 4:
        A = 0
    else:
        A = float(sys.argv[lnt-3])

    nsample = 150 #number of data to be generated
    learnrate = 0.0001 #initial learn rate
    x, y, coef = data(A, B, C, D, nsample, 1)
    coef, iteration, loss, learnrate = gradescent(x, y, coef, learnrate, nsample)

    print("iterations: %d" % (iteration))
    print("Solved coefficients:")
    print(coef)
    print("The final learnrate is %f (LR increases every iteration):" %(learnrate))
    #print(loss)

def data (A, B, C, D, npoints, var): #var is used for uniform noise manipulation
    #creating x and y matrix
    x = np.zeros(shape =(npoints, 4)) 
    y = np.zeros(shape = (npoints, 1))

    coef = np.ones(shape = (4, 1)) #initial value (guess)
    func = np.poly1d([A,B,C,D])

    for i in range(0, npoints): #with default increment of 1
        x[i][0] = i ** 3
        x[i][1] = i ** 2
        x[i][2] = i
        x[i][3] = 1
        y[i] = func(i) + np.random.uniform(-1, 1) * var #adding uniform noise

    return x, y, coef
        
def gradescent (x, y, coef, learnrate, samplesize):
    iteration = 0
    while True:
        yprime = np.matmul(x, coef)
        error = yprime - y
        #using M-P pseudoinverse to solve gradient numerically
        xpinv = np.linalg.pinv(x) 
        gradient = np.matmul(xpinv, error) / samplesize
        coef = coef - np.dot(learnrate, gradient)
        while learnrate * 1.05 < 1:
            learnrate = 1.05 * learnrate #5percent increase in learn rate per iteration
        iteration = iteration + 1
        loss = np.sum(error ** 2) / samplesize
        #xnorm = np.linalg.norm(error)#pseudo MARS norm is used
        #Iteraton limit is added to break loop when convergence is unlikely to be achieved due to noise 
        if loss <= 0.40 or iteration == 100000: #loss and iteration can be manipulated to increase precision
            break

    return coef, iteration, loss, learnrate

main()
    
