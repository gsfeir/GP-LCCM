"""
@name:            Gaussian Process - Latent Class Choice Model (GP-LCCM)
@author:          Georges Sfeir
@summary:         Contains functions necessary for estimating Gaussiam Process latent class choice models
                  using the Expectation Maximization algorithm

            
General References
------------------
This code is based on the latent class choice model (lccm) package which can be downloaded from:
    https://github.com/ferasz/LCCM
This code also relies on some functions from the GaussianProcessClassifier class of sklearn:
   https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier
"""

import gpc
import numpy as np 
import pylogit
from scipy.sparse import coo_matrix
from scipy.optimize import minimize
import scipy.stats
from datetime import datetime
import warnings
from scipy.special import logsumexp
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.utils.extmath import row_norms

import pandas as pd
from collections import OrderedDict
from sklearn import preprocessing
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel, Exponentiation, ExpSineSquared, RationalQuadratic, Product, DotProduct, Sum


# Global variables
emTol = 1e-04
llTol = 1e-06
grTol = 1e-06
maxIters = 10000
np.random.seed(42)

##################################### Gaussian Process #####################################
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder

#### kernel function
kernel = DotProduct()
copy_X_train = True

##################################### Gaussian Process Functions #####################################

def initialize_parameters(X, nClasses):
    """Initialize the parameters of the Class membership model
    """

    n_samples, n_features = X.shape
    TargetClass = np.random.randint(0,2,n_samples)
    return TargetClass
    

def processClassSpecificPanel(dms, dmID, obsID, altID, choice):
    """
    Method that constructs a tuple and three sparse matrices containing information 
    on available observations, and available and chosen alternative
 
    """
    
    nRows = choice.shape[0]
    alts = np.unique(altID)
    nAlts = alts.shape[0]
    obs = np.unique(obsID)
    nObs = obs.shape[0]
    nDms = dms.shape[0]
    
    xAlt, yAlt = np.zeros((nRows)), np.zeros((nRows))
    xChosen, yChosen = np.zeros((nObs)), np.zeros((nObs))
    xObs, yObs = np.zeros((nObs)), np.zeros((nObs))
    xRow, yRow = np.zeros((nRows)), np.zeros((nRows))

    currentRow, currentObs, currentDM = 0, 0, 0    
    for n in dms:
        obs = np.unique(np.extract(dmID == n, obsID))
        for k in obs:      
            xObs[currentObs], yObs[currentObs] = currentObs, currentDM
            cAlts = np.extract((dmID == n) & (obsID == k), altID)        
            for j in cAlts:
                xAlt[currentRow], yAlt[currentRow] = currentRow, currentObs  
                xRow[currentRow], yRow[currentRow] = currentRow, (np.where(dms == n)[0][0] * nAlts) + np.where(alts == j)[0][0]
                if np.extract((dmID == n) & (obsID == k) & (altID == j), choice) == 1:                
                    xChosen[currentObs], yChosen[currentObs] = currentRow, currentObs
                currentRow += 1
            currentObs += 1
        currentDM += 1
            
    altAvTuple = (xAlt, yAlt)
    altChosen = coo_matrix((np.ones((nObs)), (xChosen, yChosen)), shape = (nRows, nObs))
    obsAv = coo_matrix((np.ones((nObs)), (xObs, yObs)), shape = (nObs, nDms))
    rowAv = coo_matrix((np.ones((nRows)), (xRow, yRow)), shape = (nRows, nDms * nAlts))
    
    return altAvTuple, altChosen, obsAv, rowAv
    
    
def imposeCSConstraints(altID, availAlts):
    """
    Method that constrains the choice set for each of the decision-makers across the different
    latent classes following the imposed choice-set by the analyst to each class. 
    Usually, when the data is in longformat, this would not be necessary, since the 
    file would contain rows for only those alternatives that are available. However, 
    in an LCCM, the analyst may wish to impose additional constraints to introduce 
    choice-set heterogeneity.
    
    """   
    altAvVec = np.zeros(altID.shape[0]) != 0   
    for availAlt in availAlts:
        altAvVec = altAvVec | (altID == availAlt)
    return altAvVec.astype(int)



def calClassSpecificProbPanel(param, expVars, altAvMat, altChosen, obsAv):
    """
    Function that calculates the class specific probabilities for each decision-maker in the
    dataset
    
    """
    v = np.dot(param[None, :], expVars)       # v is 1 x nRows
    ev = np.exp(v)                            # ev is 1 x nRows
    ev[np.isinf(ev)] = 1e+20                  # As precaution when exp(v) is too large for machine
    ev[ev < 1e-200] = 1e-200                  # As precaution when exp(v) is too close to zero
    nev = ev * altAvMat                       # nev is 1 x nObs
    nnev = altAvMat * np.transpose(nev)       # nnev is nRows x 1
    p = np.divide(ev, np.transpose(nnev))     # p is 1 x nRows 
    p[np.isinf(p)] = 1e-200                   # When none of the alternatives are available
    pObs = p * altChosen                      # pObs is 1 x nObs
    lPObs = np.log(pObs)                      # lPObs is 1 x nObs
    lPInd = lPObs * obsAv                     # lPInd is 1 x nInds
    return np.exp(lPInd)                      # prob is 1 x nInds

          

def wtLogitPanel(param, expVars, altAv, weightsProb, weightsGr, altChosen, obsAv, choice):
    """
    Function that calculates the log-likelihood function and the gradient for a weighted
    multinomial logit model with panel data. 
    
    """       
    v = np.dot(param[None, :], expVars)         # v is 1 x nRows
    ev = np.exp(v)                              # ev is 1 x nRows
    ev[np.isinf(ev)] = 1e+20                    # As precaution when exp(v) is too large for machine
    ev[ev < 1e-200] = 1e-200                    # As precaution when exp(v) is too close to zero
    nev = ev * altAv                            # nev is 1 x nObs
    nnev = altAv * np.transpose(nev)            # nnev is nRows x 1
    p = np.divide(ev, np.transpose(nnev))       # p is 1 x nRows 
    p[np.isinf(p)] = 1e-200                     # When none of the alternatives are available
    p[p < 1e-200] = 1e-200                      # As precaution when p is too close to zero
    tgr = choice - np.transpose(p)              # ttgr is nRows x 1
    ttgr = -np.multiply(weightsGr, tgr)         # tgr is nRows x 1
    gr = np.dot(expVars, ttgr)                  # gr is nExpVars x 1
    pObs = p * altChosen                        # pObs is 1 x nObs
    lPObs = np.log(pObs)                        # lPObs is 1 x nObs
    lPInd = lPObs * obsAv                       # lPInd is 1 x nInds
    wtLPInd = np.multiply(lPInd, weightsProb)   # wtLPInd is 1 x nInds
    ll = -np.sum(wtLPInd)                       # ll is a scalar
    
    return ll, np.asarray(gr).flatten()
    

def calStdErrWtLogitPanel(param, expVars, altAv, weightsProb, weightsGr, altChosen, obsAv, choice):
    """
    Function that calculates the standard errors for a weighted multinomial logit model 
    with panel data.
       
    """ 
    v = np.dot(param[None, :], expVars)         # v is 1 x nRows
    ev = np.exp(v)                              # ev is 1 x nRows
    ev[np.isinf(ev)] = 1e+20                    # As precaution when exp(v) is too large for machine
    ev[ev < 1e-200] = 1e-200                    # As precaution when exp(v) is too close to zero
    nev = ev * altAv                            # nev is 1 x nObs
    nnev = altAv * np.transpose(nev)            # nnev is nRows x 1
    p = np.divide(ev, np.transpose(nnev))       # p is 1 x nRows 
    p[np.isinf(p)] = 1e-200                     # When none of the alternatives are available
    p[p < 1e-200] = 1e-200                      # As precaution when p is too close to zero
    tgr = choice - np.transpose(p)              # ttgr is nRows x 1
    ttgr = np.multiply(weightsGr, tgr)          # tgr is nRows x 1
    gr = np.tile(ttgr, (1, expVars.shape[0]))   # gr is nRows x nExpVars 
    sgr = np.multiply(np.transpose(expVars),gr) # sgr is nRows x nExpVars 
    hess = np.dot(np.transpose(sgr), sgr)       # hess is nExpVars x nExpVars 
    try:                                        # iHess is nExpVars x nExpVars 
        iHess = np.linalg.inv(hess)             # If hess is non-singular
    except:
        iHess = np.identity(expVars.shape[0])   # If hess is singular
    se = np.sqrt(np.diagonal(iHess))            # se is nExpVars x 1

    return se



def displayOutput(outputFile, startTime, llEstimation, llNull, lml_GP, lml_Choice, llTestNormalized, prediction_test, nClasses, 
        namesExpVarsClassSpec, paramClassSpec, stdErrClassSpec, obsID, X, pIndClass, pChoice): 
    
    num_class_specific_model = 0
    for i in range(0, nClasses):
        num_class_specific_model = num_class_specific_model + paramClassSpec[i].shape[0]

    n_samples, n_features = X.shape


    #Full Model
    rho_squared = 1 - llEstimation/llNull

    #Normalized Model
    a=np.multiply(pChoice, pIndClass.T)
    llNormalized = np.sum(np.log(np.sum(a, axis = 0)))
 
    #Membership and Class-Specific Models
    timeElapsed = datetime.now() - startTime
    timeElapsed = (timeElapsed.days * 24.0 * 60.0) + (timeElapsed.seconds/60.0)
    
    print ("\n")
    print ("Number of Observations:".ljust(45, ' ')),(str(np.unique(obsID).shape[0]).rjust(10,' '))   
    print ("Null Log-Likelihood:".ljust(45, ' ')),(str(round(llNull,2)).rjust(10,' '))   
    print ("Fitted Log-Likelihood:".ljust(45, ' ')),(str(round(llEstimation,2)).rjust(10,' '))
    print ("Rho-Squared:".ljust(45, ' ')),(str(round(rho_squared,2)).rjust(10,' ')) 
    print ("Estimation time (minutes):".ljust(45, ' ')),(str(round(timeElapsed,2)).rjust(10,' ')) 
    print ("\n")

    print ("Normalized Log-Likelihood:".ljust(45, ' ')),(str(round(llNormalized,2)).rjust(10,' '))    
    print ("\n")
    
    # Display screen

    print
    print 'Class-Specific Choice Model:'
    print '-----------------------------------------------------------------------------------------'
    print ("Number of Parameters:".ljust(45,' ')), (str(num_class_specific_model).rjust(10,' '))
    
    for s in range(0, nClasses):
        print
        print 'Class %d Model: ' %(s + 1)
        print '-----------------------------------------------------------------------------------------'
        print 'Variables                                     parameters    std_err     t_stat    p_value'
        print '-----------------------------------------------------------------------------------------'
        for k in range(0, len(namesExpVarsClassSpec[s])):
            print '%-45s %10.4f %10.4f %10.4f %10.4f' %(namesExpVarsClassSpec[s][k], paramClassSpec[s][k], 
                    stdErrClassSpec[s][k], paramClassSpec[s][k]/stdErrClassSpec[s][k], scipy.stats.norm.sf(abs(paramClassSpec[s][k]/stdErrClassSpec[s][k]))*2 )
        print '-----------------------------------------------------------------------------------------'
        
    print ("\n")
    
    if prediction_test == 'Yes':
        print
        print '-----------------------------------------------------------------------------------------'
        print ("Predicted Log-Likelihood:".ljust(45, ' ')),(str(round(llTestNormalized,2)).rjust(10,' '))
        print
    

def processData(inds, indID, nClasses, 
        obsID, altID, choice, availAlts):

    # Class membership model
    nInds = inds.shape[0] ##Check if this is needed in the class specific choice model !!!!!!!!!!!!!!!!!!
    #expVarsClassMem, indClassAv = processClassMem(expVarsClassMem, indID, nClasses, availIndClasses)

    # Class-specific model
    altAvTuple, altChosen, obsAv, rowAv = processClassSpecificPanel(inds, indID, obsID, altID, choice)
    nRows = altID.shape[0]
    nObs = np.unique(obsID).shape[0]

    altAv = []
    for k in range(0, nClasses):
        altAv.append(coo_matrix((imposeCSConstraints(altID, availAlts[k]), 
                (altAvTuple[0], altAvTuple[1])), shape = (nRows, nObs)))
    

    return (nInds, altAv, altChosen, obsAv, rowAv) 

  
def calProb(nClasses, nInds, paramClassSpec, expVarsClassSpec, altAv, altChosen, obsAv, X):
    
    gpc_model = gpc.GaussianProcessClassifier(kernel=kernel).fit(X, q_train_)
    ClassMemProb = gpc_model.predict_proba(X)
    lml_value_ = gpc_model.log_marginal_likelihood_value_
#    f_, f_star, ClassMemProb, lml_value_ = GP_Fit(X)
    #ClassMemProb2 = expit(-(q_train_ * 2 - 1) * f_)

    p = calClassSpecificProbPanel(paramClassSpec[0], expVarsClassSpec[0], altAv[0], altChosen, obsAv)
    for k in range(1, nClasses):
        p = np.vstack((p, calClassSpecificProbPanel(paramClassSpec[k], expVarsClassSpec[k], altAv[k], altChosen, obsAv)))

    Gqnk = np.multiply(p, ClassMemProb.T)
    Gqnk = np.divide(Gqnk, np.tile(np.sum(Gqnk, axis = 0), (nClasses, 1)))     # nClasses x nInds

    ### Marginal log likelihood = log q(Y|X) = log P(Y|X) = log P(Y|f)P(f|X)??
    ll = lml_value_ + np.sum(np.log(np.sum(p, axis = 0)))
    lml_choice_ = np.sum(np.log(np.sum(p, axis = 0)))

    return gpc_model, ClassMemProb, p, Gqnk, ll, lml_value_, lml_choice_ 
 

def enumClassSpecificProbPanel(param, expVars, altAvMat, obsAv, rowAv, nDms, nAlts):
    """
    Function that calculates and enumerates the class specific choice probabilities 
    for each decision-maker in the sample and for each of the available alternatives
    in the choice set.
    
    Parameters
    ----------
    param : 1D numpy array of size nExpVars.
        Contains parameter values.
    expVars : 2D numpy array of size (nExpVars x (nRows)).
        Contains explanatory variables.
    altAvMat : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith 
        column in expVars is available to the decision-maker corresponding to the 
        jth observation, and 0 otherwise.
    obsAv : sparse matrix of size (nObs x nInds).
        The (i, j)th element equals 1 if the ith observation in the dataset corresponds 
        to the jth decision-maker, and 0 otherwise.
    rowAv : sparse matrix of size (nRows x (nAlts * nDms)).
        The (i, ((n - 1) * nAlts) + j)th element of the returned matrix is 1 if the ith row 
        in the data file corresponds to the jth alternative and the nth decision-maker, 
        and 0 otherwise.  
    nDms : Integer.
        Total number of individuals/decision-makers in the dataset.
    nAlts : Integer.
        Total number of unique available alternatives to individuals in the sample.   
        
    Returns
    -------
    pAlt : 2D numpy array of size nInds x nAlts.
        The (i, j)th element of the returned 2D array is denotes the probability 
        of individual i choosing alternative j. 

    
    """ 

    v = np.dot(param[None, :], expVars)               # v is 1 x nRows
    ev = np.exp(v)                                    # ev is 1 x nRows
    ev[np.isinf(ev)] = 1e+20                          # As precaution when exp(v) is too large for machine
    ev[ev < 1e-200] = 1e-200                          # As precaution when exp(v) is too close to zero
    nev = ev * altAvMat                               # nev is 1 x nObs
    nnev = altAvMat * np.transpose(nev)               # nnev is nRows x 1
    p = np.divide(ev, np.transpose(nnev))             # p is 1 x nRows
    p[np.isinf(p)] = 1e-200                           # When none of the alternatives are available
    pAlt = p * rowAv                                  # pAlt is 1 x (nAlts * nDms)
    return pAlt.reshape((nDms, nAlts), order = 'C')


def calClassSpecificProbScenarios(param, expVars, altAvMat, altChosen, obsAv):
    """
    Function that calculates the class specific probabilities for each decision-maker in the
    dataset
    
    Parameters
    ----------
    param : 1D numpy array of size nExpVars.
        Contains parameter values.
    expVars : 2D numpy array of size (nExpVars x (nRows)).
        Contains explanatory variables.
    altAvMat : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith 
        column in expVars is available to the decision-maker corresponding to the 
        jth observation, and 0 otherwise.
    altChosen : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith 
        column in expVars was chosen by the decision-maker corresponding to the 
        jth observation, and 0 otherwise.
    obsAv : sparse matrix of size (nObs x nInds).
        The (i, j)th element equals 1 if the ith observation in the dataset corresponds 
        to the jth decision-maker, and 0 otherwise.
    
    Returns
    -------
    np.exp(lPInd) : 2D numpy array of size 1 x nInds. (k x N)
        Identifies the class specific probabilities for each individual in the 
        dataset.
    """
    v = np.dot(param[None, :], expVars)       # v is 1 x nRows
    ev = np.exp(v)                            # ev is 1 x nRows
    ev[np.isinf(ev)] = 1e+20                  # As precaution when exp(v) is too large for machine
    ev[ev < 1e-200] = 1e-200                  # As precaution when exp(v) is too close to zero
    nev = ev * altAvMat                       # nev is 1 x nObs
    nnev = altAvMat * np.transpose(nev)       # nnev is nRows x 1
    p = np.divide(ev, np.transpose(nnev))     # p is 1 x nRows 
    p[np.isinf(p)] = 1e-200                   # When none of the alternatives are available
    lp = np.log(p)
##    pObs = p * altChosen                      # pObs is 1 x nObs
##    lPObs = np.log(pObs)                      # lPObs is 1 x nObs
##    lPInd = lPObs * obsAv                     # lPInd is 1 x nInds
    return np.exp(lp)                      # prob is 1 x nInds

                                                                                                                                                                                                                                                                                                                                                                                     
def emAlgo(outputFilePath, outputFileName, outputFile, nClasses, X, XTest, prediction_test,
        indID, obsID, altID, choice, indIDTest, obsIDTest, altIDTest, choiceTest, availAlts, expVarsClassSpec, expVarsClassSpecTest, namesExpVarsClassSpec, indWeights, indWeightsTest, paramClassSpec, reg_covar, tol, max_iter):
    
    startTime = datetime.now()
    print 'Processing data'
    outputFile.write('Processing data\n')

    inds = np.unique(indID)
    n_samples, n_features = X.shape
    (nInds, altAv, altChosen, obsAv, rowAv) \
            = processData(inds, indID, 
            nClasses, obsID, altID,
            choice, availAlts) 

    print 'Initializing EM Algorithm...\n'
    outputFile.write('Initializing EM Algorithm...\n\n')

    # Initializing the parameters
    converged, iterCounter, llOld = False, 0, -np.infty


    ########################### Defining Gaussian Process Parameters ###########################
    global X_train_
    global lable_encoder
    global q_train_
    global classes_
    global Gqnk
    global f_star
    global TargetClass
    global TargetClass0
    global gpc_model
    global pTest
    global pChoiceTest
    global ClassMemProbTest
    
    X_train_ = np.copy(X) if copy_X_train else X
    TargetClass = initialize_parameters(X, nClasses)
    TargetClass0 = TargetClass
    # Encode class labels and check that it is a binary classification problem
    label_encoder = LabelEncoder()
    q_train_ = label_encoder.fit_transform(TargetClass)
    classes_ = label_encoder.classes_

    ########################### Defining Gaussian Process Parameters ###########################

   
    # calculating the null log-likelihod
    paramClassSpecNull = []    
    for k in range(0, nClasses):
        paramClassSpecNull.append(np.zeros(expVarsClassSpec[k].shape[0]))
        
    _, _, _, _, llNull, _, _, = calProb(nClasses, nInds, paramClassSpecNull, expVarsClassSpec, altAv, altChosen, obsAv, X)
    
    gpc_model, ClassMemProb, pChoice, Gqnk, llNew, lml_GP, lml_Choice = calProb(nClasses, nInds, paramClassSpec, expVarsClassSpec, altAv, altChosen, obsAv, X)
    
    TargetClass = np.argmax(Gqnk,axis=0)
    
    label_encoder = LabelEncoder()
    q_train_ = label_encoder.fit_transform(TargetClass)
    classes_ = label_encoder.classes_
      
    while not converged:

        a=np.multiply(pChoice, ClassMemProb.T)
        llNormalized = np.sum(np.log(np.sum(a, axis = 0)))
        unique, counts = np.unique(TargetClass, return_counts = True)
        Class1_Per = 100.0*counts[0]/float(len(TargetClass))
        currentTime = datetime.now().strftime('%a, %d %b %Y %H:%M:%S')
        print '<%s> Iteration %d: %.4f' %(currentTime, iterCounter, llNormalized)
        outputFile.write('<%s> Iteration %d: %.4f\n' %(currentTime, iterCounter, llNormalized))
        
        
        #### M-Step (Class-Specific Choice Model) 
        for k in range(0, nClasses):
            cWeights = np.multiply(Gqnk[k, :], indWeights)
            paramClassSpec[k] = minimize(wtLogitPanel, paramClassSpec[k], args = (expVarsClassSpec[k], altAv[k], 
                    cWeights, altAv[k] * obsAv * cWeights[:, None], altChosen, 
                    obsAv, choice), method = 'BFGS', jac = True, tol = llTol, options = {'gtol': grTol})['x']


        gpc_model, ClassMemProb, pChoice, Gqnk, llNew, lml_GP, lml_Choice = calProb(nClasses, nInds, paramClassSpec, expVarsClassSpec, altAv, altChosen, obsAv, X)
        TargetClass = np.argmax(Gqnk,axis=0)
        
        label_encoder = LabelEncoder()
        q_train_ = label_encoder.fit_transform(TargetClass)
        classes_ = label_encoder.classes_

        a=np.multiply(pChoice, ClassMemProb.T)
        llNormalized = np.sum(np.log(np.sum(a, axis = 0)))       

        converged =  (abs(llNormalized - llOld) < emTol)
        llOld = llNormalized
        iterCounter += 1


    # Calculate standard errors for the class specific choice model                                     
    stdErrClassSpec = []
    for k in range(0, nClasses):
        stdErrClassSpec.append(calStdErrWtLogitPanel(paramClassSpec[k], expVarsClassSpec[k], altAv[k], 
                    Gqnk[k, :], altAv[k] * obsAv * Gqnk[k, :][:, None], 
                    altChosen, obsAv, choice))


    gpc_model, ClassMemProb, pChoice, Gqnk, llNew, lml_GP, lml_Choice = calProb(nClasses, nInds, paramClassSpec, expVarsClassSpec, altAv, altChosen, obsAv, X)
    
    llTestNormalized = 0
    pChoiceTest = 0
    
    (nInds, altAv, altChosen, obsAv, rowAv)\
    = processData(inds, indID, nClasses, obsID, altID, choice, availAlts)
    
    nAlts = np.unique(altID).shape[0]
    
    if prediction_test == 'Yes':
        #### Prediction Test
        indsTest = np.unique(indIDTest)
        n_samples_Test, n_features_Test = XTest.shape
        (nIndsTest, altAvTest, altChosenTest, obsAvTest, rowAvTest) = processData(indsTest, indIDTest, nClasses, obsIDTest, altIDTest, choiceTest, availAlts) 
        nAltsTest = np.unique(altIDTest).shape[0]
        
        pChoiceTest = calClassSpecificProbPanel(paramClassSpec[0], expVarsClassSpecTest[0], altAvTest[0], altChosenTest, obsAvTest)
        for k in range(1, nClasses):
            pChoiceTest = np.vstack((pChoiceTest, calClassSpecificProbPanel(paramClassSpec[k], expVarsClassSpecTest[k], altAvTest[k], altChosenTest, obsAvTest)))
                
        ClassMemProbTest = gpc_model.predict_proba(XTest)
        
        aTest=np.multiply(pChoiceTest, ClassMemProbTest.T)
        llTestNormalized = np.sum(np.log(np.sum(aTest, axis = 0)))

        #Sample Enumeration for Test Data
        pTest = enumClassSpecificProbPanel(paramClassSpec[0], expVarsClassSpecTest[0], altAvTest[0], obsAvTest, rowAvTest, nIndsTest, nAltsTest)
        for s in range(1, nClasses):
            pTest = np.hstack((pTest, enumClassSpecificProbPanel(paramClassSpec[s], expVarsClassSpecTest[s], altAvTest[s], obsAvTest, rowAvTest, nIndsTest, nAltsTest)))
        pTest = np.hstack((indsTest[:, None], ClassMemProbTest, pTest))
        pTest = np.hstack((pTest, pChoiceTest.T))
        ### this p will have: first, the class membership probabilities (pIndClassTestNormalized, e.g. P(k=1))
        ###                   Second, the panel (product of probabilities for each individual n) class specific probabilities for each alternative
        ###                   Thired, the panel choice probability per class

        # Choice probability per individual per observarion/scenario per individual
        pScenarioTest = calClassSpecificProbScenarios(paramClassSpec[0], expVarsClassSpecTest[0], altAvTest[0], altChosenTest, obsAvTest)
        for k in range(1, nClasses):
            pScenarioTest = np.vstack((pScenarioTest, calClassSpecificProbScenarios(paramClassSpec[k], expVarsClassSpecTest[k], altAvTest[k], altChosenTest, obsAvTest)))

        np.savetxt(outputFilePath + outputFileName + 'SampleEnumTest.csv', pTest, delimiter = ',')
        np.savetxt(outputFilePath + outputFileName + 'SampleEnumScenarioTest.csv', pScenarioTest, delimiter = ',')

        
    print '\nEnumerating choices for the sample'
    outputFile.write('\nEnumerating choices for the sample\n')
    
    # display model fit results and parameter estimation results            
    displayOutput(outputFile, startTime, llNew, llNull, lml_GP, lml_Choice, llTestNormalized, prediction_test, nClasses, 
            namesExpVarsClassSpec, paramClassSpec, stdErrClassSpec, obsID, X, ClassMemProb, pChoice) 

    # Write parameters to file and store them in an outputfile for the user
    with open(outputFilePath + outputFileName + 'Param.txt', 'wb') as f:                        
        for k in range(0, nClasses):
            np.savetxt(f, paramClassSpec[k][None, :], delimiter = ',')
        #np.savetxt(f, paramClassMem[None, :], delimiter = ',')

def lccm_fit(data,
             X,
             dataTest,
             XTest,
             prediction_test,
             ind_id_col, 
             obs_id_col,
             alt_id_col,
             choice_col,
             n_classes,
             reg_covar,
             tol,
             max_iter,
             class_specific_specs,
             class_specific_labels, 
             indWeights = None,
             avail_classes = None,
             avail_alts = None,
             paramClassSpec = None,
             outputFilePath = '', 
             outputFileName = 'ModelResults'):
    """
    Takes a PyLogit-style dataframe and dict-based specifications, converts them into
    matrices, and invokes emAlgo().
    
    Parameters
    ----------
    data : pandas.DataFrame.
        Labeled data in long format (i.e., each alternative in a choice scenario is in a 
        separate row).
    ind_id_col : String.
        	Name of column identifying the decision maker for each row of data.
    obs_id_col : String.
        	Name of column identifying the observation (choice scenario).
    alt_id_col : String.
        	Name of column identifying the alternative represented.
    choice_col : String.
        	Name of column identifying whether the alternative represented by a row was 
         chosen during the corresponding observation. 
    n_classes : Integer.
        	Number of latent classes to be estimated by the model. 
    class_membership_spec ##Removed##: list of strings
        	List of column names to be used as explanatory variables for the class membership 
         model. If the first element is 'intercept', an intercept will be generated (and 
         any column of data with that name will be lost). 
    class_membership_labels ##Removed##: list of strings, of same length as class_membership_spec
        	Labels for the explanatory variables in the class membership model.
    class_specific_spec : list of OrderedDicts, of length n_classes
        	Each OrderedDict represents the specification for one class-specific choice model.
         Specifications should have keys representing the column names to be used as 
         explanatory variables, and values that are lists of the applicable alternative
         id's. Specs will be passed to pylogit.choice_tools.create_design_matrix().
    class_specific_labels : list of OrderedDicts, of length n_classes
         Each OrderedDict entails the names of explanatory variables for one class-
         specific choice model. Labels should have keys representing the general name
         of the explnatory variable used, and values that are lists of the names of 
         the variable associated with the respective alternative as specified by the analyst.    	
    indWeights : 1D numpy array of size nDms.
        Each element accounts for the associated weight for each individual in the data file
        to cater for the choice based sampling scheme.
    avail_classes ##Removed##: 2D array of size (n_classes x n_rows), optional
    	Which classes are available to which decision-maker? The (i,j)th element equals 1
    	if the ith latent class is available to the decision-maker corresponding to the 
    	jth row of the dataset, and 0 otherwise. If not specified, all classes are
    	available to all decision-makers. (SHOULD THIS GO IN THE DATAFRAME TOO?)
    avail_alts : list of length n_classes, optional
    	Which choice alternatives are available to members of each latent class? The sth
    	element is an array containing identifiers for the alternatives that are available
    	to decision-makers belonging to the sth latent class. If not specified, all
    	alternatives are available to members of all latent classes.
    paramClassMem ##Removed##: 1D numpy array of size nVars x ( nClasses - 1 ).
        Entails parameters of the class memebrship model, excluding those of the first class.
        It treats the first class as the base class and hence no parameters are estimated
        for this class.
    paramClassSpec : List of size nClasses.
        The jth element is a 1D numpy array containing the parameter estimates associated with 
        the explanatory variables entering the class-specific utilities for the jth latent class.
    outputFilePath : str, optional
    	Relative file path for output. If not specified, defaults to 'output/'
    outputFileName : str, optional
    	Basename for output files. If not specified, defaults to 'ModelResults'
    	
    Returns
    -------
    None
    
    """
    outputFile = open(outputFilePath + outputFileName + 'Log.txt', 'w')
    
    # Generate columns representing individual, observation, and alternative id
    # ind_id_col = 'ID'
    # obs_id_col = 'custom_id'
    # alt_id_col = 'mode_id'
    indID = data[ind_id_col].values
    obsID = data[obs_id_col].values
    altID = data[alt_id_col].values
    
    # Generate the choice column and transpose it
    # choice_col = 'choice'
    choice = np.reshape(data[choice_col].values, (data.shape[0], 1))
    
    indIDTest = []
    obsIDTest = []
    altIDTest = []
    choiceTest = []
    
    if prediction_test == 'Yes':
        # Generate columns representing individual, observation, and alternative id for the test datasets
        indIDTest = dataTest[ind_id_col].values
        obsIDTest = dataTest[obs_id_col].values
        altIDTest = dataTest[alt_id_col].values
        # Generate the choice column and transpose it
        choiceTest = np.reshape(dataTest[choice_col].values, (dataTest.shape[0], 1))
        
    # NUMBER OF CLASSES: We could infer this from the number of choice specifications 
    # provided, but it's probably better to make it explicit because that gives us the 
    # option of taking a single choice specification and using it for all the classes (?)
    
    nClasses = n_classes
    
    # AVAILABLE CLASSES: Which latent classes are available to which decision-maker? 
    # 2D array of size (nClasses x nRows) where 1=available i.e. latent class is 
    #available to thee decision-maker in that row of that data and 0 otherwise
    
    
    # AVAILABLE ALTERNATIVES: Which choice alternatives are available to each latent
    # class of decision-makers? List of size nClasses, where each element is a list of
    # identifiers of the alternatives available to members of that class.
    # Default case is to make all alternative available to all decision-makers.
    
    if avail_alts is None:
    	availAlts = [np.unique(altID) for s in class_specific_specs]  
    else:
        availAlts = avail_alts
    
    # CLASS-SPECIFIC MODELS: Use PyLogit to generate design matrices of explanatory variables
    # for each of the class specific choice models, inluding an intercept as specified by the user.
    
    design_matrices = [pylogit.choice_tools.create_design_matrix(data, spec, alt_id_col)[0] 
    						for spec in class_specific_specs]

    expVarsClassSpec = [np.transpose(m) for m in design_matrices]
    
    expVarsClassSpecTest = []
    
    if prediction_test == 'Yes':
        design_matricesTest = [pylogit.choice_tools.create_design_matrix(dataTest, spec, alt_id_col)[0] 
    						for spec in class_specific_specs]
        expVarsClassSpecTest = [np.transpose(m) for m in design_matricesTest]
        
    # NOTE: class-specific choice specifications with explanatory variables that vary
    # by alternative should work automatically thanks to PyLogit, but the output labels 
    # WILL NOT work until we update the LCCM code to handle that. 
    
    # starting values for the parameters of the class specific models
    # making the starting value of the class specfic choice models random
    # in case the user does not specify those starting values.
    if paramClassSpec is None:
        paramClassSpec = []
        for s in range(0, nClasses):
            paramClassSpec.append(-np.random.rand(expVarsClassSpec[s].shape[0])/10)
    
    # weights to account for choice-based sampling
    # By default the weights will be assumed to be equal to one for all individuals unless the user
    # specifies the weights
    # indWeights is 1D numpy array of size nInds accounting for the weight for each individual in the sample
    # as given by the user
    indWeightsTest = []
    if indWeights is None:    
        indWeights = np.ones((np.unique(indID).shape[0]))
        if prediction_test == 'Yes':
            indWeightsTest = np.ones((np.unique(indIDTest).shape[0]))
    
    # defining the names of the explanatory variables for class specific model
    # getting the requried list elements that comprise string of names of
    # explanatory variables to be used in displaying parameter estimates in the output tables.
    namesExpVarsClassSpec = []
    for i in range(0, len(class_specific_labels)):
        name_iterator=[]
        for key, value in class_specific_labels[i].iteritems() :
            if type(value) is list:
                name_iterator += value
            else:
                name_iterator.append(value)
        namesExpVarsClassSpec.append(name_iterator)

    # Invoke emAlgo()
    emAlgo(outputFilePath = outputFilePath, 
           outputFileName = outputFileName, 
           outputFile = outputFile, 
           nClasses = nClasses,
           X = X,
           XTest = XTest,
           prediction_test = prediction_test,
           indID = indID,
           obsID = obsID, 
           altID = altID, 
           choice = choice, 
           indIDTest = indIDTest,
           obsIDTest = obsIDTest,
           altIDTest = altIDTest,
           choiceTest = choiceTest,
           availAlts = availAlts, 
           expVarsClassSpec = expVarsClassSpec,
           expVarsClassSpecTest = expVarsClassSpecTest,
           namesExpVarsClassSpec = namesExpVarsClassSpec, 
           indWeights = indWeights,
           indWeightsTest = indWeightsTest,
           paramClassSpec = paramClassSpec,
           reg_covar = reg_covar,
           tol = tol,
           max_iter = max_iter)
    
    outputFile.close()
    return


