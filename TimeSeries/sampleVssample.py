###
## You have a set of experiments in the past, say N samples 
## You have a set today, say M sample.
##
## Can we say any thing if M belongs to N?  do M have the same
## properties of N or not ?
##
##
## https://docs.scipy.org/doc/scipy/reference/stats.html
##
## If you like this subject, check the github
## Software
## https://github.com/paolodalberto/FastCDL/
## Paper
## https://github.com/paolodalberto/FastCDL/blob/master/PDF/seriesPaolo.pdf
###


import numpy
import math
import scipy 




###
##  Classics: The M and the N Sample have the same distribution ?
##  Kolmogorov's is bread and butter. This is mostly a comparison
##  about the "average" but using CDF ... I love CDFs and the
##  multivariate can be reduced easily .. "well easily is subjective"
##
###

def KS_use(
        x :numpy.array = numpy.random.rand(10),
        y :numpy.array = numpy.random.rand(10)
):
    # the H0 assumption is equal distribution
    #
    # the statistics is the difference of the CDF based on the data
    # the p-value is the error we commit if we assume that H0 is
    # false. 
    
    statistic, p_value = scipy.stats.ks_2samp(x, y)

    print("K-S statistic:", statistic)
    print("p-value:", p_value) 



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ## Assume we have an history of R sample 
    R  = scipy.stats.gamma(1.2).rvs(100)

    ## N is the sample we have for this experiment
    N  = scipy.stats.norm().rvs(30)

    ## If they have the same distribution they have a similar distance
    ## by KS and the p-value is really the one telling if we have any
    ## confidence in the error small pvalue error in considering H0
    ## false is small and they are different ... Notice the different
    ## number of sample for each .. the historical data can be stored
    ## as CDF directly. 
    KS_use(R,N)

    fig, ax = plt.subplots(1, 1)
    ax.hist(R, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    ax.hist(N, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    plt.show()
    R  = scipy.stats.norm().rvs(100)
    N  = scipy.stats.norm().rvs(10)
    KS_use(R,N)


    fig, ax = plt.subplots(1, 1)
    ax.hist(R, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    ax.hist(N, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    plt.show()










    
