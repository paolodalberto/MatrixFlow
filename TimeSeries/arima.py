###
##  The problem of addressing the point x[i], is it good enough with
##  respect of the previous ones x[:-1] ?
##
##  The information we are taking is just one time series until
##  x[i-1]. We are building a very simple ARIMA model, see below and
##  we use to forecast the y[i]. This is basically a moving average
##  with memory. Simple and straightforward. 
##
##  We can use the whole series, or we can use the last few times.
## 
##  If the experiment is a single item like the average of set of
##  experiments: we can do forecast a single item.  x[i] is the
##  average and we create a model for the average.
##
##  If we have a set of experiments, we can forecast a set of results
##  and then compare each ... I personally like the idea to use all
##  samples instead of a summary. Why would you do that ? compare a
##  set of experiments?  Because if the set is large enough we could
##  also deploy N-to-N sample comparison
##  
##  
###



import numpy
import math
import scipy 


from statsmodels.tsa.arima.model import ARIMA
import numpy

###
##  Time series classic y_i = alpha y_(i-1) + x_i + beta x_(i-1)
##  ARIMA
##
##  create a model
##  forecast
##  example
###

def fit_arima(df, p : int = 1,d : int = 1 ,q : int =1) :
    # Fit ARIMA model
    model = ARIMA(df, order=(p, d, q))
    model_fit = model.fit()

    # Summary of the model
    print(model_fit.summary())

    return model_fit, model

def forecast_arima(model, step : int =2):
    cast = model.forecast(step)
    print(cast)
    return cast

def arima_use(
        y :numpy.array = numpy.random.rand(10),
        p : int = 1,d : int = 1 ,q : int =1
): 
    mu = numpy.mean(y)
    var = numpy.var(y)
    std = numpy.sqrt(var)

    print(mu,std)
    
    model = ARIMA(y[:-1], order=(p, d, q))
    model_fit = model.fit()
    
    ## we forecast the last 1 step the last element
    f=list(model_fit.forecast(1).flatten())
    ym2 = y[:-1] + f

    print(f, y[-1])
    if numpy.abs(f[0]-y[-1])>3*std:
        
        print("Warning > 3sigma")
    else:
        print("All is well")


def arima_use_2(
        y :numpy.array = numpy.random.rand(10),
        p : int = 1,d : int = 1 ,q : int =1
): 
    
    model = ARIMA(y[:-1], order=(p, d, q))
    model_fit = model.fit()
    
    ## we forecast the last 1 step the last element
    
    forecast = model_fit.get_forecast(1)
    yhat = forecast.predicted_mean
    yhat_conf_int = forecast.conf_int(alpha=0.05)[0]
    
    print(yhat_conf_int)
    if (yhat_conf_int[0]>y[-1] or yhat_conf_int[1]<y[-1] ):
        print("This is out and is below the 0.05 confidence", y[-1])
    else:
        print("All is well")




if __name__ == '__main__':

    
    R  = numpy.random.rand(10)
    N  = numpy.array([i for i in range(10)])
    
    arima_use_2(y=N+R )
    arima_use(y=N+R )
    
