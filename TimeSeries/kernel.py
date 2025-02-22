


import numpy
import scipy

M_PI=3.141592653589793238462643383280
M_SQRT2 = numpy.sqrt(2)

ERFINV_SQRT2= [ 
  1.1631* M_SQRT2,    
  1.2379* M_SQRT2,    
  1.3299* M_SQRT2,    
  1.4522* M_SQRT2,    
  1.6450* M_SQRT2,    
  2.1851* M_SQRT2,    
  2.6297* M_SQRT2
]
PVAL = [ 0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 0.9999 ]


###
## Inner Product: this is an abstraction of the inner product and
## <x,y> but we call it ip(x,y) and and it is related to the norm
## ||x||^2 = <x,x> 
### 

def ip( a : numpy.array, b : numpy.array):
    return numpy.dot(a,b)

###
## ||X||^2
### 
def squareNorm(a : numpy.array, b : numpy.array, ip = ip):
    return ip(a,b)

###
## <a-b, a-b> = <a,a> + <b,b> - <a,b> - <b,a>
## for us is actually  <a,a> + <b,b> - 2<a,b>
##
def squaredNormAminusB(a : numpy.array, b : numpy.array, ip = ip):
    return ip(a,a)+ip(b,b) -2*ip(a,b)


###
## exp(-||a-b||^2/2s^2) 
##
## the sigma is a scaling/normalizing factor and usually is an experimental
## measure, a and b are vector but <a-b,a-b> is a scalar
##
##  K(a,b) = K(b,a)

def rbfKernel(a : numpy.array, 
	      b : numpy.array , 
	      sigmasquare : float , 
	      ip = ip
	      ):

    t = a-b;
    v = ip(t,t)
    return numpy.exp(-v/(2*sigmasquare))


###
## this makes sense only for scalars
###
def pow(x,y): return x**y


###
##       exp(-||a-b||^2/2s^2)/ 2*pi^(N/2)*s
##
##  As above but it is scaled to the number of elemenets of the <x,y>
##
def gaussianKernel(a : numpy.array, 
	           b : numpy.array , 
	           sigmasquare : float , 
	           ip = ip 
	           ):
    
    n = rbfKernel(a,b,sigmasquare,ip)
    d = (2*pow(M_PI,min(a.size,b.size)/2)*numpy.sqrt(sigmasquare)); 
    
    return n/d
  
## Linear and Polynomial K(a,b)

def linearKernel(a : numpy.array, 
	         b : numpy.array , 
	         sigmasquare : float =None , 
	         ip = ip
	         ):
  
  return ip(a,b);
  
def polyKernel(a : numpy.array, 
	       b : numpy.array , 
	       d : float =None , 
	       ip = ip
	       ):
    
         
    return pow(ip(a,b)+1,d)    


##
## ph used to be a feature extraction
##
##

class Hdata:
    def __init__(self, x_i, y_i, x_j, y_j,  ph, K, ip,s=1.0):
        # Sample from A 
        self.x_i = x_i
        self.y_i = y_i
        
        # Sample from B 
        self.x_j = x_j
        self.y_i = y_j
        
        self.ph = ph  
        self.k  = K   # kernel K(a,b) based on ip
        self.ip = ip  # inner product <a,b> = ip(a,b)
        
        self.sigmasquare=s  ## external 
        


def hsimplified_gaussian(hdata : Hdata):
    ## self correlation  
    result   = gaussianKernel(hdata.xi,hdata.yi,hdata.sigmasquare,hdata.ip)
    result  += gaussianKernel(hdata.xj,hdata.yj,hdata.sigmasquare,hdata.ip)
    ## cross correlation
    result  -= 2*gaussianKernel(hdata.xi,hdata.yj,hdata.sigmasquare,hdata.ip)
  
    return result;

def hsimplified_general(hdata : Hdata):
    ## self correlation K(xi, xj) + K(yi,yj)   
    result   = hdata.k(hdata.xi,hdata.yi,hdata.sigmasquare,hdata.ip)
    result  += hdata.k(hdata.xj,hdata.yj,hdata.sigmasquare,hdata.ip)
    ## cross correlation 2*
    result  -= hdata.k(hdata.xi,hdata.yj,hdata.sigmasquare,hdata.ip) 
    result  -= hdata.k(hdata.xj,hdata.yi,hdata.sigmasquare,hdata.ip) 
      
  
    return result;

def h_general(hdata: Hdata):
    
    xi =  hdata.ph(hdata.xi);
    xj =  hdata.ph(hdata.xj);
    yi =  hdata.ph(hdata.yi);
    yj =  hdata.ph(hdata.yj);
    
    l = xi -yi
    r = xj-yj
    
    res = hdata.ip(l,r)
    return result

def MMD_u_g(
        r : list, w : list,
        h , # h above 
        k , # kernel 
        sp  # <x,y>
) -> list :

    hd = Hdata(0,0,0,0,None,k,sp)
    
    N    = (len(r))*(len(w)-1);
    M    = (len(r))*(len(w));

    hi  = numpy.zeros(4*M)
    kxx = numpy.zeros(M)
    kyy = numpy.zeros(M)
    kxy = numpy.zeros(M)
    kyx = numpy.zeros(M)
    mmd2 = 0
    sigma =0
    if (k == rbfKernel): # no parameters I will estimate sigma
        counter =-1
        SIZE = len(w)
        for i in range(len(r)):
            for j in range(len(w)):
                if (i!=j):
                    # remeber each point is a vector !
                    kxx[i*SIZE +j] = squaredNormAminusB(r[i],r[j],sp);
                    kyy[i*SIZE +j] = squaredNormAminusB(w[i],w[j],sp);
                    kxy[i*SIZE +j] = squaredNormAminusB(r[i],w[j],sp);
                    kyx[i*SIZE +j] = squaredNormAminusB(w[i],r[j],sp);
                    if (kxx[i*SIZE +j]):
                        counter +=1; hi[counter]   =  kxx[i*SIZE +j];
                    if (kyy[i*SIZE +j]):
                        counter +=1; hi[counter]   =  kyy[i*SIZE +j];
                    if (kxy[i*SIZE +j]):
                        counter +=1; hi[counter]   =  kxy[i*SIZE +j];
                    if (kyx[i*SIZE +j]):
                        counter +=1; hi[counter]   =  kyx[i*SIZE +j];
        W = sorted(hi[:counter])
        sigma1 =  W[counter//2] if W[counter//2]>0 else 1
        for i in range(len(r)):
            result = 0;
            for j in range(len(w)):
                if i==j: continue
                result += numpy.exp(-kxx[i*SIZE +j]/sigma1) +numpy.exp(-kyy[i*SIZE+j]/sigma1)\
                    -numpy.exp(-kxy[i*SIZE+j]/sigma1)  -numpy.exp(-kyx[i*SIZE+j]/sigma1)
            mmd2 += result/N;
            sigma += result*result;
    else: 
        for i in range(len(r)):
            result = 0;
            for j in range(len(w)):
                if i==j: continue
	            
                hd.xi = r[i];      hd.xj = r[j];
                hd.yi = w[i];      hd.yj = w[j];
                result += h(hd);
            mmd2 += result/N;
            sigma += result*result;
  
    sigma = numpy.sqrt(
        numpy.fabs(
            (4*sigma/N)/N - 4*mmd2*mmd2/len(r)
        )
    )

    
    val = mmd2
    mmd2 = numpy.fabs(mmd2)
    pval = 1
    #import pdb; pdb.set_trace()
    if (mmd2 < sigma*ERFINV_SQRT2[0] or mmd2==0 or sigma==0):
        pval = 0
        return val, pval
    
    for i in range(len(PVAL)):
        
        if (mmd2 <= sigma*ERFINV_SQRT2[i]):
            pval = PVAL[i];
      
    return val, pval
    
    
  
  
def MMD_l_g(r : list , w : list, 
	    h,  k,  sp) -> list  :


    m2  = min(len(r),len(w))//2
    res = [1, 1 ];
    hi  = numpy.zeros(4*m2)
    kxx = numpy.zeros(m2)
    kyy = numpy.zeros(m2)
    kxy = numpy.zeros(m2)
    kyx = numpy.zeros(m2)
    
    sum = 0
    siglin = 0

    if k == rbfKernel:
        counter=-1
        for i in range(m2):
            kxx[i] = squaredNormAminusB(r[ 2*i],r[ +2*i+1],sp)
            kyy[i] = squaredNormAminusB(w[ 2*i],w[ +2*i+1],sp)
            kxy[i] = squaredNormAminusB(r[ 2*i],w[ +2*i+1],sp)
            kyx[i] = squaredNormAminusB(w[ 2*i],r[ +2*i+1],sp)
            
            if (kxx[i]):
                counter +=1; hi[counter]   =  kxx[i]
            if (kyy[i]):
                counter +=1; hi[counter]   =  kyy[i]
            if (kxy[i]):
                counter +=1; hi[counter]   =  kxy[i]
            if (kyx[i]):
                counter +=1; hi[counter]   =  kyx[i]
    
        W = sorted(hi[:counter])
        sigma =  W[counter//2] if W[counter//2]>0 else 1
        sum = 0
        
        for i in range(m2):
          hi[i]= numpy.exp(-kxx[i]/sigma)+ numpy.exp(-kyy[i]/sigma)-  numpy.exp(-kxy[i]/sigma)-  numpy.exp(-kyx[i]/sigma);
          sum += hi[i]
    
          
    else:
        hd = Hdata(0,0,0,0,None,k,sp)
        for i in range(m2):
            hd.xi = r[2*i];      hd.xj = r[2*i+1];
            hd.yi = w[2*i];      hd.yj = w[2*i+1];
            hi[i]= h(hd);
            sum += hi[i];

    mmdlin = sum /m2;
    for i in range(m2):
        t = (hi[i]-mmdlin);
        siglin += t*t;
    
    val = mmdlin;
    siglin = numpy.sqrt(siglin/(m2*(m2-1)));
    pval = 1
    #import pdb; pdb.set_trace()
    if (mmdlin < siglin*ERFINV_SQRT2[0] or mmdlin==0 or siglin==0):
        pval = 0
        return val, pval
    
    for i in range(len(PVAL)):
        
        if (mmdlin <= siglin*ERFINV_SQRT2[i]):
            pval = PVAL[i];
      
    return val, pval



def Kernel(R : list, W : list,
           k, h=hsimplified_general,
           ip = ip,
           method= MMD_l_g):


    
    result = method(R,W,h,k,ip)
    return result



if __name__ == '__main__':



    import matplotlib.pyplot as plt


    if True:
        print("Testing equality")
        QQ = 500
        M = 10
        X = []
        Y = []
        for i in range(QQ):
            X.append(scipy.stats.norm().rvs(M))
            Y.append(scipy.stats.norm().rvs(M))
            #Y.append(numpy.random.rand(M)-0/2)


        for k in [ rbfKernel, gaussianKernel,linearKernel,polyKernel] :
            print(k)
            result = Kernel(X,Y, k=k,
                            h=hsimplified_general,
                            ip = ip,
                            method = MMD_l_g)
            print(result)
            
            result = Kernel(X,Y, k=k,
                            h=hsimplified_general,
                            ip = ip,
                            method = MMD_u_g)
            print(result)

    if True:
        print("Testing inequality")
        QQ = 500
        M = 10
        X = []
        Y = []
        for i in range(QQ):
            X.append(scipy.stats.norm().rvs(M))
            #Y.append(scipy.stats.norm().rvs(M))
            Y.append(numpy.random.rand(M)-0/2)


        for k in [ rbfKernel, gaussianKernel,linearKernel,polyKernel] :
            print(k)
            result = Kernel(X,Y, k=k,
                            h=hsimplified_general,
                            ip = ip,
                            method = MMD_l_g)
            print(result)
            
            result = Kernel(X,Y, k=k,
                            h=hsimplified_general,
                            ip = ip,
                            method = MMD_u_g)
            print(result)
