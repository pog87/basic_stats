import numpy as np

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as np2r

def floatall(x):
    return [float(k) for k in x]

def ignore_nan(x):
    x=floatall(x)
    k=[a for a in x if not np.isnan(a)]
    return k

def ignore_couple_nan(x,y):
    x=floatall(x)
    y=floatall(y)
    k=[i for i in range(0,len(x)) if not np.isnan(x[i]) and not np.isnan(y[i])]
    xx=[x[i] for i in k]
    yy=[y[i] for i in k]
    return (xx,yy)

def check_const(x):
    #check if array contains all the same value
    x=np.array(x)
    xx=x-x[0]*np.ones(x.shape)
    return np.sum(xx**2)==0.

def t_test(x,y):
    '''
    perform a test to assess the difference between two distributions;
    
    t-test if both distributions are normally distributed,
    wilcox-test otherwise
    '''
    xx=np2r.numpy2ri(np.array(ignore_nan(x)))
    yy=np2r.numpy2ri(np.array(ignore_nan(y)))
    if len(xx)<=3 or len(yy)<=3:
        return np.nan
    if check_const(xx) or check_const(yy):
        return np.nan
    sx=robjects.r['shapiro.test'](xx)
    sy=robjects.r['shapiro.test'](yy)
    p_x=sx.rx('p.value')[0][0]
    p_y=sy.rx('p.value')[0][0]
    if p_x>.05 and p_y>.05:
        t=robjects.r['t.test'](xx,yy)
        p=t.rx('p.value')[0][0]
    else:
        w=robjects.r['wilcox.test'](xx,yy)
        p=w.rx('p.value')[0][0]
    return p

def correlaz(a,b):
        '''
    perform a test to assess the correlation between two distributions,
    using sperman method
    '''
    x,y=ignore_couple_nan(a,b)
    if len(x)==0 or len(y)==0:
        return np.nan, np.nan
    if np.mean(x)==0 and np.mean(y)==0:
        return np.nan, np.nan
    xx=np2r.numpy2ri(np.array(x))
    yy=np2r.numpy2ri(np.array(y))
    cx=robjects.r['cor.test'](xx,yy,method="spearman")
    p_val=(cx.rx('p.value')[0][0])
    r_val=(cx.rx('estimate')[0][0])
    return p_val, r_val
