import numpy as np

def threestats(X):
    """Third order statistics.   
    
       TODO:  Improve the memory signature of this code (rewrite in cython?)    
    
       input:
        
       X:  input array (n_filters, n_channels)
        
       returns:
       
       Z:  array (n_filters, n_filters, n_filters)
       
           Z[i, j, k] = num / denom
           
           where 
            
           num = ((X[i] - X[i].mean()) * (X[j] - X[j].mean()) * (X[k] - X[k].mean())).mean()
           
           denom = X[i].std() * X[j].std() * X[k].std()

    
    """
    s0, s1 = X.shape
    s = X.std(1)
    s = np.outer(s, np.outer(s,s)).reshape((s0, s0, s0))
    X = X - X.mean(1)[:, np.newaxis]
    Y = np.kron(X, X).reshape((s0**2, s1, s1))[:, range(s1), range(s1)]
    Y = np.kron(X, Y).reshape((s0**3, s1, s1))[:, range(s1), range(s1)]
    Y = Y.mean(1).reshape((s0, s0, s0))
    return Y / s



def threestats_mem(X):
    """Third order statistics.   
    
       TODO:  Improve the memory signature of this code (rewrite in cython?)    
    
       input:
        
       X:  input array (n_filters, n_channels)
        
       returns:
       
       Z:  array (n_filters, n_filters, n_filters)
       
           Z[i, j, k] = num / denom
           
           where 
            
           num = ((X[i] - X[i].mean()) * (X[j] - X[j].mean()) * (X[k] - X[k].mean())).mean()
           
           denom = X[i].std() * X[j].std() * X[k].std()

    
    """
    s0, s1 = X.shape
    s = X.std(1)
    s = np.outer(s, np.outer(s,s)).reshape((s0, s0, s0))
    X = X - X.mean(1)[:, np.newaxis]
    Y = np.array([np.outer(X[:, i],
                  np.outer(X[:, i],
                  X[:, i])).reshape((s0, s0, s0)) for i in range(s1)]).mean(0)
    return Y / s