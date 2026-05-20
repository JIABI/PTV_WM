import numpy as np

def bootstrap_ci(values, level=0.95, reps=10000, seed=0):
    rng=np.random.default_rng(seed); arr=np.asarray(values,float)
    boots=np.array([rng.choice(arr,size=len(arr),replace=True).mean() for _ in range(reps)])
    a=(1-level)/2
    return float(np.quantile(boots,a)), float(np.quantile(boots,1-a))

def paired_permutation_test(a,b,direction,reps=100000,seed=0):
    a=np.asarray(a,float); b=np.asarray(b,float); d=a-b
    obs=d.mean(); rng=np.random.default_rng(seed)
    cnt=0
    for _ in range(reps):
        s=rng.choice([-1,1],size=len(d)); val=(d*s).mean()
        if direction=='lower': cnt += (val<=obs)
        else: cnt += (val>=obs)
    return cnt/reps

def holm_adjust(p_values):
    m=len(p_values); idx=np.argsort(p_values); adj=[0]*m
    running=0
    for r,i in enumerate(idx):
        v=(m-r)*p_values[i]; running=max(running,v); adj[i]=min(1.0,running)
    return adj
