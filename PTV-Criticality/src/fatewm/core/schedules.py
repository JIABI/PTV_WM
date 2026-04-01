def should_calibrate(step:int, M:int)->bool:
    return (M is not None) and (M>0) and (step%M==0) and (step>0)
