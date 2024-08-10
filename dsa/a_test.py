import numpy as np 

def myfun(a: object):
    print(type(a))
    print(a)

test_x=np.array([0.0,1.0,0.0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0]).reshape(1, -1)
myfun(test_x)
