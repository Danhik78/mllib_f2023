import math


def poly_funs(dim:int = 1,max_pow:int = 1) -> list:
    f = [lambda x,i_=0: 1]
    f.extend([ lambda x,i_=i+dim: x[i_%dim]**(i_/dim) for i in range(max_pow*dim)])
    return f
def exp_funs(dim:int =1, max_pow:int =1) -> list:
    f = [lambda x,i_=0: 1]
    f.extend([lambda x,i_=i+dim: math.exp(x[i_%dim]*(i_/dim)) for i in range(max_pow*dim)])
    return f
def sin_funs(dim:int = 1,max_pow:int = 1) ->list:
    f = [lambda x,i_=0: 1]
    f.extend([ lambda x,i_=i+dim: math.sin(x[i_%dim]*(i_/dim)) for i in range(max_pow*dim)])
    return f