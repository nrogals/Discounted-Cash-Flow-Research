#nonlinear(c) = 
#nonlinear(k + r) =
#linear(c) - p(c) =
#linear(k+r) - p(k+r) =
#linear(r) + linear(k) - p(k)
#= linear(r) - p(k)

"""
I would like to make sure that the penalization term cannot be larger than 10 percent of the 
value of the cash-flow. 

So bascially, we have something like:

nonlinear(c) = linear(k + r) - p(c) = linear(r) - p(c) = linear(r) - alpha * (||k|| /||c||) * linear(r). 
where above we get that: p(c) = alpha * (||k|| /||c||) * linear(r). 

From the above formulation, we have that: 
    1. alpha in [0,1]
    2. ||K|| / ||C|| is [0,1]
    3. nonlinear(c) <= linear(c). 
    4. (||p(c)|| / ||linear(c)||)  <= alpha. 
"""
