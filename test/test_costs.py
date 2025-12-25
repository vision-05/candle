import candle.costs as c
import numpy as np

cost = c.MSE()
c = cost.compute_cost(np.array([[1,2,3,4]]), np.array([[2.5,3,3,4]]), 2)

print(c)