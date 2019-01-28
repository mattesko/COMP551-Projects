import numpy as np
b = np.array([1.3, 0.7])
c = np.array([[1, 3],[3, 10]])
c_inverse = -np.linalg.inv(c)
b_transpose = b.transpose()
x = np.matmul(b_transpose,c_inverse)
z = np.matmul(x,b)*0.5

0.5*(-0.2471**2)+5*(-0.052**2)+3*-0.2471*-0.052 + 1.3*-0.2471 + 0.7*-0.052 + 1

q = -0.2471 + 3*-0.052 + 1.3
w = 10 *-0.052 + 3*-0.2471 +0.7