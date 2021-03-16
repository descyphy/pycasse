import numpy as np

M = 10**4

class lin_dyn:
    """
    A linear dynamics class for a discrete-time state-space model with process and measurement noise.
    The process and measurement noises defaults to "None" if not specified.

    x[t+1] = A x[t] + B u[t] + w[t] where w[t] ~ Normal(0, Q)
    y[t] = C x[t] + D u[t] + v[t] where v[t] ~ Normal(0, R)

    x[t] is a vector of system states
    u[t] is a vector of system inputs
    y[t] is a vector of measured system outputs
    w[t] is the Gaussian process noise with zero mean and the covariance matrix Q
    v[t] is the Gaussian measurement noise with zero mean and the covariance matrix R
    """

    def __init__(self, x_len=None, u_len=None, y_len=None, x_bounds=None, u_bounds=None, y_bounds=None, x0=None, A=None, B=None, C=None, D=None, Q=None, R=None):	
        """
        Constructor method.
        """	
        # Set length of x, u, and y
        if x_len is not None:
            self.x_len = x_len
        else: 
            self.x_len = None
        if u_len is not None:
            self.u_len = u_len
        else: 
            self.u_len = None
        if y_len is not None:
            self.y_len = y_len
        else: 
            self.y_len = None

        # Check the dimensions of x_bounds
        if x_bounds is not None:
            x_bounds_row, x_bounds_col = x_bounds.shape
            if x_bounds_row != x_len or x_bounds_col != 2: 
                raise Exception("The bounds for the system states x has wrong dimensions. It should be a len(x)-by-2 matrix.\n")
            else:
                self.x_bounds = x_bounds
        elif x_len is not None:
            self.x_bounds = np.zeros((x_len,2), dtype=np.int)
            self.x_bounds[:, 0] =  -M
            self.x_bounds[:, 1] =  M

        # Check the dimensions of u_bounds
        if u_bounds is not None:
            u_bounds_row, u_bounds_col = u_bounds.shape
            if u_bounds_row != u_len or u_bounds_col != 2: 
                raise Exception("The bounds for the system states u has wrong dimensions. It should be a len(u)-by-2 matrix.\n")
            else:
                self.u_bounds = u_bounds
        elif u_len is not None:
            self.u_bounds = np.zeros((u_len,2), dtype=np.int)
            self.u_bounds[:, 0] =  -M
            self.u_bounds[:, 1] =  M

        # Check the dimensions of y_bounds
        if y_bounds is not None:
            y_bounds_row, y_bounds_col = y_bounds.shape
            if y_bounds_row != y_len or y_bounds_col != 2: 
                raise Exception("The bounds for the system states y has wrong dimensions. It should be a len(y)-by-2 matrix.\n")
            else:
                self.y_bounds = y_bounds
        elif y_len is not None:
            self.y_bounds = np.zeros((y_len,2), dtype=np.int)
            self.y_bounds[:, 0] =  -M
            self.y_bounds[:, 1] =  M

        # Check the dimensions of x0
        x0_row, x0_col = x0.shape
        if x0_row != x_len or x0_col != 1: 
            raise Exception("The initial system states x0 has wrong dimensions. It should be a len(x)-by-1 matrix.\n")
        else:
            self.x0 = x0

        # Check the dimensions of the matrix A
        if A is not None:
            A_row, A_col = A.shape
            if A_row != x_len or A_col != x_len: 
                raise Exception("Matrix A has wrong dimensions. It should be a len(x)-by-len(x) matrix.\n")
            else:
                self.A = A

        # Check the dimensions of the matrix B
        if B is not None:
            B_row, B_col = B.shape
            if B_row != x_len or B_col != u_len: 
                raise Exception("Matrix B has wrong dimensions. It should be a len(x)-by-len(u) matrix.\n")
            else:
                self.B = B

        # Check the dimensions of the matrix C
        if C is not None:
            C_row, C_col = C.shape
            if C_row != y_len or C_col != x_len: 
                raise Exception("Matrix C has wrong dimensions. It should be a len(y)-by-len(x) matrix.\n")
            else:
                self.C = C

        # Check the dimensions of the matrix D
        if D is not None:
            D_row, D_col = D.shape
            if D_row != y_len or D_col != u_len: 
                raise Exception("Matrix D has wrong dimensions. It should be a len(y)-by-len(u) matrix.\n")
            else:
                self.D = D

        # Check the dimensions of the matrix Q
        if Q is not None:
            Q_row, Q_col = Q.shape
            if Q_row != x_len or Q_col != x_len: 
                raise Exception("Matrix Q has wrong dimensions. It should be a len(x)-by-len(x) matrix.\n")
            else:
                self.Q = Q

        # Check the dimensions of the matrix R
        if R is not None:
            R_row, R_col = R.shape
            if R_row != y_len or R_col != y_len: 
                raise Exception("Matrix R has wrong dimensions. It should be a len(y)-by-len(y) matrix.\n")
            else:
                self.R = R