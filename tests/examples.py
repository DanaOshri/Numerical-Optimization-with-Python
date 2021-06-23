import numpy as np

'''
----------------------------------------get_q-----------------------------------------------------

Q1 =    [1   0]
        [0   1]
Q2 =    [5   0]
        [0   1]
Q3 =    [√3/2   −0.5]T    [5  0]    [√3/2   −0.5]
        [0.5     √3/2]    [0  1]    [0.5    √3/2]

'''


def get_q(index):
    if index == 1:
        return np.array([[1, 0], [0, 1]])

    if index == 2:
        return np.array([[5, 0], [0, 1]])

    else:
        a = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
        b = np.array([[5, 0], [0, 1]])

        ba = np.matmul(b, a)
        return np.matmul(np.transpose(a), ba)


'''
----------------------------------------quadratic-----------------------------------------------------

The function take a vector x and return two output values: 
1. The scalar function value evaluated at x 
2. The vector valued gradient at x.

The quadratic function is f(x) = xT Q x
'''

def quadratic_1(x, hess_flag=False):
    return quadratic(x, 1, hess_flag)

def quadratic_2(x, hess_flag=False):
    return quadratic(x, 2, hess_flag)

def quadratic_3(x, hess_flag=False):
    return quadratic(x, 3, hess_flag)

def quadratic(x, index, hess_flag=False):
    q = get_q(index)

    # Calculate f(x) = (xT)Qx
    qx = np.matmul(q, x)
    f_x = np.matmul(np.transpose(x), qx)

    # The derivative of f(x) is 2Qx
    df_x = 2*qx

    if hess_flag:
        hess = q + q.T
    else:
        hess = np.zeros((df_x.shape[0], df_x.shape[0]))

    return f_x, df_x, hess


'''
----------------------------------------quadratic-----------------------------------------------------

The function take a vector x and return two output values: 
1. The scalar function value evaluated at x 
2. The vector valued gradient at x.

The Rosenbrock function: f(x) = 100(x_2 - x_1^2)^2 + (1-x_1)^2
'''


def rosenbrock(x, hess_flag=False):
    x_1 = x[0][0]
    x_2 = x[1][0]
    f_x = (100*np.power((x_2 - np.power(x_1, 2)), 2)) + np.power(1-x_1, 2)

    df_x = np.array([[400*np.power(x_1, 3) - (400*x_1*x_2) + (2*x_1) - 2],
                     [(-200)*np.power(x_1, 2) + (200*x_2)]])

    if hess_flag:
        hess = np.array([[1200*np.power(x_1, 2) - (400*x_2) + 2, (-400)*x_1], [(-400)*x_1, 200]])
    else:
        hess = np.zeros((df_x.shape[0], df_x.shape[0]))

    return f_x, df_x, hess


'''
----------------------------------------linear-----------------------------------------------------

The function take a vector x and return two output values: 
1. The scalar function value evaluated at x 
2. The vector valued gradient at x.

The linear function: f(x) = aTx
'''


def linear(x, hess_flag=False):
    a = np.array([[1], [2]])
    f_x = np.matmul(a.T, x)
    df_x = a

    hess = np.zeros((df_x.shape[0], df_x.shape[0]))

    return f_x, df_x, hess


'''
----------------------------------------quadratic_problem-----------------------------------------------------

Given (x,y,z) the function return x^2 + y^2 + (z+1)^2 
The problem is minimization problem - min (x^2 + y^2 + (z+1)^2)

x.shape - (2,1)
'''


def quadratic_problem(x):
    f_x = (x[0][0] * x[0][0]) + (x[1][0] * x[1][0]) + ((x[2][0]+1) * (x[2][0]+1))
    df_x = np.array([[2 * x[0][0]], [2 * x[1][0]], [2 * (x[2][0]+1)]])
    hess_x = np.array([[2, 0, 0],
                       [0, 2, 0],
                       [0, 0, 2]])

    return f_x, df_x, hess_x


'''
--------------------------------------quadratic_problem_eq_constraints--------------------------------------------------

Return A and b such that Ax = b is a equality constraint of the quadratic problem
'''


def quadratic_problem_eq_constraints():
    A = np.array([[1, 1, 1]])
    b = [[1]]

    return A, b


'''
--------------------------------------quadratic_problem_ineq_constraints------------------------------------------------

Return a list of functions which are the inequality constraints of the quadratic problem.
For every h_i in the list we would like that h_i(x) <= 0. 

x.shape - (2,1)
'''


def quadratic_problem_ineq_constraints():

    h_list = [lambda x: -x[0][0], lambda x: -x[1][0], lambda x: -x[2][0]]
    dh_list = [np.array([[-1], [0], [0]]), np.array([[0], [-1], [0]]), np.array([[0], [0], [-1]])]
    d2h_list = [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))]
    return h_list, dh_list, d2h_list


'''
----------------------------------------linear_problem-----------------------------------------------------

Given (x,y) the function return -x -y.
The problem is minimization problem - min (-x-y)

x.shape - (2,1)
'''


def linear_problem(x):
    f_x = (-1)*(x[0][0] + x[1][0])
    df_x = np.array([[-1], [-1]])
    hess_x = np.zeros((2, 2))
    return f_x, df_x, hess_x


'''
--------------------------------------linear_problem_ineq_constraints------------------------------------------------

Return a list of functions which are the inequality constraints of the quadratic problem.
For every h_i in the list we would like that h_i(x) <= 0.

x.shape - (2,1)
'''


def linear_problem_ineq_constraints():
    h_list = [lambda x: 1-x[0][0]-x[1][0], lambda x: x[1][0]-1, lambda x: x[0][0]-2, lambda x: -x[1][0]]
    dh_list = [np.array([[-1], [-1]]), np.array([[0], [1]]), np.array([[1], [0]]), np.array([[0], [-1]])]
    d2h_list = [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))]
    return h_list, dh_list, d2h_list
