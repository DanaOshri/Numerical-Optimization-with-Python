import src.utils as utils
import numpy as np
'''
----------------------------------------gradient_descent-----------------------------------------------------

    f                       the function minimized.

    x0                      the starting point.

    step_len                the coefficient multiplying the gradient vector in the algorithm update rule.

    obj_tol                 the numeric tolerance for successful termination in terms of small enough
                            change in objective function values, between two consecutive iterations
                            (𝑓(𝑥𝑖+1) and 𝑓(𝑥𝑖)).

    param_tol               the numeric tolerance for successful termination in terms of small enough
                            distance between two consecutive iterations iteration locations (𝑥𝑖+1 and 𝑥𝑖).
                        
    max_iter                the maximum allowed number of iterations.
    
    dir_selection_method    specify the method for selecting the direction for line search 
                            (gradient descent, Newton or BFGS, respectively).
    
    init_step_len           the initial step length is chosen to be 1 in Newton and quasi- Newton methods.
                
    slope_ratio             constant value used in Backtracking Line Search algorithm.
                
    back_track_factor       determine the update of step_len in Backtracking Line Search algorithm.
'''


def gradient_descent(f, x0, obj_tol, param_tol, max_iter, dir_selection_method,
                     init_step_len, slope_ratio, back_track_factor):
    history = []
    x_next = x_prev = x0

    if dir_selection_method == 'nt' or dir_selection_method == 'bfgs':
        hess_flag = True
    else:
        hess_flag = False

    f_prev, df_prev, hess_prev = f(x0, hess_flag)
    i = 0
    success = False
    while i <= max_iter:

        # Calculate step direction
        if dir_selection_method == 'gd':
            hess_flag = False
            dir_step = (-1)*df_prev
        if dir_selection_method == 'nt':
            hess_flag = True
            dir_step = newton_dir(df_prev, hess_prev)
        if dir_selection_method == 'bfgs':
            hess_flag = False
            dir_step = bfgs_dir(df_prev, hess_prev)

        # Calculate next step length
        step_len = backtracking_line_search(x_prev, f, dir_step, init_step_len, slope_ratio, back_track_factor)

        # Update next point using step direction and step length.
        x_next = x_prev + step_len * dir_step

        f_next, df_next, hess_next = f(x_next, hess_flag)

        if dir_selection_method == 'bfgs':
            hess_next = bfgs_hess_approx(x_next, x_prev, df_next, df_prev, hess_prev)

        i += 1

        # Print iteration data.
        utils.iteration_reporting(i, x_next, f_next,
                                  np.linalg.norm(x_next-x_prev),
                                  np.linalg.norm(f_next-f_prev))
        history.append({'x_prev': x_prev,
                        'x_next': x_next,
                        'f_next': f_next,
                        'step_len': np.linalg.norm(x_next-x_prev),
                        'obj_change': np.linalg.norm(f_next-f_prev)})

        # Check convergence.
        if check_converge(x_next, x_prev, param_tol, f_next, f_prev, obj_tol, dir_step):
            success = True
            break

        x_prev = x_next
        f_prev, df_prev, hess_prev = f_next, df_next, hess_next

    print("success flag = ", success)
    return x_next, success, history




'''
----------------------------------------check_converge-----------------------------------------------------

The function returns true if either one of the tolerances successfully achieved.
'''


def check_converge(x_next, x_prev, param_tol, f_next, f_prev, obj_tol, dir_step):

    if np.linalg.norm(x_next - x_prev) < param_tol:
        return True

    if np.linalg.norm(f_next - f_prev) < obj_tol:
        return True

    if np.linalg.norm(dir_step) == 0:
        return True

    return False



'''
----------------------------------------newton_dir-----------------------------------------------------

The function returns the direction of the next step when using Newton's method.
'''


def newton_dir(df, hess):
    I = np.identity(hess.shape[0])
    hess_inv = np.linalg.solve(hess, I)
    return (-1) * hess_inv.dot(df)



'''
----------------------------------------bfgs_dir-----------------------------------------------------

The function returns the direction of the next step when using BFGS method.
'''


def bfgs_dir(df, hess):
    return newton_dir(df, hess)


'''
----------------------------------------bfgs_hess_approx-----------------------------------------------------

The function returns the approximate hessian of the next step.
'''


def bfgs_hess_approx(x_next, x_prev, df_next, df_prev, hess_prev):
    sk = x_next - x_prev
    yk = df_next - df_prev

    A = np.matmul(hess_prev.dot(sk), sk.T.dot(hess_prev))
    B = np.matmul(sk.T.dot(hess_prev), sk)
    C = yk.dot(yk.T)
    D = yk.T.dot(sk)
    hess_next = hess_prev - (A/B) + (C/D)

    return hess_next





'''
----------------------------------------backtracking_line_search-----------------------------------------------------

The backtracking approach ensures either that the selected step length alpha_k is some fixed value,
or else that it is short enough to satisfy the sufficient decrease condition but not too short.
'''


def backtracking_line_search(x, f, step_dir, init_step_len, slope_ratio, back_track_factor):
    alpha = init_step_len
    f_x, df_x, _ = f(x)
    f_x_next, _, _ = f(x + (alpha * step_dir))

    while (f_x_next - f_x) > (slope_ratio * alpha * np.matmul(df_x.T, step_dir)):
        alpha *= back_track_factor
        f_x_next, _, _ = f(x + (alpha * step_dir))

    return alpha

