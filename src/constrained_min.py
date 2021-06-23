import numpy as np
import src.unconstrained_min as unconstrained_min
import src.utils as utils

'''
----------------------------------------interior_pt-----------------------------------------------------

    func                        function to minimize.
    ineq_constraints            list of inequality constraints.
    eq_constraints_mat          Matrix A of affine equality constraints Ax = b.
    eq_constraints_rhs          Vector b of affine equality constraints Ax = b.
    x0                          Starting point for the outer iterations .
'''


def interior_pt(f, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0, init_step_len,
                slope_ratio, back_track_factor, t=1, mu=10, max_iter_outer=100, max_iter_inner=100):
    inner_history = []
    epsilon = 1e-4

    A = eq_constraints_mat
    b = eq_constraints_rhs
    N = A.shape[0] # N - number of equality constraints

    h_list, dh_list, hess_list = ineq_constraints()
    M = len(h_list) # M - number of inequality constraints

    x_prev = x0
    success_outer = False
    for i in range(max_iter_outer):

        # Newton's Method (Equality constraint version)
        x_next, success, history = newton_method_equality_constraints(x_prev, f, ineq_constraints, t, N, A, epsilon,
                                                                      init_step_len, slope_ratio, back_track_factor,
                                                                      max_iter_inner)
        if not success:
            break

        inner_history += history

        if M/t < epsilon:
            success_outer = True
            break

        t = t*mu
        x_prev = x_next

    return x_next, success_outer, inner_history


'''
----------------------------------------newton_method_equality_constraints---------------------------------------------

 Newton’s method for constrained problem is a descent method that generates a sequence of feasible points.
 This requires in particular a feasible point as a starting point.
'''


def newton_method_equality_constraints(x0, f, ineq_constraints, t, N, A, epsilon, init_step_len, slope_ratio,
                                       back_track_factor, max_iter=100):

    history = []
    success = False

    xk = x0
    for k in range(max_iter):
        log_barr_f_xk, log_barr_df_xk, log_barr_hess_xk = f_log_barrier(xk, f, ineq_constraints, t)

        if N > 0:
            tmp_M1 = np.concatenate((log_barr_hess_xk, A.T), axis=1)
            tmp_M2 = np.concatenate((A, np.zeros((N, N))), axis=1)
            M = np.concatenate((tmp_M1, tmp_M2), axis=0)
            v = np.concatenate(((-1) * log_barr_df_xk, np.zeros((N, 1))))
            sol = np.linalg.solve(M, v)
            step_dir = sol[:log_barr_hess_xk.shape[0]]
        else:
            step_dir = unconstrained_min.newton_dir(log_barr_df_xk, log_barr_hess_xk)

        lambda_x = np.sqrt(np.matmul(np.matmul(step_dir.T, log_barr_hess_xk), step_dir))
        if (0.5 * lambda_x * lambda_x) < epsilon:
            success = True
            break

        step_len = backtracking_line_search(xk, f, step_dir, ineq_constraints, t, init_step_len, slope_ratio,
                                            back_track_factor)

        xk1 = xk + (step_len * step_dir)
        log_barr_f_xk1, _, _ = f_log_barrier(xk1, f, ineq_constraints, t)

        # Print inner iteration data.
        utils.iteration_reporting(k, xk1, log_barr_f_xk1,
                                  np.linalg.norm(xk - xk1),
                                  np.linalg.norm(log_barr_f_xk1 - log_barr_f_xk))
        history.append({'x_prev': xk,
                        'x_next': xk1,
                        'f_next': log_barr_f_xk1,
                        'step_len': np.linalg.norm(xk1 - xk),
                        'obj_change': np.linalg.norm(log_barr_f_xk1 - log_barr_f_xk)})

        xk = xk1

    return xk, success, history


'''
----------------------------------------backtracking_line_search-----------------------------------------------------

The backtracking approach ensures either that the selected step length alpha_k is some fixed value,
or else that it is short enough to satisfy the sufficient decrease condition but not too short.

'''


def backtracking_line_search(x, f, step_dir, ineq_constraints, t, init_step_len, slope_ratio, back_track_factor):
    alpha = init_step_len
    log_barr_f_x, log_barr_df_xk, _ = f_log_barrier(x, f, ineq_constraints, t)
    log_barr_f_x_next, _, _ = f_log_barrier((x + (alpha * step_dir)), f, ineq_constraints, t)

    while np.isnan(log_barr_f_x_next) or \
            (log_barr_f_x_next - log_barr_f_x) > (slope_ratio * alpha * np.matmul(log_barr_df_xk.T, step_dir)):
        alpha *= back_track_factor
        log_barr_f_x_next, _, _ = f_log_barrier((x + (alpha * step_dir)), f, ineq_constraints, t)

    return alpha


'''
----------------------------------------calculate_teta-----------------------------------------------------

Calculating the theta function at point x.

'''


def calculate_theta(x, ineq_constraints):

    sum_log_ineq = 0
    sum_d_ineq = 0
    sum_hess_ineq_1 = 0
    sum_hess_ineq_2 = 0

    h_list, dh_list, hess_list = ineq_constraints()

    for h_i, dh_i, hess_i in zip(h_list, dh_list, hess_list):
        h_i_x = h_i(x)
        sum_log_ineq += np.log((-1) * h_i_x)
        sum_d_ineq += (dh_i / ((-1) * h_i_x))
        sum_hess_ineq_1 += (np.matmul(dh_i, dh_i.T) / (h_i_x * h_i_x))
        sum_hess_ineq_2 += (hess_i / ((-1) * h_i_x))

    theta = (-1) * sum_log_ineq
    d_theta = sum_d_ineq
    hess_theta = sum_hess_ineq_1 + sum_hess_ineq_2


    return theta, d_theta, hess_theta


'''
----------------------------------------f_log_barrier-----------------------------------------------------

Calculating the log barrier function at point x.

'''


def f_log_barrier(x, f, ineq_constraints, t):

    theta_x, d_theta_x, hess_theta_x = calculate_theta(x, ineq_constraints)
    f_x, df_x, hess_x = f(x)

    log_barr_f_x = (t * f_x) + theta_x
    log_barr_df_x = (t * df_x) + d_theta_x
    log_barr_hess_x = (t * hess_x) + hess_theta_x


    return log_barr_f_x, log_barr_df_x, log_barr_hess_x
