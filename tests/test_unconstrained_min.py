import unittest
import math
import numpy as np
import src.utils as utils
import src.unconstrained_min as unconstrained_min
import tests.examples as examples


class TestGradientDescentMethods(unittest.TestCase):

    def test_quad_min(self):
        print("test_quad_min")

        start_point = np.array([[1], [1]])
        max_iter = 1000
        param_tol = math.pow(10, -8)
        obj_tol = math.pow(10, -12)
        init_step_len = 1.0
        slope_ratio = 1e-4
        back_track_factor = 0.2

        methods = ['gd', 'nt', 'bfgs']
        for dir_selection_method in methods:
            sol, success, history = unconstrained_min.gradient_descent(examples.quadratic_1,
                                                                       start_point,
                                                                       obj_tol,
                                                                       param_tol,
                                                                       max_iter,
                                                                       dir_selection_method,
                                                                       init_step_len,
                                                                       slope_ratio,
                                                                       back_track_factor)

            if success:
                utils.plot_outlines(examples.quadratic_1, sol, start_point, history,
                                    "Plot quadratic 1 - " + dir_selection_method)

            self.assertTrue(success)

            sol, success, history = unconstrained_min.gradient_descent(examples.quadratic_2,
                                                                       start_point,
                                                                       obj_tol,
                                                                       param_tol,
                                                                       max_iter,
                                                                       dir_selection_method,
                                                                       init_step_len,
                                                                       slope_ratio,
                                                                       back_track_factor)

            if success:
                utils.plot_outlines(examples.quadratic_2, sol, start_point, history,
                                    "Plot quadratic 2 - " + dir_selection_method)

            self.assertTrue(success)

            sol, success, history = unconstrained_min.gradient_descent(examples.quadratic_3,
                                                                       start_point,
                                                                       obj_tol,
                                                                       param_tol,
                                                                       max_iter,
                                                                       dir_selection_method,
                                                                       init_step_len,
                                                                       slope_ratio,
                                                                       back_track_factor)

            if success:
                utils.plot_outlines(examples.quadratic_3, sol, start_point, history,
                                    "Plot quadratic 3 - " + dir_selection_method)

            self.assertTrue(success)



    def test_rosenbrock_min(self):
        print("test_rosenbrock_min")

        start_point = np.array([[2], [2]])
        max_iter = 50000
        param_tol = math.pow(10, -8)
        obj_tol = math.pow(10, -15)
        init_step_len = 1.0
        slope_ratio = 1e-4
        back_track_factor = 0.2

        methods = ['gd', 'nt', 'bfgs']
        for dir_selection_method in methods:
            sol, success, history = unconstrained_min.gradient_descent(examples.rosenbrock,
                                                                       start_point,
                                                                       obj_tol,
                                                                       param_tol,
                                                                       max_iter,
                                                                       dir_selection_method,
                                                                       init_step_len,
                                                                       slope_ratio,
                                                                       back_track_factor)

            if success:
                utils.plot_outlines(examples.rosenbrock, sol, start_point, history,
                                    "Plot Rosenbrock - " + dir_selection_method)

            self.assertTrue(success)

    def test_lin_min(self):
        start_point = np.array([[1], [1]])
        max_iter = 10000
        param_tol = math.pow(10, -8)
        obj_tol = math.pow(10, -7)
        init_step_len = 1.0
        slope_ratio = 1e-4
        back_track_factor = 0.2

        methods = ['gd']
        for dir_selection_method in methods:
            sol, success, history = unconstrained_min.gradient_descent(examples.linear,
                                                                       start_point,
                                                                       obj_tol,
                                                                       param_tol,
                                                                       max_iter,
                                                                       dir_selection_method,
                                                                       init_step_len,
                                                                       slope_ratio,
                                                                       back_track_factor)

            if len(history) > 0:
                utils.plot_outlines(examples.linear, sol, start_point, history,
                                    "Plot linear - " + dir_selection_method)

            #self.assertTrue(success)



# if __name__ == '__main__':
#     unittest.main()