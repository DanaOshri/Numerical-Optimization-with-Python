import unittest
import numpy as np
import src.utils as utils
import src.constrained_min as constrained_min
import tests.examples as examples


class TestLogBarrierMethods(unittest.TestCase):

    def test_qp(self):
        print("test_qp")

        start_point = np.array([[0.1], [0.2], [0.7]])
        init_step_len = 1.0
        slope_ratio = 1e-4
        back_track_factor = 0.2
        A, b = examples.quadratic_problem_eq_constraints()

        sol, success, history = constrained_min.interior_pt(examples.quadratic_problem,
                                                            examples.quadratic_problem_ineq_constraints,
                                                            A,
                                                            b,
                                                            start_point,
                                                            init_step_len,
                                                            slope_ratio,
                                                            back_track_factor)

        self.assertTrue(success)

        if success:
            utils.plot_feasible_region_3d("Quadratic minimization problem - feasible region")
            utils.plot_3d_outlines(examples.quadratic_problem, sol, start_point, history,
                                   "Plot quadratic minimization problem")

    def test_lp(self):
        print("test_lp")

        start_point = np.array([[0.5], [0.75]])
        init_step_len = 1.0
        slope_ratio = 1e-4
        back_track_factor = 0.2
        A = np.zeros((0, 0))
        b = np.zeros((0))

        sol, success, history = constrained_min.interior_pt(examples.linear_problem,
                                                            examples.linear_problem_ineq_constraints,
                                                            A,
                                                            b,
                                                            start_point,
                                                            init_step_len,
                                                            slope_ratio,
                                                            back_track_factor)

        self.assertTrue(success)

        if success:
            utils.plot_linear_feasible_region("linear minimization problem - feasible region")
            utils.plot_outlines(examples.linear_problem, sol, start_point, history, "Plot linear minimization problem")

#
# if __name__ == '__main__':
#     unittest.main()