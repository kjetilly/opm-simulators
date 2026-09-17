import os
import unittest
from pathlib import Path
from .pytest_common import pushd, create_black_oil_simulator


class TestSubStepCallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_dir = Path(os.path.dirname(__file__))
        cls.data_dir = test_dir.parent.joinpath("test_data/SPE1CASE1a")

    def test_callback_observes_and_controls_substeps(self):
        from opm.simulators import BlackOil

        with pushd(self.data_dir):
            sim = create_black_oil_simulator(
                args=['--linear-solver=cprw,ilu0', '--output-dir=substep_callback'],
                filename="SPE1CASE1.DATA")
            sim.setup_mpi(init=True, finalize=False)
            infos = []

            def callback(info):
                infos.append(info)
                decision = BlackOil.SubStepDecision()
                # alternate the linear solver, tighten the tolerance and cap the substep at 0.25 days
                decision.linear_solver_index = len(infos) % 2
                decision.linear_solver_tolerance = 1e-3
                decision.dt = min(info.proposed_dt, 0.25 * 86400.0)
                return decision

            sim.set_substep_callback(callback)
            sim.step_init()
            sim.step()
            sim.step()

            self.assertGreater(len(infos), 2)
            first = infos[0]
            self.assertEqual(first.report_step, 0)
            self.assertEqual(first.sub_step, 0)
            self.assertEqual(first.linear_solvers, ['cprw', 'ilu0'])
            self.assertEqual(first.total_sub_steps, 0)
            # the second attempt sees the outcome and the settings of the first one
            second = infos[1]
            self.assertTrue(second.last_converged)
            self.assertGreater(second.last_linear_iterations, 0)
            self.assertLessEqual(second.last_dt, 0.25 * 86400.0 + 1e-6)
            self.assertEqual(second.active_linear_solver, 1)
            self.assertAlmostEqual(second.linear_solver_tolerance, 1e-3)
            # every substep was capped at 0.25 days, so two 1 day report steps need at least 8 attempts
            self.assertGreaterEqual(len(infos), 8)

            totals = sim.get_substep_totals()
            # every attempt either converged or failed
            self.assertEqual(totals['sub_steps'] + totals['failed_sub_steps'], len(infos))
            self.assertGreater(totals['linear_iterations'], 0)

            # a dict decision and a None decision are accepted as well
            sim.set_substep_callback(lambda info: {'linear_solver_max_iterations': 50})
            sim.step()
            sim.set_substep_callback(lambda info: None)
            sim.step()
            sim.clear_substep_callback()
            sim.step()

    def test_invalid_solver_index_raises(self):
        with pushd(self.data_dir):
            sim = create_black_oil_simulator(
                args=['--linear-solver=ilu0', '--output-dir=substep_callback_invalid'],
                filename="SPE1CASE1.DATA")
            sim.setup_mpi(init=False, finalize=False)
            sim.set_substep_callback(lambda info: {'linear_solver_index': 3})
            sim.step_init()
            with self.assertRaises(Exception):
                sim.step()


if __name__ == '__main__':
    unittest.main()
