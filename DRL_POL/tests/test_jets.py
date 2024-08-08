import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the directory containing your python files to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jets import (
    build_jets,
    JetCylinder,
    atan2_str,
    Jet,
    Q_smooth_exp,
    Q_smooth_linear,
    heav_func,
)


class MockJet:
    def __init__(self, name, params, T_smoo):
        self.name = name
        self.params = params
        self.T_smoo = T_smoo


class TestBuildJets:

    def test_build_jets_basic(self):
        jets_definition = {
            "jet1": {"param1": 1, "param2": 2},
            "jet2": {"param1": 3, "param2": 4},
        }
        delta_t_smooth = 0.2
        jets = build_jets(MockJet, jets_definition, delta_t_smooth)

        assert len(jets) == 2
        assert isinstance(jets["jet1"], MockJet)
        assert jets["jet1"].params == jets_definition["jet1"]
        assert jets["jet1"].T_smoo == delta_t_smooth

    def test_build_jets_empty(self):
        jets_definition = {}
        delta_t_smooth = 0.2
        jets = build_jets(MockJet, jets_definition, delta_t_smooth)

        assert len(jets) == 0

    def test_build_jets_single(self):
        jets_definition = {"jet1": {"param1": 1, "param2": 2}}
        delta_t_smooth = 0.2
        jets = build_jets(MockJet, jets_definition, delta_t_smooth)

        assert len(jets) == 1
        assert isinstance(jets["jet1"], MockJet)
        assert jets["jet1"].params == jets_definition["jet1"]
        assert jets["jet1"].T_smoo == delta_t_smooth


class TestAtan2Str:
    def test_atan2_str_basic(self):
        result = atan2_str("x", "y")
        expected = "2*atan(y/(x + sqrt(x^2+y^2)))"
        assert result == expected


class TestQSmoothLinear:
    def test_q_smooth_linear_basic(self):
        Qnew = 10
        Qpre = 5
        timestart = 0
        Tsmooth = 2
        result = Q_smooth_linear(Qnew, Qpre, timestart, Tsmooth)
        expected = "(5/2*(t-0) + (5))"
        assert result == expected


class TestQSmoothExp:
    def test_q_smooth_exp_basic(self):
        ts = 0
        Tsmooth = 2
        result = Q_smooth_exp(ts, Tsmooth)
        expected = "exp(-1/(pos((t-0.00)/2.00)))/(exp(-1/(pos((t-0.00)/2.00)))+exp(-1/pos(1-(pos((t-0.00)/2.00)))))"
        assert result == expected


class TestHeavFunc:
    def test_heav_func_basic(self):
        position_z = 1.0
        delta_z = 0.5
        result = heav_func(position_z, delta_z)
        expected = "heav((z-0.750)*(1.250-z))"
        assert result == expected


class TestJetCylinder:
    def setup_method(self):
        # Set up parameters for the JetCylinder class
        self.name = "test_jet"
        self.params = {"radius": 0.5, "width": 1.0, "positions_angle": 0}  # Changed from 0 to 1.0
        self.Q_pre = [0.0] * 10
        self.Q_new = [0.0] * 10
        self.time_start = 4.53
        self.T_smoo = 0.25
        self.smooth_func = "EXPONENTIAL"
        self.Qs_position_z = [i * 0.4 for i in range(10)]
        self.delta_Q_z = 0.4

        # Create a mock parameters module
        mock_parameters = MagicMock()
        mock_parameters.dimension = 2
        mock_parameters.Qs_position_z = self.Qs_position_z
        mock_parameters.delta_Q_z = self.delta_Q_z
        mock_parameters.short_spacetime_func = False
        mock_parameters.nb_inv_per_CFD = 10

        # Add cylinder_coordinates to mock_parameters
        mock_parameters.cylinder_coordinates = [2.5, 5.0, 0.0]

        # Patch the parameters module
        self.patcher = patch.dict("sys.modules", {"parameters": mock_parameters})
        self.patcher.start()

        # Create an instance of JetCylinder
        self.jet = JetCylinder(name=self.name, params=self.params)

    def teardown_method(self):
        self.patcher.stop()

    def test_create_smooth_funcs(self):
        result = self.jet.create_smooth_funcs(
            self.Q_new,
            self.Q_pre,
            self.time_start,
            self.T_smoo,
            self.smooth_func,
            self.Qs_position_z,
            self.delta_Q_z
        )

        # Print the result for debugging purposes
        print(result)

        expected_start = "(18.0)*(((heav((z-0.000)*(0.400-z))*(0.0000)+ heav((z-0.400)*(0.800-z))*(0.0000)+ heav((z-0.800)*(1.200-z))*(0.0000)+ heav((z-1.200)*(1.600-z))*(0.0000)+ heav((z-1.600)*(2.000-z))*(0.0000)+ heav((z-2.000)*(2.400-z))*(0.0000)+ heav((z-2.400)*(2.800-z))*(0.0000)+ heav((z-2.800)*(3.200-z))*(0.0000)+ heav((z-3.200)*(3.600-z))*(0.0000)+ heav((z-3.600)*(4.000-z))*(0.0000)) + (exp(-1/(pos((t-4.53)/0.25"
        assert result.startswith(expected_start)

    def test_update(self):
        Q_pre = [1.0] * 10
        Q_new = [2.0] * 10
        time_start = 0.0
        smooth_func = "EXPONENTIAL"

        self.jet.update(Q_pre, Q_new, time_start, smooth_func)

        assert self.jet.Q_pre == Q_pre
        assert self.jet.Q_new == Q_new
        assert self.jet.time_start == time_start
        assert self.jet.smooth_func == smooth_func