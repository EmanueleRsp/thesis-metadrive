from thesis_rl.adapters.base import BaseAdapter
from thesis_rl.adapters.identity import IdentityAdapter
from thesis_rl.adapters.neural_adapter import NeuralAdapter
from thesis_rl.adapters.policy_adapter import PolicyAdapter
from thesis_rl.planners.algorithms import PpoPlannerBackend, SacPlannerBackend, Td3PlannerBackend
from thesis_rl.curriculum.manager import CurriculumManager
from thesis_rl.networks.encoders import LQEncoder, MLPEncoder, NoneEncoder
from thesis_rl.observations import ObservationSpec
from thesis_rl.preprocessors.identity import IdentityPreprocessor


def test_core_symbols_import() -> None:
    assert BaseAdapter is not None
    assert IdentityPreprocessor is not None
    assert IdentityAdapter is not None
    assert NeuralAdapter is not None
    assert PolicyAdapter is not None
    assert Td3PlannerBackend is not None
    assert SacPlannerBackend is not None
    assert PpoPlannerBackend is not None
    assert NoneEncoder is not None
    assert MLPEncoder is not None
    assert LQEncoder is not None
    assert ObservationSpec is not None
    assert CurriculumManager is not None
