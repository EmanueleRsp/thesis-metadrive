from thesis_rl.adapters.base import BaseAdapter
from thesis_rl.adapters.direct_action import DirectActionAdapter
from thesis_rl.adapters.neural_adapter import NeuralAdapter
from thesis_rl.adapters.policy_adapter import PolicyAdapter
from thesis_rl.agents.planner_agent import Td3PlannerBackend
from thesis_rl.curriculum.manager import CurriculumManager
from thesis_rl.preprocessors.identity import IdentityPreprocessor


def test_core_symbols_import() -> None:
    assert BaseAdapter is not None
    assert IdentityPreprocessor is not None
    assert DirectActionAdapter is not None
    assert NeuralAdapter is not None
    assert PolicyAdapter is not None
    assert Td3PlannerBackend is not None
    assert CurriculumManager is not None
