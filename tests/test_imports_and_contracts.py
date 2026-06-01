from thesis_rl.agent.adapters.interfaces.base import BaseAdapter
from thesis_rl.agent.adapters.identity import IdentityAdapter
from thesis_rl.agent.planners.algorithms import PpoPlannerBackend, SacPlannerBackend, Td3PlannerBackend
from thesis_rl.curriculum.manager import CurriculumManager
from thesis_rl.agent.planners.encoders import LQEncoder, MLPEncoder, NoneEncoder
from thesis_rl.observations import ObservationSpec
from thesis_rl.agent.preprocessors.identity import IdentityPreprocessor


def test_core_symbols_import() -> None:
    assert BaseAdapter is not None
    assert IdentityPreprocessor is not None
    assert IdentityAdapter is not None
    assert Td3PlannerBackend is not None
    assert SacPlannerBackend is not None
    assert PpoPlannerBackend is not None
    assert NoneEncoder is not None
    assert MLPEncoder is not None
    assert LQEncoder is not None
    assert ObservationSpec is not None
    assert CurriculumManager is not None
