import numpy as np

from thesis_rl.preprocessors.identity import IdentityPreprocessor


def test_identity_preprocessor_is_passthrough() -> None:
    preprocessor = IdentityPreprocessor()
    obs = np.array([1.0, 2.0], dtype=np.float64)
    out = preprocessor(obs)
    assert out is obs
