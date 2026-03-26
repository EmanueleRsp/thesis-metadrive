import numpy as np

from thesis_rl.preprocessors.identity import IdentityPreprocessor


def test_identity_preprocessor_casts_float32() -> None:
    preprocessor = IdentityPreprocessor(cast_to_float32=True)
    obs = np.array([1.0, 2.0], dtype=np.float64)
    out = preprocessor(obs)
    assert out.dtype == np.float32
