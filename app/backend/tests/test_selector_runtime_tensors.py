import warnings

import numpy as np

from app.backend.selector_runtime import selector_sample_intent


class _ReadOnlySelectorPolicy:
    def has_intent_selector(self):
        return True

    def get_intent_selector_outputs(self, obs):
        logits = np.asarray([[0.0, 2.0, -1.0]], dtype=np.float32)
        values = np.asarray([0.25], dtype=np.float32)
        logits.setflags(write=False)
        values.setflags(write=False)
        return logits, values


def test_selector_sample_intent_accepts_readonly_numpy_outputs_without_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = selector_sample_intent(
            {
                "intent_selector_enabled": True,
                "intent_selector_mode": "integrated",
                "intent_selector_alpha_start": 1.0,
                "intent_selector_alpha_end": 1.0,
                "intent_selector_eps_start": 0.0,
                "intent_selector_eps_end": 0.0,
            },
            _ReadOnlySelectorPolicy(),
            {},
            num_intents=3,
            allow_uniform_fallback=False,
            selection_mode="best_intent",
            rng=np.random.default_rng(0),
        )

    assert result["intent_index"] == 1
    assert result["used_selector"] is True
    assert not any("not writable" in str(item.message) for item in caught)
