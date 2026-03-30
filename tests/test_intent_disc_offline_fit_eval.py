import numpy as np

from analytics.intent_disc_offline_fit_eval import _fit_offline_discriminator
from basketworld.utils.intent_discovery import CompletedIntentEpisode, IntentTransition


def _episode(intent_index: int, center: float, steps: int = 3) -> CompletedIntentEpisode:
    transitions = [
        IntentTransition(
            feature=np.array([center, center + 0.5, center - 0.5, 0.0], dtype=np.float32),
            buffer_step_idx=t,
            env_idx=0,
            role_flag=1.0,
            intent_active=True,
            intent_index=int(intent_index),
        )
        for t in range(steps)
    ]
    return CompletedIntentEpisode(intent_index=int(intent_index), transitions=transitions)


def test_fit_offline_discriminator_learns_separable_synthetic_data():
    train_episodes = []
    test_episodes = []
    centers = {0: -4.0, 1: 0.0, 2: 4.0}
    for intent_index, center in centers.items():
        for i in range(12):
            train_episodes.append(_episode(intent_index, center + i * 0.01))
        for i in range(4):
            test_episodes.append(_episode(intent_index, center + 1.0 + i * 0.01))

    result = _fit_offline_discriminator(
        train_episodes,
        test_episodes,
        disc_config={
            "input_dim": 4,
            "hidden_dim": 32,
            "num_intents": 3,
            "dropout": 0.0,
            "encoder_type": "mlp_mean",
            "step_dim": 16,
            "max_obs_dim": 4,
            "max_action_dim": 0,
            "disc_lambda_shot": 0.0,
            "disc_lambda_q": 0.0,
            "enable_shot_end_head": False,
            "enable_shot_quality_head": False,
        },
        device="cpu",
        epochs=60,
        batch_size=16,
        lr=1e-2,
        seed=0,
    )
    assert float(result["train"]["top1_acc"]) >= 0.95
    assert float(result["test"]["top1_acc"]) >= 0.95
