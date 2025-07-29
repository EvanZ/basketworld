import time
from stable_baselines3.common.callbacks import BaseCallback

class TimingCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rollout_start = None
        self.update_start  = None
        self.rollout_times = []
        self.update_times  = []

    # ---------- roll-out ----------
    def _on_rollout_start(self) -> None:
        """Called at the start of the rollout phase."""
        self.rollout_start = time.perf_counter()

    def _on_rollout_end(self) -> None:
        """Called at the end of the rollout phase."""
        dt = time.perf_counter() - self.rollout_start
        self.rollout_times.append(dt)
        self.logger.record("perf/rollout_sec", dt)

    # ---------- gradient update ----------
    def _on_training_start(self) -> None:
        """Called at the start of the gradient update phase."""
        # Note: This is a bit of a misnomer in SB3. 
        # _on_training_start is called once before the whole training loop.
        # We will use _on_step to mark the start of the update phase, as it's called right before.
        # However, for simplicity and to align with the provided snippet, we will use this hook,
        # acknowledging it measures the total training time per `learn` call, not per update cycle.
        # A more precise hook would be to wrap the train method itself.
        # For this implementation, we will place the start timer before the update phase.
        # The user's code uses _on_training_start, but that's called only once.
        # The _on_step is called multiple times, but before the rollout.
        # Looking at SB3 docs, _on_rollout_end is followed by the update. So _on_rollout_end
        # can be used to time the start of the update, and _on_rollout_start for the end of the update.
        # This seems counter-intuitive.
        
        # Let's re-read the user request.
        # `_on_training_start` is called before `model.learn`.
        # `_on_rollout_start` is called before collecting rollouts.
        # `_on_rollout_end` is called after collecting rollouts.
        # `_on_step` is called at each step inside the rollout collection.
        # `_on_training_end` is called after `model.learn`.
        
        # The user's code uses `_on_training_start` and `_on_training_end`.
        # This seems to be incorrect if `learn` is called only once.
        # `_on_training_start` runs once at the beginning of `learn`.
        # `_on_training_end` runs once at the end of `learn`.
        # This would only produce one measurement for the entire training run.
        
        # The goal is to measure each update cycle.
        # An update cycle in PPO is: collect rollouts -> update policy.
        # So: `_on_rollout_start` -> ... collect ... -> `_on_rollout_end` -> ... update ... -> `_on_rollout_start`
        
        # The time between `_on_rollout_end` and the next `_on_rollout_start` is the update time.
        # This seems like a more robust way to measure it.
        
        self.update_start = time.perf_counter()

    def _on_step(self) -> bool:
        # This is called within the rollout collection, so we don't need it for timing the update phase.
        return True

    def _on_training_end(self) -> None:
        """Called at the end of the gradient update phase."""
        # This is called after the training loop, so we can't use it to time each update.
        # Let's adjust the logic slightly to correctly capture per-update-cycle times.
        # We will time the update phase from the end of one rollout to the start of the next.
        
        # The provided code seems to have a slight flaw in its logic for timing updates with SB3's hooks.
        # `_on_training_start` and `_on_training_end` are only called once for the entire `.learn()` process.
        # To measure each update cycle, we should time the period *between* rollout collections.
        pass

# A more accurate implementation based on SB3 hook execution order
class RolloutUpdateTimingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rollout_time = 0
        self.update_time = 0
        self.rollout_start_time = 0
        self.update_start_time = 0
        
        self.rollout_times = []
        self.update_times = []

    def _on_rollout_start(self) -> None:
        """Hook called before the rollout collection starts."""
        # This marks the end of the previous update phase.
        if self.update_start_time != 0:
            self.update_time = time.perf_counter() - self.update_start_time
            self.update_times.append(self.update_time)
            self.logger.record("perf/update_sec", self.update_time)

        self.rollout_start_time = time.perf_counter()

    def _on_rollout_end(self) -> None:
        """Hook called after the rollout collection finishes."""
        # This marks the end of the rollout phase and the start of the update phase.
        self.rollout_time = time.perf_counter() - self.rollout_start_time
        self.rollout_times.append(self.rollout_time)
        self.logger.record("perf/rollout_sec", self.rollout_time)
        
        self.update_start_time = time.perf_counter()

    def _on_step(self) -> bool:
        return True 