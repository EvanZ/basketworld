import os
import tempfile
import mlflow

import basketworld
from basketworld.utils.self_play_wrapper import SelfPlayEnvWrapper
from basketworld.utils.action_resolution import IllegalActionStrategy
from basketworld.utils.evaluation_helpers import create_and_log_gif


def run_evaluation(args, unified_policy, global_alt: int):
    """Run evaluation episodes and log GIFs."""
    print(f"\n--- Running Evaluation for Alternation {global_alt} ---")

    base_eval_env = basketworld.HexagonBasketballEnv(
        grid_size=args.grid_size,
        players=args.players,
        shot_clock_steps=args.shot_clock,
        min_shot_clock=getattr(args, "min_shot_clock", 10),
        render_mode="rgb_array",
        three_point_distance=args.three_point_distance,
        three_point_short_distance=getattr(args, "three_point_short_distance", None),
        layup_pct=args.layup_pct,
        layup_std=getattr(args, "layup_std", 0.0),
        three_pt_pct=args.three_pt_pct,
        three_pt_std=getattr(args, "three_pt_std", 0.0),
        allow_dunks=args.allow_dunks,
        dunk_pct=args.dunk_pct,
        dunk_std=getattr(args, "dunk_std", 0.0),
        shot_pressure_enabled=args.shot_pressure_enabled,
        shot_pressure_max=args.shot_pressure_max,
        shot_pressure_lambda=args.shot_pressure_lambda,
        shot_pressure_arc_degrees=args.shot_pressure_arc_degrees,
        spawn_distance=getattr(args, "spawn_distance", 3),
        max_spawn_distance=getattr(args, "max_spawn_distance", None),
        defender_spawn_distance=getattr(args, "defender_spawn_distance", 0),
        pass_reward=getattr(args, "pass_reward", 0.0),
        turnover_penalty=getattr(args, "turnover_penalty", 0.0),
        made_shot_reward_inside=getattr(args, "made_shot_reward_inside", 2.0),
        made_shot_reward_three=getattr(args, "made_shot_reward_three", 3.0),
        missed_shot_penalty=getattr(args, "missed_shot_penalty", 0.0),
        potential_assist_reward=getattr(args, "potential_assist_reward", 0.1),
        full_assist_bonus=getattr(args, "full_assist_bonus", 0.2),
        assist_window=getattr(args, "assist_window", 2),
        potential_assist_pct=getattr(args, "potential_assist_pct", 0.10),
        full_assist_bonus_pct=getattr(args, "full_assist_bonus_pct", 0.05),
        enable_profiling=args.enable_env_profiling,
        profiling_sample_rate=getattr(args, "profiling_sample_rate", 1.0),
        use_egocentric_obs=args.use_egocentric_obs,
        egocentric_rotate_to_hoop=args.egocentric_rotate_to_hoop,
        include_hoop_vector=args.include_hoop_vector,
        normalize_obs=args.normalize_obs,
        mask_occupied_moves=args.mask_occupied_moves,
        enable_pass_gating=getattr(args, "enable_pass_gating", True),
    )
    eval_env = SelfPlayEnvWrapper(
        base_eval_env,
        opponent_policy=unified_policy,
        training_strategy=IllegalActionStrategy.SAMPLE_PROB,
        opponent_strategy=IllegalActionStrategy.SAMPLE_PROB,
        deterministic_opponent=True,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        for ep_num in range(args.eval_episodes):
            obs, info = eval_env.reset()
            done = False
            episode_frames = []

            while not done:
                full_action, _ = unified_policy.predict(obs, deterministic=True)
                obs, reward, done, _, info = eval_env.step(full_action)
                frame = eval_env.render()
                episode_frames.append(frame)

            final_info = info
            action_results = final_info.get("action_results", {})
            outcome = "Unknown"

            if action_results.get("shots"):
                shooter_id = list(action_results["shots"].keys())[0]
                shot_result = list(action_results["shots"].values())[0]
                shooter_pos = eval_env.positions[int(shooter_id)]
                bq, br = eval_env.basket_position
                dist = (
                    abs(shooter_pos[0] - bq)
                    + abs((shooter_pos[0] + shooter_pos[1]) - (bq + br))
                    + abs(shooter_pos[1] - br)
                ) // 2
                is_three = dist >= getattr(eval_env, "three_point_distance", 4)
                if shot_result["success"]:
                    outcome = "Made 3" if is_three else "Made 2"
                else:
                    outcome = "Missed 3" if is_three else "Missed 2"
            elif action_results.get("turnovers"):
                turnover_reason = action_results["turnovers"][0]["reason"]
                if turnover_reason == "intercepted":
                    outcome = "Turnover (Intercepted)"
                elif turnover_reason in ("pass_out_of_bounds", "move_out_of_bounds"):
                    outcome = "Turnover (OOB)"
                elif turnover_reason == "defender_pressure":
                    outcome = "Turnover (Pressure)"
            elif eval_env.unwrapped.shot_clock <= 0:
                outcome = "Turnover (Shot Clock Violation)"

            artifact_path = f"training_eval/alternation_{global_alt}"
            create_and_log_gif(
                frames=episode_frames,
                episode_num=ep_num,
                outcome=outcome,
                temp_dir=temp_dir,
                artifact_path=artifact_path,
            )

    eval_env.close()
    print(f"--- Evaluation for Alternation {global_alt} Complete ---")
