"""Bundled Overcooked environment.

The core gridworld is usable without Gym. Registration is opportunistic so
imports and offline evaluation remain available in the base environment.
"""

try:
    from gym.envs.registration import register
except ModuleNotFoundError:  # Gym is needed only through gym.make().
    register = None

if register is not None:
    register(
        id="Overcooked-v0",
        entry_point=(
            "multimodal_comms.apps.collab_overcooked.overcooked_ai_py.mdp."
            "overcooked_env:Overcooked"
        ),
    )
