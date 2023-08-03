from gym.envs.registration import register

# Mujoco
# ----------------------------------------

# - randomised reward functions

register(
    "AntDir-v0",
    entry_point="environments.wrappers:mujoco_wrapper",
    kwargs={
        "entry_point": "environments.mujoco.ant_dir:AntDirEnv",
        "max_episode_steps": 200,
    },
    max_episode_steps=200,
)

register(
    "AntDir2D-v0",
    entry_point="environments.wrappers:mujoco_wrapper",
    kwargs={
        "entry_point": "environments.mujoco.ant_dir:AntDir2DEnv",
        "max_episode_steps": 200,
    },
    max_episode_steps=200,
)

register(
    "AntGoal-v0",
    entry_point="environments.wrappers:mujoco_wrapper",
    kwargs={
        "entry_point": "environments.mujoco.ant_goal:AntGoalEnv",
        "max_episode_steps": 200,
    },
    max_episode_steps=200,
)

register(
    "HalfCheetahDir-v0",
    entry_point="environments.wrappers:mujoco_wrapper",
    kwargs={
        "entry_point": "environments.mujoco.half_cheetah_dir:HalfCheetahDirEnv",
        "max_episode_steps": 200,
    },
    max_episode_steps=200,
)

register(
    "HalfCheetahVel-v0",
    entry_point="environments.wrappers:mujoco_wrapper",
    kwargs={
        "entry_point": "environments.mujoco.half_cheetah_vel:HalfCheetahVelEnv",
        "max_episode_steps": 200,
    },
    max_episode_steps=200,
)

register(
    "HumanoidDir-v0",
    entry_point="environments.wrappers:mujoco_wrapper",
    kwargs={
        "entry_point": "environments.mujoco.humanoid_dir:HumanoidDirEnv",
        "max_episode_steps": 200,
    },
    max_episode_steps=200,
)

# - randomised dynamics

register(
    id="Walker2DRandParams-v0",
    entry_point="environments.mujoco.rand_param_envs.walker2d_rand_params:Walker2DRandParamsEnv",
    max_episode_steps=200,
)

register(
    id="HopperRandParams-v0",
    entry_point="environments.mujoco.rand_param_envs.hopper_rand_params:HopperRandParamsEnv",
    max_episode_steps=200,
)


# # 2D Navigation
# # ----------------------------------------
#
register(
    "PointEnv-v0",
    entry_point="environments.navigation.point_robot:PointEnv",
    kwargs={
        "max_episode_steps": 100,
        "goal_sampler": "semi-circle",
    },
    max_episode_steps=100,
)

register(
    "SparsePointEnv-v0",
    entry_point="environments.navigation.point_robot:SparsePointEnv",
    kwargs={
        "goal_radius": 0.2,
        "max_episode_steps": 100,
        "goal_sampler": "semi-circle",
    },
    max_episode_steps=100,
)

#
# # GridWorld
# # ----------------------------------------

register(
    "GridNavi-v0",
    entry_point="environments.navigation.gridworld:GridNavi",
    kwargs={"num_cells": 5, "num_steps": 15},
)

#
# # Alchemy
# # ----------------------------------------

register(
    "Alchemy-v0",
    entry_point="environments.alchemy.alchemy:AlchemyEnv",
    kwargs={"seed": 0},
)


# # ICPI
# # ----------------------------------------

# elif env_id == "chain":
#     env = TimeLimit(
#         chain.Env(d=1, data=data, goal=4, n=8, random_seed=seed, hint=hint),
#         max_episode_steps=8,
#     )
register(
    "chain-v0",
    entry_point="environments.icpi.chain:create",
    kwargs={"d": 1, "goal": 4, "n": 8, "random_seed": 0, "max_episode_steps": 8},
)
# elif env_id == "distractor-chain":
#     env = TimeLimit(
#         chain.Env(d=2, data=data, goal=4, n=8, random_seed=seed, hint=hint),
#         max_episode_steps=8,
#     )
register(
    "distractor-chain-v0",
    entry_point="environments.icpi.chain:create",
    kwargs={"d": 2, "goal": 4, "n": 8, "random_seed": 0, "max_episode_steps": 8},
)
# elif env_id == "maze":
#     env = TimeLimit(
#         maze.Env(data=data, hint=hint, random_seed=seed), max_episode_steps=8
#     )
register(
    "maze-v0",
    entry_point="environments.icpi.maze:create",
    kwargs={"random_seed": 0, "max_episode_step": 8},
)
# elif env_id == "mini-catch":
#     env = catch.Wrapper(
#         data=data, env=catch.Env(columns=4, rows=5, seed=seed), hint=hint
#     )
register(
    "mini-catch-v0",
    entry_point="environments.icpi.catch:create",
    kwargs={"columns": 4, "rows": 5, "seed": 0},
)
# elif env_id == "point-mass":
#     max_steps = 8
#     env = TimeLimit(
#         point_mass.Env(
#             data=data,
#             hint=hint,
#             max_distance=6,
#             _max_trajectory=max_steps,
#             pos_threshold=2,
#             random_seed=seed,
#         ),
#         max_episode_steps=max_steps,
#     )
register(
    "point-mass-v0",
    entry_point="environments.icpi.point_mass:create",
    kwargs={
        "max_distance": 6,
        "max_trajectory": 8,
        "pos_threshold": 2,
        "random_seed": 0,
        "max_episode_step": 8,
    },
)
# elif env_id == "space-invaders":
#     env = space_invaders.Env(
#         data=data,
#         width=4,
#         height=5,
#         n_aliens=2,
#         random_seed=seed,
#         hint=hint,
#     )
register(
    "space-invaders-v0",
    entry_point="environments.icpi.space_invaders:create",
    kwargs={
        "width": 4,
        "height": 5,
        "n_aliens": 2,
        "random_seed": 0,
    },
)
