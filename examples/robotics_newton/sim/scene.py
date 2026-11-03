"""Panda (FR3 + Franka hand) on a table with one cube, replicated over worlds.

The robot base is the simulation origin, so the public base frame equals the
simulation frame: x forward over the table, y left, z up, table top at z = 0.
Every world shares these coordinates; worlds never interact.
"""

from __future__ import annotations

import dataclasses

import newton
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

TASKS = ("lift", "reach_pose")

# FR3 ready pose: TCP about 0.31 m in front of the base, pointing down.
ARM_HOME = (-3.68e-3, 2.39e-2, 3.68e-3, -2.3683, -1.29e-4, 2.3922, 0.7855)
FINGER_OPEN = 0.04  # per finger [m]; full opening 0.08 m

# One world's layout, verified against the finalized model in build_model().
TCP_BODY, HAND_BODY, CUBE_BODY = 11, 10, 14
LEFT_FINGER_BODY, RIGHT_FINGER_BODY = 12, 13
BODIES_PER_WORLD, COORDS_PER_WORLD, DOFS_PER_WORLD = 15, 16, 15
ARM_COORDS, ROBOT_COORDS = 7, 9

# TCP pointing straight down (hand z along -world z), fingers along world y.
DOWN = Rotation.from_euler("x", np.pi)


@dataclasses.dataclass(frozen=True)
class SceneConfig:
    table_center_xy: tuple[float, float] = (0.5, 0.0)
    table_half_xy: tuple[float, float] = (0.35, 0.45)
    floor_z: float = -0.75
    cube_half: float = 0.025
    cube_density: float = 400.0
    friction: float = 0.75
    # Lift success: cube bottom face at least this far above the table top.
    lift_clearance: float = 0.03
    # Reach success: TCP pose error bounds.
    reach_position_tol: float = 0.01
    reach_rotation_tol: float = np.radians(5.0)


DEFAULT_SCENE = SceneConfig()


@dataclasses.dataclass(frozen=True)
class Layout:
    """Initial state of one episode, fully determined by (task, seed)."""

    task: str
    seed: int
    arm_q: tuple[float, ...]
    cube_position: tuple[float, float, float]
    cube_quat_xyzw: tuple[float, float, float, float]
    goal_position: tuple[float, float, float] | None = None
    goal_quat_xyzw: tuple[float, float, float, float] | None = None


def sample_layout(task: str, seed: int, cfg: SceneConfig = DEFAULT_SCENE) -> Layout:
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {TASKS}")
    rng = np.random.default_rng([seed, TASKS.index(task)])
    arm_q = np.asarray(ARM_HOME) + rng.uniform(-0.03, 0.03, ARM_COORDS)
    cube = (rng.uniform(0.42, 0.62), rng.uniform(-0.15, 0.15), cfg.cube_half)
    cube_quat = Rotation.from_euler("z", rng.uniform(-np.pi / 4, np.pi / 4)).as_quat()
    goal_position = goal_quat = None
    if task == "reach_pose":
        goal_position = (rng.uniform(0.35, 0.6), rng.uniform(-0.2, 0.2), rng.uniform(0.12, 0.4))
        tilt_axis = np.array([*rng.normal(size=2), 0.0])
        tilt = Rotation.from_rotvec(tilt_axis / np.linalg.norm(tilt_axis) * rng.uniform(0.0, 0.3))
        yaw = Rotation.from_euler("z", rng.uniform(-np.pi / 3, np.pi / 3))
        goal_quat = tuple((tilt * yaw * DOWN).as_quat().tolist())
    return Layout(
        task=task,
        seed=seed,
        arm_q=tuple(arm_q.tolist()),
        cube_position=tuple(float(v) for v in cube),
        cube_quat_xyzw=tuple(cube_quat.tolist()),
        goal_position=goal_position,
        goal_quat_xyzw=goal_quat,
    )


def _robot_builder(cfg: SceneConfig, *, with_scene: bool) -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_urdf(
        newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
        floating=False,
        enable_self_collisions=False,
        parse_visuals_as_colliders=False,
        joint_damping_mode="passive",
    )
    # Gains, limits, armature and gravity compensation follow Newton's
    # ik_cube_stacking example, which tracks IK targets to ~1 mm / 0.5 deg.
    builder.joint_q[:ROBOT_COORDS] = [*ARM_HOME, FINGER_OPEN, FINGER_OPEN]
    builder.joint_target_q[:ROBOT_COORDS] = [*ARM_HOME, FINGER_OPEN, FINGER_OPEN]
    builder.joint_target_ke[:ROBOT_COORDS] = [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]
    builder.joint_target_kd[:ROBOT_COORDS] = [450, 450, 350, 350, 200, 200, 200, 10, 10]
    builder.joint_effort_limit[:ROBOT_COORDS] = [87, 87, 87, 87, 12, 12, 12, 100, 100]
    builder.joint_armature[:ROBOT_COORDS] = [0.3] * 4 + [0.11] * 3 + [0.15] * 2
    gravcomp_joint = builder.custom_attributes["mujoco:jnt_actgravcomp"]
    gravcomp_joint.values = {dof: True for dof in range(ARM_COORDS)}
    gravcomp_body = builder.custom_attributes["mujoco:gravcomp"]
    gravcomp_body.values = {body: 1.0 for body in range(2, 14)}
    if not with_scene:
        return builder

    contact = newton.ModelBuilder.ShapeConfig(
        margin=0.0, density=1000.0, ke=5.0e4, kd=5.0e2, kf=1.0e3, mu=cfg.friction
    )
    table_height = -cfg.floor_z
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(*cfg.table_center_xy, -0.5 * table_height), wp.quat_identity()),
        hx=cfg.table_half_xy[0],
        hy=cfg.table_half_xy[1],
        hz=0.5 * table_height,
        cfg=contact,
        color=(0.55, 0.38, 0.22),
        label="table",
    )
    pedestal = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False)
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.5 * table_height), wp.quat_identity()),
        hx=0.11,
        hy=0.11,
        hz=0.5 * table_height,
        cfg=pedestal,
        color=(0.25, 0.25, 0.28),
        label="pedestal",
    )
    cube = builder.add_body(xform=wp.transform(wp.vec3(0.5, 0.0, cfg.cube_half), wp.quat_identity()), label="cube")
    builder.add_shape_box(
        cube,
        hx=cfg.cube_half,
        hy=cfg.cube_half,
        hz=cfg.cube_half,
        cfg=newton.ModelBuilder.ShapeConfig(margin=0.0, density=cfg.cube_density, mu=cfg.friction),
        color=(0.85, 0.1, 0.1),
        label="cube",
    )
    return builder


def build_model(world_count: int, cfg: SceneConfig, device=None) -> tuple[newton.Model, newton.Model]:
    """Return (scene model with world_count worlds, single-robot model for IK)."""
    template = _robot_builder(cfg, with_scene=True)
    scene = newton.ModelBuilder()
    scene.replicate(template, world_count)
    scene.add_ground_plane(height=cfg.floor_z, label="floor")
    model = scene.finalize(device=device)
    ik_model = _robot_builder(cfg, with_scene=False).finalize(device=device)

    expected = {
        "bodies": (model.body_count, BODIES_PER_WORLD),
        "coords": (model.joint_coord_count, COORDS_PER_WORLD),
        "dofs": (model.joint_dof_count, DOFS_PER_WORLD),
    }
    for name, (total, per_world) in expected.items():
        if total != per_world * world_count:
            raise RuntimeError(f"Unexpected {name} layout: {total} != {per_world} x {world_count}")
    labels = model.body_label
    if not (
        labels[TCP_BODY].endswith("fr3_hand_tcp")
        and labels[HAND_BODY].endswith("fr3_hand")
        and labels[LEFT_FINGER_BODY].endswith("fr3_leftfinger")
        and labels[RIGHT_FINGER_BODY].endswith("fr3_rightfinger")
    ):
        raise RuntimeError("Franka body order changed; update TCP_BODY/HAND_BODY")
    if ik_model.joint_coord_count != ROBOT_COORDS:
        raise RuntimeError("IK model must contain only the robot")
    return model, ik_model
