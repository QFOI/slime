"""Batched Panda worlds on one GPU: command state machine, IK, physics, cameras.

One control frame (60 Hz) is one CUDA graph over all worlds: per-world
controller, batched IK, joint-target write, collision + physics substeps, and a
status kernel that decides when each world's command has settled. The host
starts commands with small begin_* kernels between frames and reads back one
int per world after each frame. Worlds without a command hold their last
commanded pose while time advances for the others, as a real arm would while
its policy is thinking.
"""

from __future__ import annotations

import dataclasses
import functools

import newton
import newton.ik as ik
import numpy as np
import warp as wp
from newton.sensors import SensorCamera

from .scene import (
    ARM_COORDS,
    BODIES_PER_WORLD,
    COORDS_PER_WORLD,
    CUBE_BODY,
    DEFAULT_SCENE,
    DOFS_PER_WORLD,
    FINGER_OPEN,
    HAND_BODY,
    ROBOT_COORDS,
    TCP_BODY,
    Layout,
    SceneConfig,
    build_model,
    sample_layout,
)

IDLE = wp.constant(0)
MOVE = wp.constant(1)
GRIPPER = wp.constant(2)
EVALUATE = wp.constant(3)

RUNNING = wp.constant(0)
COMPLETED = wp.constant(1)
PARTIAL = wp.constant(2)

EVAL_LIFT = wp.constant(0)
EVAL_REACH = wp.constant(1)


@dataclasses.dataclass(frozen=True)
class SimConfig:
    fps: int = 60
    substeps: int = 10
    ik_iterations: int = 12
    # A move is complete once the TCP is within tolerance and the arm is still
    # for settle_frames frames; partial if that never happens within timeout.
    position_tol: float = 0.002
    rotation_tol: float = float(np.radians(1.0))
    joint_speed_tol: float = 0.02
    settle_frames: int = 3
    settle_timeout: float = 1.0
    gripper_min_time: float = 0.3
    gripper_timeout: float = 1.5
    # Evaluation hold: sample_count samples every sample_every frames (20 Hz, 0.5 s).
    sample_every: int = 3
    sample_count: int = 10
    use_graph: bool = True


DEFAULT_SIM = SimConfig()


@dataclasses.dataclass(frozen=True)
class CameraSpec:
    """Pinhole camera; fixed in the base frame, or mounted on a robot body."""

    name: str
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    mount_body: int | None = None  # eye/target/up are body-local when mounted
    fov_deg: float = 55.0
    width: int = 256
    height: int = 256


DEFAULT_CAMERAS = (
    CameraSpec("front", eye=(1.25, 0.55, 0.75), target=(0.42, 0.0, 0.12), fov_deg=55.0),
    # Hand frame: +z points from the flange toward the fingertips, fingers open along y.
    CameraSpec(
        "wrist",
        eye=(0.085, 0.0, 0.04),
        target=(0.0, 0.0, 0.18),
        up=(1.0, 0.0, 0.0),
        fov_deg=70.0,
        mount_body=HAND_BODY,
    ),
)


def look_at(eye, target, up) -> wp.transformf:
    """Camera frame looking along local -Z with +Y up (SensorCamera convention)."""
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    rotation = np.stack([right, np.cross(right, forward), -forward], axis=1)
    return wp.transformf(wp.vec3f(*eye.tolist()), wp.quat_from_matrix(wp.mat33(*rotation.flatten().tolist())))


@wp.func
def quat_distance(a: wp.quat, b: wp.quat) -> float:
    d = wp.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
    return 2.0 * wp.acos(wp.min(d, 1.0))


@wp.kernel
def reset_joints_kernel(
    worlds: wp.array(dtype=int),
    q: wp.array2d(dtype=float),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_target_q: wp.array(dtype=float),
    ik_q: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    w = worlds[i]
    joint_q[w * COORDS_PER_WORLD + j] = q[i, j]
    if j < DOFS_PER_WORLD:
        joint_qd[w * DOFS_PER_WORLD + j] = 0.0
    if j < ROBOT_COORDS:
        joint_target_q[w * COORDS_PER_WORLD + j] = q[i, j]
        ik_q[w, j] = q[i, j]


@wp.kernel
def reset_world_kernel(
    worlds: wp.array(dtype=int),
    eval_kind_in: wp.array(dtype=int),
    goal_pos_in: wp.array(dtype=wp.vec3),
    goal_rot_in: wp.array(dtype=wp.quat),
    body_q: wp.array(dtype=wp.transform),
    mode: wp.array(dtype=int),
    result: wp.array(dtype=int),
    elapsed: wp.array(dtype=float),
    settle: wp.array(dtype=int),
    hold_pos: wp.array(dtype=wp.vec3),
    hold_rot: wp.array(dtype=wp.quat),
    finger_target: wp.array(dtype=float),
    eval_kind: wp.array(dtype=int),
    goal_pos: wp.array(dtype=wp.vec3),
    goal_rot: wp.array(dtype=wp.quat),
):
    i = wp.tid()
    w = worlds[i]
    tcp = body_q[w * BODIES_PER_WORLD + TCP_BODY]
    hold_pos[w] = wp.transform_get_translation(tcp)
    hold_rot[w] = wp.transform_get_rotation(tcp)
    finger_target[w] = FINGER_OPEN
    mode[w] = IDLE
    result[w] = RUNNING
    elapsed[w] = 0.0
    settle[w] = 0
    eval_kind[w] = eval_kind_in[i]
    goal_pos[w] = goal_pos_in[i]
    goal_rot[w] = goal_rot_in[i]


@wp.kernel
def begin_move_kernel(
    worlds: wp.array(dtype=int),
    translation: wp.array(dtype=wp.vec3),
    rotvec: wp.array(dtype=wp.vec3),
    duration_in: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    mode: wp.array(dtype=int),
    result: wp.array(dtype=int),
    elapsed: wp.array(dtype=float),
    settle: wp.array(dtype=int),
    duration: wp.array(dtype=float),
    start_pos: wp.array(dtype=wp.vec3),
    start_rot: wp.array(dtype=wp.quat),
    target_pos: wp.array(dtype=wp.vec3),
    target_rot: wp.array(dtype=wp.quat),
):
    # Anchored to the measured TCP; the rotation is a base-frame axis-angle
    # applied about the TCP, i.e. R_target = R(rotvec) * R_measured.
    i = wp.tid()
    w = worlds[i]
    tcp = body_q[w * BODIES_PER_WORLD + TCP_BODY]
    p = wp.transform_get_translation(tcp)
    q = wp.transform_get_rotation(tcp)
    angle = wp.length(rotvec[i])
    dq = wp.quat_identity()
    if angle > 1.0e-9:
        dq = wp.quat_from_axis_angle(rotvec[i] / angle, angle)
    qt = wp.normalize(dq * q)
    if qt[0] * q[0] + qt[1] * q[1] + qt[2] * q[2] + qt[3] * q[3] < 0.0:
        qt = -qt
    start_pos[w] = p
    start_rot[w] = q
    target_pos[w] = p + translation[i]
    target_rot[w] = qt
    duration[w] = duration_in[i]
    elapsed[w] = 0.0
    settle[w] = 0
    result[w] = RUNNING
    mode[w] = MOVE


@wp.kernel
def begin_gripper_kernel(
    worlds: wp.array(dtype=int),
    width: wp.array(dtype=float),
    mode: wp.array(dtype=int),
    result: wp.array(dtype=int),
    elapsed: wp.array(dtype=float),
    settle: wp.array(dtype=int),
    finger_target: wp.array(dtype=float),
):
    i = wp.tid()
    w = worlds[i]
    finger_target[w] = width[i]
    elapsed[w] = 0.0
    settle[w] = 0
    result[w] = RUNNING
    mode[w] = GRIPPER


@wp.kernel
def begin_evaluate_kernel(
    worlds: wp.array(dtype=int),
    mode: wp.array(dtype=int),
    result: wp.array(dtype=int),
    elapsed: wp.array(dtype=float),
    eval_frames: wp.array(dtype=int),
    eval_count: wp.array(dtype=int),
    eval_bits: wp.array(dtype=int),
):
    i = wp.tid()
    w = worlds[i]
    elapsed[w] = 0.0
    eval_frames[w] = 0
    eval_count[w] = 0
    eval_bits[w] = 0
    result[w] = RUNNING
    mode[w] = EVALUATE


@wp.kernel
def control_kernel(
    dt: float,
    mode: wp.array(dtype=int),
    elapsed: wp.array(dtype=float),
    duration: wp.array(dtype=float),
    start_pos: wp.array(dtype=wp.vec3),
    start_rot: wp.array(dtype=wp.quat),
    target_pos: wp.array(dtype=wp.vec3),
    target_rot: wp.array(dtype=wp.quat),
    hold_pos: wp.array(dtype=wp.vec3),
    hold_rot: wp.array(dtype=wp.quat),
    ik_pos: wp.array(dtype=wp.vec3),
    ik_rot: wp.array(dtype=wp.vec4),
):
    w = wp.tid()
    m = mode[w]
    if m != IDLE:
        elapsed[w] = elapsed[w] + dt
    p = hold_pos[w]
    q = hold_rot[w]
    if m == MOVE:
        s = wp.min(1.0, elapsed[w] / wp.max(duration[w], 1.0e-6))
        s = s * s * (3.0 - 2.0 * s)
        p = start_pos[w] * (1.0 - s) + target_pos[w] * s
        q = wp.quat_slerp(start_rot[w], target_rot[w], s)
        hold_pos[w] = target_pos[w]
        hold_rot[w] = target_rot[w]
    ik_pos[w] = p
    ik_rot[w] = wp.vec4(q[0], q[1], q[2], q[3])


@wp.kernel
def write_targets_kernel(
    ik_q: wp.array2d(dtype=float),
    finger_target: wp.array(dtype=float),
    joint_target_q: wp.array(dtype=float),
):
    w, j = wp.tid()
    value = ik_q[w, j]
    if j >= ARM_COORDS:
        value = finger_target[w]
    joint_target_q[w * COORDS_PER_WORLD + j] = value


@wp.kernel
def status_kernel(
    body_q: wp.array(dtype=wp.transform),
    joint_qd: wp.array(dtype=float),
    mode: wp.array(dtype=int),
    result: wp.array(dtype=int),
    elapsed: wp.array(dtype=float),
    duration: wp.array(dtype=float),
    settle: wp.array(dtype=int),
    target_pos: wp.array(dtype=wp.vec3),
    target_rot: wp.array(dtype=wp.quat),
    hold_pos: wp.array(dtype=wp.vec3),
    hold_rot: wp.array(dtype=wp.quat),
    eval_kind: wp.array(dtype=int),
    goal_pos: wp.array(dtype=wp.vec3),
    goal_rot: wp.array(dtype=wp.quat),
    eval_frames: wp.array(dtype=int),
    eval_count: wp.array(dtype=int),
    eval_bits: wp.array(dtype=int),
    position_tol: float,
    rotation_tol: float,
    joint_speed_tol: float,
    settle_frames: int,
    settle_timeout: float,
    gripper_min_time: float,
    gripper_timeout: float,
    sample_every: int,
    sample_count: int,
    lift_min_z: float,
    reach_position_tol: float,
    reach_rotation_tol: float,
):
    w = wp.tid()
    m = mode[w]
    if m == IDLE:
        return
    tcp = body_q[w * BODIES_PER_WORLD + TCP_BODY]
    p = wp.transform_get_translation(tcp)
    q = wp.transform_get_rotation(tcp)
    if m == MOVE:
        if elapsed[w] >= duration[w]:
            speed = 0.0
            for j in range(ARM_COORDS):
                speed = wp.max(speed, wp.abs(joint_qd[w * DOFS_PER_WORLD + j]))
            still = speed < joint_speed_tol
            close = wp.length(target_pos[w] - p) < position_tol and quat_distance(target_rot[w], q) < rotation_tol
            if still and close:
                settle[w] = settle[w] + 1
            else:
                settle[w] = 0
            if settle[w] >= settle_frames:
                result[w] = COMPLETED
                mode[w] = IDLE
            elif elapsed[w] >= duration[w] + settle_timeout:
                # Blocked or out of reach: stop pushing and hold where the arm is.
                hold_pos[w] = p
                hold_rot[w] = q
                result[w] = PARTIAL
                mode[w] = IDLE
    elif m == GRIPPER:
        finger_speed = wp.max(
            wp.abs(joint_qd[w * DOFS_PER_WORLD + ARM_COORDS]), wp.abs(joint_qd[w * DOFS_PER_WORLD + ARM_COORDS + 1])
        )
        if finger_speed < 0.005:
            settle[w] = settle[w] + 1
        else:
            settle[w] = 0
        if (elapsed[w] >= gripper_min_time and settle[w] >= settle_frames) or elapsed[w] >= gripper_timeout:
            result[w] = COMPLETED
            mode[w] = IDLE
    elif m == EVALUATE:
        eval_frames[w] = eval_frames[w] + 1
        if eval_frames[w] % sample_every == 0:
            ok = False
            if eval_kind[w] == EVAL_LIFT:
                cube = wp.transform_get_translation(body_q[w * BODIES_PER_WORLD + CUBE_BODY])
                ok = cube[2] > lift_min_z
            else:
                ok = (
                    wp.length(goal_pos[w] - p) <= reach_position_tol
                    and quat_distance(goal_rot[w], q) <= reach_rotation_tol
                )
            if ok:
                eval_bits[w] = eval_bits[w] | (1 << eval_count[w])
            eval_count[w] = eval_count[w] + 1
            if eval_count[w] >= sample_count:
                result[w] = COMPLETED
                mode[w] = IDLE


@wp.kernel
def mounted_camera_kernel(
    body_q: wp.array(dtype=wp.transform), body: int, local: wp.transformf, out: wp.array(dtype=wp.transformf)
):
    w = wp.tid()
    out[w] = wp.transform_multiply(body_q[w * BODIES_PER_WORLD + body], local)


class _Camera:
    def __init__(self, spec: CameraSpec, sensor: SensorCamera, world_count: int, device):
        self.spec = spec
        self.rays = SensorCamera.compute_camera_rays_pinhole(
            spec.width, spec.height, camera_fov=float(np.radians(spec.fov_deg)), device=device
        )
        self.pose = look_at(spec.eye, spec.target, spec.up)
        self.transforms = wp.array([self.pose] * world_count, dtype=wp.transformf, device=device)
        self.color = sensor.allocate_image_color(world_count, spec.width, spec.height)


def _on_device(method):
    """Make the worlds' device Warp's default for temporaries and launches."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with wp.ScopedDevice(self.device):
            return method(self, *args, **kwargs)

    return wrapper


class PandaWorlds:
    """W independent Panda + cube worlds. Not thread-safe; drive from one thread."""

    def __init__(
        self,
        world_count: int,
        *,
        device=None,
        scene: SceneConfig = DEFAULT_SCENE,
        sim: SimConfig = DEFAULT_SIM,
        cameras: tuple[CameraSpec, ...] = DEFAULT_CAMERAS,
    ):
        if sim.substeps % 2:
            raise ValueError("substeps must be even so the captured graph ends in state_0")
        self.world_count = world_count
        self.scene = scene
        self.sim = sim
        self.frame_dt = 1.0 / sim.fps
        self.device = wp.get_device(device)
        self.frames = 0
        with wp.ScopedDevice(self.device):
            self._build(cameras)

    def _build(self, cameras):
        n = self.world_count
        self.model, ik_model = build_model(n, self.scene, self.device)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            iterations=20,
            ls_iterations=100,
            nconmax=1000,
            njmax=2000,
            cone="elliptic",
            impratio=1000.0,
            use_mujoco_contacts=False,
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()

        def per_world(dtype, value=None):
            return wp.zeros(n, dtype=dtype) if value is None else wp.full(n, value, dtype=dtype)

        self.mode, self.result, self.settle = per_world(int), per_world(int), per_world(int)
        self.elapsed, self.duration = per_world(float), per_world(float)
        self.start_pos, self.target_pos, self.hold_pos = per_world(wp.vec3), per_world(wp.vec3), per_world(wp.vec3)
        self.start_rot, self.target_rot, self.hold_rot = per_world(wp.quat), per_world(wp.quat), per_world(wp.quat)
        self.finger_target = per_world(float, FINGER_OPEN)
        self.eval_kind, self.goal_pos, self.goal_rot = per_world(int), per_world(wp.vec3), per_world(wp.quat)
        self.eval_frames, self.eval_count, self.eval_bits = per_world(int), per_world(int), per_world(int)

        # The IK objectives keep references to these arrays; control_kernel writes them.
        self.ik_pos, self.ik_rot = per_world(wp.vec3), per_world(wp.vec4)
        lower = np.tile(ik_model.joint_limit_lower.numpy(), n)
        upper = np.tile(ik_model.joint_limit_upper.numpy(), n)
        self.ik_q = wp.array(np.tile(ik_model.joint_q.numpy(), (n, 1)), dtype=float)
        self.ik_solver = ik.IKSolver(
            model=ik_model,
            n_problems=n,
            objectives=[
                ik.IKObjectivePosition(link_index=TCP_BODY, link_offset=wp.vec3(0.0), target_positions=self.ik_pos),
                ik.IKObjectiveRotation(
                    link_index=TCP_BODY, link_offset_rotation=wp.quat_identity(), target_rotations=self.ik_rot
                ),
                ik.IKObjectiveJointLimit(
                    joint_limit_lower=wp.array(lower, dtype=float), joint_limit_upper=wp.array(upper, dtype=float)
                ),
            ],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        self.sensor = SensorCamera(self.model)
        self.sensor.create_default_light(enable_shadows=True)
        self.sensor.default_render_config.enable_shadows = True
        self.sensor.default_clear_data = SensorCamera.ClearData(clear_color=0xFF7A6A5C, clear_albedo=0xFF000000)
        floor = [i for i, label in enumerate(self.model.shape_label) if label == "floor"]
        self.sensor.assign_checkerboard_material(shape_indices=floor)
        self.cameras = [_Camera(spec, self.sensor, n, self.device) for spec in cameras]
        self.world_indices = wp.zeros(n, dtype=wp.int32)

        all_worlds = list(range(n))
        self.reset(all_worlds, [sample_layout("lift", 0, self.scene)] * n)
        self._frame()  # compile and allocate before capture
        self.reset(all_worlds, [sample_layout("lift", 0, self.scene)] * n)
        self.graph = None
        if self.sim.use_graph:
            with wp.ScopedCapture() as capture:
                self._frame()
            self.graph = capture.graph

    def _frame(self):
        sim, n = self.sim, self.world_count
        wp.launch(
            control_kernel,
            dim=n,
            inputs=[self.frame_dt, self.mode, self.elapsed, self.duration, self.start_pos, self.start_rot],
            outputs=[self.target_pos, self.target_rot, self.hold_pos, self.hold_rot, self.ik_pos, self.ik_rot],
        )
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=sim.ik_iterations)
        wp.launch(
            write_targets_kernel,
            dim=(n, ROBOT_COORDS),
            inputs=[self.ik_q, self.finger_target],
            outputs=[self.control.joint_target_q],
        )
        self.pipeline.collide(self.state_0, self.contacts)
        for _ in range(sim.substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.frame_dt / sim.substeps)
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(
            status_kernel,
            dim=n,
            inputs=[
                self.state_0.body_q,
                self.state_0.joint_qd,
                self.mode,
                self.result,
                self.elapsed,
                self.duration,
                self.settle,
                self.target_pos,
                self.target_rot,
                self.hold_pos,
                self.hold_rot,
                self.eval_kind,
                self.goal_pos,
                self.goal_rot,
                self.eval_frames,
                self.eval_count,
                self.eval_bits,
                sim.position_tol,
                sim.rotation_tol,
                sim.joint_speed_tol,
                sim.settle_frames,
                sim.settle_timeout,
                sim.gripper_min_time,
                sim.gripper_timeout,
                sim.sample_every,
                sim.sample_count,
                self.scene.cube_half + self.scene.lift_clearance,
                self.scene.reach_position_tol,
                self.scene.reach_rotation_tol,
            ],
        )

    # ----- host API, called between frames ---------------------------------

    @_on_device
    def reset(self, worlds: list[int], layouts: list[Layout]) -> None:
        q = np.zeros((len(worlds), COORDS_PER_WORLD), dtype=np.float32)
        goal_pos = np.zeros((len(worlds), 3), dtype=np.float32)
        goal_rot = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (len(worlds), 1))
        kinds = np.zeros(len(worlds), dtype=np.int32)
        for i, layout in enumerate(layouts):
            q[i, :ARM_COORDS] = layout.arm_q
            q[i, ARM_COORDS:ROBOT_COORDS] = FINGER_OPEN
            q[i, ROBOT_COORDS : ROBOT_COORDS + 3] = layout.cube_position
            q[i, ROBOT_COORDS + 3 :] = layout.cube_quat_xyzw
            if layout.task == "reach_pose":
                kinds[i] = EVAL_REACH
                goal_pos[i], goal_rot[i] = layout.goal_position, layout.goal_quat_xyzw
        ids = wp.array(worlds, dtype=int)
        wp.launch(
            reset_joints_kernel,
            dim=(len(worlds), COORDS_PER_WORLD),
            inputs=[ids, wp.array(q, dtype=float)],
            outputs=[self.state_0.joint_q, self.state_0.joint_qd, self.control.joint_target_q, self.ik_q],
        )
        mask = np.zeros(self.world_count + 1, dtype=bool)
        mask[worlds] = True
        self.solver.reset(self.state_0, world_mask=wp.array(mask, dtype=bool), flags=0)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        wp.launch(
            reset_world_kernel,
            dim=len(worlds),
            inputs=[
                ids,
                wp.array(kinds, dtype=int),
                wp.array(goal_pos, dtype=wp.vec3),
                wp.array(goal_rot, dtype=wp.quat),
                self.state_0.body_q,
            ],
            outputs=[
                self.mode,
                self.result,
                self.elapsed,
                self.settle,
                self.hold_pos,
                self.hold_rot,
                self.finger_target,
                self.eval_kind,
                self.goal_pos,
                self.goal_rot,
            ],
        )

    @_on_device
    def begin_move(self, worlds: list[int], translation, rotvec, duration) -> None:
        wp.launch(
            begin_move_kernel,
            dim=len(worlds),
            inputs=[
                wp.array(worlds, dtype=int),
                wp.array(np.asarray(translation, dtype=np.float32).reshape(-1, 3), dtype=wp.vec3),
                wp.array(np.asarray(rotvec, dtype=np.float32).reshape(-1, 3), dtype=wp.vec3),
                wp.array(np.asarray(duration, dtype=np.float32), dtype=float),
                self.state_0.body_q,
            ],
            outputs=[
                self.mode,
                self.result,
                self.elapsed,
                self.settle,
                self.duration,
                self.start_pos,
                self.start_rot,
                self.target_pos,
                self.target_rot,
            ],
        )

    @_on_device
    def begin_gripper(self, worlds: list[int], width) -> None:
        wp.launch(
            begin_gripper_kernel,
            dim=len(worlds),
            inputs=[wp.array(worlds, dtype=int), wp.array(np.asarray(width, dtype=np.float32), dtype=float)],
            outputs=[self.mode, self.result, self.elapsed, self.settle, self.finger_target],
        )

    @_on_device
    def begin_evaluate(self, worlds: list[int]) -> None:
        wp.launch(
            begin_evaluate_kernel,
            dim=len(worlds),
            inputs=[wp.array(worlds, dtype=int)],
            outputs=[self.mode, self.result, self.elapsed, self.eval_frames, self.eval_count, self.eval_bits],
        )

    @_on_device
    def step(self) -> np.ndarray:
        """Advance every world by one control frame; return per-world result codes."""
        if self.graph is None:
            self._frame()
        else:
            wp.capture_launch(self.graph)
        self.frames += 1
        return self.result.numpy()

    @_on_device
    def snapshot(self, worlds: list[int]) -> list[dict]:
        """Measured state plus the command bookkeeping of the given worlds."""
        body_q = self.state_0.body_q.numpy().reshape(self.world_count, BODIES_PER_WORLD, 7)
        joint_q = self.state_0.joint_q.numpy().reshape(self.world_count, COORDS_PER_WORLD)
        arrays = {
            name: getattr(self, name).numpy()
            for name in ("target_pos", "target_rot", "elapsed", "finger_target", "eval_count", "eval_bits")
        }
        out = []
        for w in worlds:
            tcp, cube = body_q[w, TCP_BODY], body_q[w, CUBE_BODY]
            out.append(
                {
                    "tcp_position": tcp[:3].astype(float),
                    "tcp_quat_xyzw": tcp[3:].astype(float),
                    "cube_position": cube[:3].astype(float),
                    "cube_quat_xyzw": cube[3:].astype(float),
                    "arm_q": joint_q[w, :ARM_COORDS].astype(float),
                    "gripper_width": float(joint_q[w, ARM_COORDS] + joint_q[w, ARM_COORDS + 1]),
                    "finger_target": float(arrays["finger_target"][w]),
                    "target_position": arrays["target_pos"][w].astype(float),
                    "target_quat_xyzw": arrays["target_rot"][w].astype(float),
                    "elapsed": float(arrays["elapsed"][w]),
                    "hold_samples": [
                        bool(arrays["eval_bits"][w] >> k & 1) for k in range(int(arrays["eval_count"][w]))
                    ],
                }
            )
        return out

    @_on_device
    def render(self, worlds: list[int]) -> dict[int, dict[str, np.ndarray]]:
        """RGB uint8 images per world and camera, rendered in one batch per camera."""
        indices = np.full(self.world_count, int(SensorCamera.WorldRenderFlag.DISABLE_PRESERVE), dtype=np.int32)
        indices[worlds] = worlds
        self.world_indices.assign(indices)
        self.model.bvh_refit_shapes(self.state_0)
        images = {w: {} for w in worlds}
        for camera in self.cameras:
            if camera.spec.mount_body is not None:
                wp.launch(
                    mounted_camera_kernel,
                    dim=self.world_count,
                    inputs=[self.state_0.body_q, camera.spec.mount_body, camera.pose],
                    outputs=[camera.transforms],
                )
            self.sensor.update(
                self.state_0,
                camera.transforms,
                camera.rays,
                color_image=camera.color,
                world_indices=self.world_indices,
            )
            rgba = SensorCamera.Utils.to_rgba_from_color(camera.color).numpy()
            for w in worlds:
                images[w][camera.spec.name] = rgba[w, :, :, :3].copy()
        return images
