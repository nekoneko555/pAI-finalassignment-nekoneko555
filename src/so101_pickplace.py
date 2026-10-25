"""Physical SO-101 pick/place: IK motion, contact grasp, free object, no welds."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import imageio.v2 as imageio
import mujoco
import numpy as np

from .config import ROOT
from .so101_env import ARM_JOINTS, MODES, model_sources
from .rendering import caption
from .render_video import verify_video


def make_pick_model():
    xml, assets = model_sources()
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    world.find("geom[@name='desk']").set("pos", "0 0 0")
    world.find("body[@name='reach_goal']/geom").set("rgba", "0 0 0 0")
    cube = ET.SubElement(world, "body", name="parcel", pos="0.23 -0.09 0.018")
    ET.SubElement(cube, "freejoint", name="parcel_free")
    ET.SubElement(
        cube,
        "geom",
        name="parcel_geom",
        type="box",
        size="0.018 0.018 0.018",
        mass="0.035",
        friction="1.2 0.01 0.001",
        condim="6",
        solref="0.008 1",
        rgba="0.95 0.32 0.16 1",
    )
    for name, xy, color in [
        ("A", "0.23 -0.09", "0.9 0.55 0.1 0.8"),
        ("B", "0.23 0.09", "0.1 0.75 0.6 0.8"),
    ]:
        ET.SubElement(
            world,
            "geom",
            name=f"zone_{name}",
            type="box",
            pos=f"{xy} 0.0005",
            size="0.047 0.047 0.0004",
            contype="0",
            conaffinity="0",
            rgba=color,
        )
    return mujoco.MjModel.from_xml_string(
        ET.tostring(root, encoding="unicode"), assets=assets
    )


class PickPlaceSimulation:
    def __init__(self, mode="standard"):
        if mode not in MODES:
            raise ValueError(mode)
        self.model = make_pick_model()
        self.data = mujoco.MjData(self.model)
        self.qids = np.array([self.model.joint(n).qposadr[0] for n in ARM_JOINTS])
        self.vids = np.array([self.model.joint(n).dofadr[0] for n in ARM_JOINTS])
        self.aids = np.array([self.model.actuator(n).id for n in ARM_JOINTS])
        self.low = self.model.actuator_ctrlrange[self.aids, 0] + 0.025
        self.high = self.model.actuator_ctrlrange[self.aids, 1] - 0.025
        self.grip_aid = self.model.actuator("gripper").id
        self.grip_qid = self.model.joint("gripper").qposadr[0]
        self.tip_id = self.model.site("gripperframe").id
        self.cube_id = self.model.body("parcel").id
        self.cube_gid = self.model.geom("parcel_geom").id
        self.cube_qid = self.model.joint("parcel_free").qposadr[0]
        self.fixed_jaw = self.model.body("gripper").id
        self.moving_jaw = self.model.body("moving_jaw_so101_v1").id
        self.mode = mode
        self.speed, self.acceleration = MODES[mode]
        self.dt = 0.04
        self.renderer = None
        self.reset()

    @property
    def tip(self):
        return self.data.site_xpos[self.tip_id].copy()

    @property
    def object_position(self):
        return self.data.xpos[self.cube_id].copy()

    def reset(self, a=(0.23, -0.09), b=(0.23, 0.09)):
        self.a, self.b = np.asarray(a, float), np.asarray(b, float)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qids] = [0, -0.5, 0.3, 1.77, 1.61947]
        self.data.qpos[self.qids] = np.clip(
            self.data.qpos[self.qids], self.low, self.high
        )
        self.data.qpos[self.grip_qid] = 0.65
        self.data.qpos[self.cube_qid : self.cube_qid + 7] = [*self.a, 0.018, 1, 0, 0, 0]
        self.data.ctrl[self.aids] = self.data.qpos[self.qids]
        self.data.ctrl[self.grip_aid] = 0.65
        for name, xy in [("A", self.a), ("B", self.b)]:
            self.model.geom_pos[self.model.geom(f"zone_{name}").id, :2] = xy
        mujoco.mj_forward(self.model, self.data)
        self.command = self.data.qpos[self.qids].copy()
        self.velocity = np.zeros(5)
        self.elapsed = 0
        self.max_height = self.object_position[2]
        self.lifted = False
        self.both_contact_while_lifted = False
        self.trace = []

    def inverse_pose(self, xyz, start=None):
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        q = (
            self.data.qpos[self.qids].copy()
            if start is None
            else np.asarray(start).copy()
        )
        for _ in range(200):
            scratch.qpos[self.qids] = q
            mujoco.mj_forward(self.model, scratch)
            pos = scratch.site_xpos[self.tip_id]
            ep = np.asarray(xyz) - pos
            orientation_error = np.array([np.pi / 2 - q[1:4].sum(), 1.61947 - q[4]])
            if (
                np.linalg.norm(ep) < 0.0004
                and np.linalg.norm(orientation_error) < 0.005
            ):
                break
            jp = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, scratch, jp, None, self.tip_id)
            j = np.vstack(
                [jp[:, self.vids], [0, 0.12, 0.12, 0.12, 0], [0, 0, 0, 0, 0.12]]
            )
            error = np.r_[ep, 0.12 * orientation_error]
            dq = j.T @ np.linalg.solve(j @ j.T + 0.00001 * np.eye(5), error)
            q = np.clip(q + np.clip(dq, -0.1, 0.1), self.low, self.high)
        if np.linalg.norm(np.asarray(xyz) - pos) > 0.012:
            raise ValueError(
                f"Unreachable grasp pose: {xyz}, residual={np.linalg.norm(np.asarray(xyz) - pos):.4f}"
            )
        return q

    def jaw_contacts(self):
        touched = set()
        for c in self.data.contact[: self.data.ncon]:
            if self.cube_gid in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.cube_gid else c.geom1
                if c.dist < 0.001:
                    touched.add(int(self.model.geom_bodyid[other]))
        return self.fixed_jaw in touched, self.moving_jaw in touched

    def step(self, target_q, grip, stage, residual=None):
        previous_tip = self.tip
        error = np.asarray(target_q) - self.command
        # Braking speed keeps both modes stable near a waypoint.
        desired = np.sign(error) * np.minimum(
            self.speed, np.sqrt(2 * self.acceleration * np.abs(error))
        )
        if residual is not None:
            residual = np.asarray(residual, dtype=float)
            if residual.shape != (5,) or not np.isfinite(residual).all():
                raise ValueError("PPO residual must contain five finite values")
            # Bounded residual RL: modify each joint's speed profile around IK tracking.
            # The sign toward the waypoint is preserved; PPO does not invent a grasp.
            desired *= 1 + 0.35 * np.clip(residual, -1, 1)
            desired = np.clip(desired, -self.speed, self.speed)
        self.velocity = np.clip(
            desired,
            self.velocity - self.acceleration * self.dt,
            self.velocity + self.acceleration * self.dt,
        )
        delta = self.velocity * self.dt
        delta = np.where(np.abs(delta) > np.abs(error), error, delta)
        self.command = np.clip(self.command + delta, self.low, self.high)
        self.data.ctrl[self.aids] = self.command
        # The gripper closes through a position servo; the object remains a free body.
        self.data.ctrl[self.grip_aid] = np.clip(grip, -0.15, 0.75)
        for _ in range(8):
            mujoco.mj_step(self.model, self.data)
        self.elapsed += self.dt
        obj = self.object_position
        fixed, moving = self.jaw_contacts()
        self.max_height = max(self.max_height, float(obj[2]))
        self.lifted |= obj[2] > 0.06
        self.both_contact_while_lifted |= bool(fixed and moving and obj[2] > 0.06)
        row = {
            "time_s": round(self.elapsed, 6),
            "stage": stage,
            "object_x": obj[0],
            "object_y": obj[1],
            "object_z": obj[2],
            "tip_x": self.tip[0],
            "tip_y": self.tip[1],
            "tip_z": self.tip[2],
            "tip_speed_m_s": float(np.linalg.norm(self.tip - previous_tip) / self.dt),
            "fixed_jaw_contact": fixed,
            "moving_jaw_contact": moving,
            "gripper_q": float(self.data.qpos[self.grip_qid]),
            **{f"q{i}": float(x) for i, x in enumerate(self.data.qpos[self.qids])},
            **{f"velocity_command{i}": float(x) for i, x in enumerate(self.velocity)},
            **{
                f"ppo_residual{i}": float(x)
                for i, x in enumerate(np.zeros(5) if residual is None else residual)
            },
        }
        self.trace.append(row)
        return row

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=640, width=960)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.15, 0, 0.12]
        camera.distance, camera.azimuth, camera.elevation = 0.85, 135, -28
        option = mujoco.MjvOption()
        option.geomgroup[3:5] = 0
        # Sites remain available to control code, but debug markers stay hidden.
        option.sitegroup[:] = 0
        self.renderer.update_scene(self.data, camera=camera, scene_option=option)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None

    def grasp_site(self, xy, height):
        angle = np.arctan2(xy[1], xy[0])
        return np.array(
            [xy[0] + 0.010 * np.sin(angle), xy[1] - 0.010 * np.cos(angle), height]
        )

    def run(self, writer=None):
        stages = [
            ("APPROACH A", self.grasp_site(self.a, 0.07), 0.65, 0.25),
            ("DESCEND", self.grasp_site(self.a, 0.025), 0.65, 0.35),
            ("GRASP", self.grasp_site(self.a, 0.025), -0.15, 1.0),
            ("LIFT", self.grasp_site(self.a, 0.075), -0.15, 0.3),
            ("TRANSPORT TO B", self.grasp_site(self.b, 0.075), -0.15, 0.3),
            ("LOWER AT B", self.grasp_site(self.b, 0.026), -0.15, 0.3),
            ("RELEASE", self.grasp_site(self.b, 0.026), 0.65, 0.8),
            ("RETREAT", self.grasp_site(self.b, 0.075), 0.65, 1.0),
        ]
        for name, xyz, grip, dwell in stages:
            q = self.inverse_pose(xyz)
            stable = 0
            for _ in range(350):
                row = self.step(q, grip, name)
                arrived = (
                    np.max(np.abs(self.command - q)) < 0.002
                    and np.max(np.abs(self.data.qpos[self.qids] - q)) < 0.035
                )
                stable = stable + 1 if arrived else 0
                if writer is not None:
                    frame = caption(
                        self.render(),
                        [
                            f"SO-101 | {name} | mode={self.mode} | t={self.elapsed:.2f}s",
                            "IK control / physical contact grasp / A (orange) -> B (green)",
                            f"object height={row['object_z']:.3f}m | jaw contacts={int(row['fixed_jaw_contact'])}/{int(row['moving_jaw_contact'])}",
                        ],
                        footer="MuJoCo simulation | real time 1x | free object; no attachment",
                    )
                    writer.append_data(frame)
                if stable * self.dt >= dwell:
                    break
            print(
                name,
                "time",
                round(self.elapsed, 2),
                "object",
                self.object_position,
                "contacts",
                self.jaw_contacts(),
                flush=True,
            )
        obj = self.object_position
        fixed, moving = self.jaw_contacts()
        result = {
            "mode": self.mode,
            "elapsed_time_s": round(self.elapsed, 3),
            "lifted": bool(self.lifted),
            "both_jaws_contact_while_lifted": bool(self.both_contact_while_lifted),
            "max_object_height_m": self.max_height,
            "final_object_position": obj.tolist(),
            "target_b": self.b.tolist(),
            "final_xy_error_m": float(np.linalg.norm(obj[:2] - self.b)),
            "released": bool(
                not fixed and not moving and self.data.qpos[self.grip_qid] > 0.5
            ),
            "object_is_free_body": True,
            "weld_constraints": int(self.model.neq),
            "controller": "IK + staged gripper",
        }
        result["success"] = bool(
            result["lifted"]
            and result["both_jaws_contact_while_lifted"]
            and result["released"]
            and result["final_xy_error_m"] < 0.035
            and 0.01 < obj[2] < 0.035
        )
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, default="standard")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "videos/so101_pickplace_ik.mp4"
    )
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    sim = PickPlaceSimulation(args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.no_video:
            result = sim.run()
        else:
            with imageio.get_writer(
                str(args.output), fps=25, codec="libx264", quality=8, macro_block_size=1
            ) as writer:
                result = sim.run(writer)
            result["video"] = verify_video(args.output)
        import pandas as pd

        pd.DataFrame(sim.trace).to_csv(
            args.output.with_suffix(".trajectory.csv"), index=False
        )
        args.output.with_suffix(".json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
    finally:
        sim.close()


if __name__ == "__main__":
    main()
