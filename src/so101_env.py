"""Pinned SO-101 model sources with an additive scene, gravity and servos."""

from functools import lru_cache
import xml.etree.ElementTree as ET
import mujoco

from .config import ROOT

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
MODES = {"standard": (1.2, 3.0), "fast": (1.2, 3.0), "careful": (0.4, 0.8)}


@lru_cache(maxsize=1)
def model_sources():
    folder = ROOT / "assets/so101/upstream"
    root = ET.fromstring((folder / "so101.xml").read_text(encoding="utf-8"))
    root.find("visual").append(ET.Element("global", offwidth="960", offheight="640"))
    root.find("visual").append(
        ET.Element("headlight", ambient="0.45 0.45 0.45", diffuse="0.7 0.7 0.7")
    )
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="0 -0.4 1.5", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(
        world,
        "geom",
        name="desk",
        type="plane",
        size="0.65 0.65 0.02",
        pos="0 0 -0.005",
        rgba="0.17 0.22 0.29 1",
    )
    marker = ET.SubElement(
        world, "body", name="reach_goal", mocap="true", pos="0.2 0 0.2"
    )
    ET.SubElement(
        marker,
        "geom",
        name="goal_marker",
        type="sphere",
        size="0.015",
        rgba="0.1 0.95 0.75 0.75",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        root.find(".//body[@name='gripper']"),
        "site",
        name="reach_tip",
        pos="0 0 -0.10",
        size="0.006",
        rgba="0.2 0.8 1 1",
    )
    assets = {p.name: p.read_bytes() for p in (folder / "assets").glob("*.stl")}
    return ET.tostring(root, encoding="unicode"), assets


def make_model():
    xml, assets = model_sources()
    return mujoco.MjModel.from_xml_string(xml, assets=assets)
