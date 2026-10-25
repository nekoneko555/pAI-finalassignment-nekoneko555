"""Fallback visualization draws actual MuJoCo body positions, never fake motion."""

import os
import textwrap
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=32)
def font(size=22):
    candidates = [
        os.environ.get("ARM_FONT", ""),
        "C:/Windows/Fonts/meiryo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in candidates:
        if name and Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def caption(
    frame,
    lines,
    footer="MuJoCo physics | real time 1x | end-effector motion; no grasping",
):
    canvas = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 100), fill=(12, 20, 34))
    draw.rectangle(
        (0, canvas.height - 40, canvas.width, canvas.height), fill=(12, 20, 34)
    )
    for i, line in enumerate(lines[:3]):
        draw.text((22, 8 + i * 29), str(line), font=font(19), fill=(236, 243, 250))
    draw.text((18, canvas.height - 32), footer, font=font(16), fill=(166, 188, 211))
    return np.asarray(canvas)


def title_frame(lines, width=960, height=640):
    img = Image.new("RGB", (width, height), (12, 20, 34))
    d = ImageDraw.Draw(img)
    d.rectangle((45, 80, 53, height - 80), fill=(49, 200, 184))
    y = 90
    for line in lines:
        for wrapped in textwrap.wrap(str(line), width=62) or [""]:
            current = ""
            for character in wrapped:
                if (
                    current
                    and d.textlength(current + character, font=font(25)) > width - 120
                ):
                    d.text((80, y), current, font=font(25), fill=(232, 240, 248))
                    y += 42
                    current = ""
                current += character
            d.text((80, y), current, font=font(25), fill=(232, 240, 248))
            y += 42
    return np.asarray(img)


def render_2d(env):
    img = Image.new("RGB", (env.width, env.height), (18, 28, 43))
    d = ImageDraw.Draw(img)
    scale = min(env.width, env.height) * 0.58

    def xy(p):
        return (env.width * 0.50 + p[0] * scale, env.height * 0.57 - p[1] * scale)

    def circle(p, r, fill):
        x, y = xy(p)
        d.ellipse(
            (x - r * scale, y - r * scale, x + r * scale, y + r * scale), fill=fill
        )

    for v in np.arange(-0.8, 0.9, 0.1):
        d.line([xy((v, -0.8)), xy((v, 0.8))], fill=(28, 42, 59))
        d.line([xy((-0.8, v)), xy((0.8, v))], fill=(28, 42, 59))
    if env.obstacle is not None:
        circle(env.obstacle[:2], env.obstacle[2], (242, 78, 92))
    circle(env.goal, env.cfg["goal_tolerance"], (113, 92, 33))
    circle(env.goal, 0.025, (251, 201, 67))
    a = env.data.body("upper").xpos[:2]
    b = env.data.body("forearm").xpos[:2]
    c = env.tip
    d.line([xy(a), xy(b)], fill=(36, 186, 218), width=max(1, int(0.044 * scale)))
    d.line([xy(b), xy(c)], fill=(67, 222, 169), width=max(1, int(0.044 * scale)))
    circle(a, 0.036, (138, 156, 176))
    circle(b, 0.027, (217, 231, 243))
    circle(c, 0.025, (249, 252, 255))
    return np.asarray(img)
