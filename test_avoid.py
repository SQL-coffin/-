#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用合成图像（灰色地面 + 红色障碍）测试感知和规划，不需要树莓派、摄像头和 pigpio。

运行：
  python3 test_avoid.py        # 直接运行，打印每个场景的结果
  python3 -m pytest -q         # 或者用 pytest
"""

import math

import cv2
import numpy as np

from rc_avoid import Config, Perception, Planner

W, H = 320, 240


def scene(box=None, seed=0):
    """灰色地面 + 顶部浅色墙/天空，box=((x0,y0),(x1,y1)) 时画一个红色障碍"""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), (90, 110, 120), np.uint8)
    img = cv2.add(img, rng.integers(0, 12, img.shape, dtype=np.uint8))
    img[:40] = (200, 180, 150)
    if box:
        cv2.rectangle(img, box[0], box[1], (20, 20, 200), -1)
    return img


def make():
    cfg = Config()
    p = Perception(cfg)
    pl = Planner(cfg, len(p.scan_cols))
    p.process(scene())  # 先看一帧干净地面，初始化参考地面直方图
    return cfg, p, pl


def run(p, pl, box, now=100.0):
    pl.heading = 0.0
    pl.escape_until = 0
    _, _, pts = p.process(scene(box))
    return pts, pl.step(pts, now=now)


def test_inverse_perspective():
    cfg, p, _ = make()
    uv = np.array([[p.cx, p.cy], [p.cx, H - 1.0], [p.cx, 10.0], [0, H - 1.0]])
    pts, valid = p.pixel_to_ground(uv)
    # 画面中心应投影到 h / tan(俯角)
    assert abs(pts[0][0] - cfg.cam_height_m / math.tan(math.radians(cfg.cam_pitch_deg))) < 1e-3
    assert 0.10 < pts[1][0] < 0.15      # 最底行约 0.12m
    assert not valid[2]                 # 地平线以上的点丢弃
    assert pts[3][1] > 0                # 画面左侧对应车体 Y>0（左）


def test_open_ground_drives_straight():
    _, p, pl = make()
    pts, o = run(p, pl, None)
    assert len(pts) == 0
    assert o["mode"] == "DRIVE" and o["throttle"] > 0 and abs(o["heading_deg"]) < 1


def test_obstacle_right_turns_left():
    _, p, pl = make()
    _, o = run(p, pl, ((200, 60), (319, 140)))
    assert o["fy"] > 0 and o["heading_deg"] > 0


def test_obstacle_left_turns_right():
    _, p, pl = make()
    _, o = run(p, pl, ((0, 60), (120, 140)))
    assert o["fy"] < 0 and o["heading_deg"] < 0


def test_closer_obstacle_slower():
    _, p, pl = make()
    _, far = run(p, pl, ((120, 40), (200, 100)))
    _, mid = run(p, pl, ((200, 60), (319, 140)))
    _, opn = run(p, pl, None)
    assert opn["throttle"] > far["throttle"] > mid["throttle"] >= 0


def test_center_blocked_escapes_backward():
    _, p, pl = make()
    _, o = run(p, pl, ((110, 60), (210, 140)))
    assert o["mode"] == "ESCAPE" and o["throttle"] < 0


def test_sample_covered_uses_reference():
    """障碍完全盖住底部样本区时，应改用参考地面直方图，仍然识别出障碍并倒车"""
    _, p, pl = make()
    pts, o = run(p, pl, ((0, 60), (319, 239)))
    assert p.sample_polluted
    assert len(pts) > 0 and o["mode"] == "ESCAPE"


def test_escape_sequence():
    """倒车时序：双击倒车（反向 -> 回中 -> 反向）后持续倒车"""
    cfg, p, pl = make()
    pl.escape_until = 0
    _, _, pts = p.process(scene(((110, 60), (210, 140))))
    thr = [pl.step(pts, now=t)["throttle"] for t in (0, .05, .15, .3)]
    assert thr[0] < 0 and thr[2] == 0.0 and thr[3] < 0


def _print_scenes():
    _, p, pl = make()
    for name, box in [("open", None), ("right far", ((220, 50), (319, 110))),
                      ("right mid", ((200, 60), (319, 140))), ("left mid", ((0, 60), (120, 140))),
                      ("center far", ((120, 40), (200, 100))), ("center mid", ((110, 60), (210, 140))),
                      ("wall covers sample", ((0, 60), (319, 239)))]:
        pts, o = run(p, pl, box)
        print(f"{name:18s} n={len(pts):3d} mind={o['min_dist']:.2f} F=({o['fx']:+.2f},{o['fy']:+.2f}) "
              f"hdg={o['heading_deg']:+.1f} thr={o['throttle']:+.2f} {o['mode']} "
              f"polluted={p.sample_polluted}")


if __name__ == "__main__":
    _print_scenes()
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"全部 {len(tests)} 项通过")
