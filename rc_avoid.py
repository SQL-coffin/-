#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
树莓派 RC 模型车 单目视觉自动避障

流程：
  摄像头 -> 地面分割(HSV 直方图反投影) -> 逐列找障碍接触点 -> 逆透视投影到车体坐标
        -> 人工势场(引力+斥力) -> 转向/油门 -> pigpio 输出舵机/电调 PWM

运行前：
  sudo pigpiod                      # 启动 pigpio 守护进程
  python3 rc_avoid.py --debug       # 带调试窗口（需要桌面/VNC）
  python3 rc_avoid.py --dry-run --debug   # 不输出 PWM，只看感知和规划结果（建议先这样调）

坐标约定：车体坐标 X 向前、Y 向左，单位米；heading 为正表示向左转。

【安全提示】首次上电务必把车架空（轮子离地），确认舵机方向、油门方向、中位都正确后再落地。
"""

import argparse
import math
import signal
import time
from dataclasses import dataclass

import cv2
import numpy as np


# ============================================================================
# 参数配置（标 [标定] 的需要按实车实测修改）
# ============================================================================
@dataclass
class Config:
    # ---------------- 摄像头 ----------------
    width: int = 320
    height: int = 240
    hfov_deg: float = 62.0          # [标定] 水平视场角。Pi Camera v2 约 62°，v3 约 66°，广角版更大
    cam_height_m: float = 0.12      # [标定] 镜头光心离地高度，用尺子量
    cam_pitch_deg: float = 20.0     # [标定] 向下俯角。可在地上放已知距离的标记，调到投影距离一致
    camera_index: int = 0           # cv2.VideoCapture 回退时使用的设备号

    # ---------------- 地面分割 ----------------
    sample_h_frac: float = 0.15     # 地面样本区域：画面底部中间，高 15%
    sample_w_frac: float = 0.30     # 宽 30%（确保这块区域正常情况下只看到地面，不被车头挡住）
    hist_bins_h: int = 30           # H 通道直方图 bin 数
    hist_bins_s: int = 32           # S 通道直方图 bin 数
    backproj_thresh: int = 40       # [标定] 反投影二值化阈值，地面识别不全就调低，误把障碍当地面就调高
    morph_kernel: int = 5           # 形态学核大小（闭运算填洞，开运算去噪点）
    scan_step: int = 4              # 每隔几列扫描一次
    # 样本污染保护（附加功能）：障碍物贴近车头、盖住样本区时，样本直方图会变成"障碍的颜色"，
    # 导致障碍被当成地面。这里维护一个缓慢更新的参考地面直方图，当前样本与参考的相关度
    # 低于阈值时改用参考直方图。启动时请让车停在干净地面上（参考直方图取自前几帧）。
    ref_hist_corr_min: float = 0.3  # [标定] 相关度阈值，地面颜色变化大的场地可调低
    ref_hist_update: float = 0.05   # 参考直方图更新速率

    # ---------------- 距离过滤 ----------------
    max_range_m: float = 2.0        # 超过此距离的点丢弃

    # ---------------- 人工势场 ----------------
    goal_angle_deg: float = 0.0     # 目标方向，0 = 正前方（预留：可由遥控器输入覆盖）
    k_att: float = 1.0              # 引力增益
    k_rep: float = 0.02             # [标定] 斥力增益，太小撞上去，太大会在开阔处也绕来绕去
    d0_m: float = 0.8               # 斥力影响半径
    rep_norm_cols: float = 20.0     # 斥力归一化：乘以 rep_norm_cols / 扫描列数
    min_fx: float = 0.05            # 计算 heading 时 Fx 的下限，避免合力朝后时直接倒车
    max_steer_deg: float = 30.0     # [标定] 转向限幅，应与舵机实际最大转角对应
    steer_lpf_alpha: float = 0.5    # 一阶低通系数：new = a*old + (1-a)*raw

    # ---------------- 油门策略 ----------------
    stop_dist_m: float = 0.25       # 最近障碍小于此距离油门为 0
    slow_dist_m: float = 0.8        # 小于此距离开始线性减速
    max_throttle: float = 0.35      # [标定] 最大油门（归一化 0~1），先从小值试起
    min_turn_factor: float = 0.3    # 转弯降速：油门 *= max(0.3, cos(heading))

    # ---------------- 脱困（倒车） ----------------
    stuck_fy: float = 0.15          # |Fy| 小于此值认为左右斥力抵消
    stuck_fx: float = 0.15          # Fx 小于此值认为被正面顶住
    reverse_time_s: float = 0.8     # 倒车时长
    reverse_throttle: float = -0.25 # [标定] 倒车油门
    # 很多车用电调(带刹车)需要"反向->回中->再反向"才会真正倒车；
    # 如果你的电调是直接可倒车的(如攀爬车电调)，设为 False
    esc_double_tap_reverse: bool = True
    double_tap_s: float = 0.12

    # ---------------- PWM 输出 ----------------
    steer_gpio: int = 18            # 转向舵机 BCM18
    esc_gpio: int = 13              # 电调 BCM13
    pwm_center_us: int = 1500       # [标定] 中位。舵机中位不正可微调这里（或单独设 steer_center_us）
    pwm_range_us: int = 400         # 满行程 ±400µs
    steer_center_us: int = 1500     # [标定] 舵机中位（车走不直就调这个）
    esc_center_us: int = 1500       # [标定] 电调中位
    steer_reverse: bool = False     # [标定] 舵机方向取反：heading>0 时车如果向右打，就改成 True
    esc_reverse: bool = False       # [标定] 油门方向取反：正油门时车如果后退，就改成 True
    esc_arm_s: float = 2.0          # 启动时电调中位解锁时间

    # ---------------- 其他 ----------------
    loop_hz: float = 30.0           # 主循环上限频率


# ============================================================================
# 摄像头：优先 Picamera2，失败则回退 OpenCV
# ============================================================================
class Camera:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.picam = None
        self.cap = None
        try:
            from picamera2 import Picamera2
            self.picam = Picamera2()
            # 注意：Picamera2 的 "RGB888" 实际内存顺序是 BGR，可直接给 OpenCV 用
            conf = self.picam.create_video_configuration(
                main={"size": (cfg.width, cfg.height), "format": "RGB888"})
            self.picam.configure(conf)
            self.picam.start()
            time.sleep(0.5)  # 等待自动曝光稳定
            print("[Camera] 使用 Picamera2")
        except Exception as e:  # noqa: BLE001
            print(f"[Camera] Picamera2 不可用({e})，回退 cv2.VideoCapture({cfg.camera_index})")
            self.picam = None
            self.cap = cv2.VideoCapture(cfg.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 尽量减少延迟
            if not self.cap.isOpened():
                raise RuntimeError("无法打开摄像头")

    def read(self):
        """返回 BGR 图像，尺寸统一为 cfg.width x cfg.height；失败返回 None"""
        if self.picam is not None:
            frame = self.picam.capture_array()
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            ok, frame = self.cap.read()
            if not ok:
                return None
        if frame.shape[1] != self.cfg.width or frame.shape[0] != self.cfg.height:
            frame = cv2.resize(frame, (self.cfg.width, self.cfg.height))
        return frame

    def close(self):
        try:
            if self.picam is not None:
                self.picam.stop()
                self.picam.close()
            if self.cap is not None:
                self.cap.release()
        except Exception as e:  # noqa: BLE001
            print(f"[Camera] 关闭出错: {e}")


# ============================================================================
# PWM 输出（pigpio）
# ============================================================================
class Actuator:
    def __init__(self, cfg: Config, dry_run: bool = False):
        self.cfg = cfg
        self.pi = None
        if dry_run:
            print("[PWM] dry-run 模式，不输出 PWM")
            return
        import pigpio
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("连接 pigpiod 失败，请先运行: sudo pigpiod")
        # set_servo_pulsewidth 固定输出 50Hz 舵机脉冲
        self.neutral()
        print(f"[PWM] 电调中位解锁 {cfg.esc_arm_s}s ...")
        time.sleep(cfg.esc_arm_s)

    def _us(self, center, value, reverse):
        value = float(np.clip(value, -1.0, 1.0))
        if reverse:
            value = -value
        return int(round(center + value * self.cfg.pwm_range_us))

    def set(self, steer: float, throttle: float):
        """steer: -1~1，正=左转；throttle: -1~1，正=前进"""
        if self.pi is None:
            return
        self.pi.set_servo_pulsewidth(
            self.cfg.steer_gpio, self._us(self.cfg.steer_center_us, steer, self.cfg.steer_reverse))
        self.pi.set_servo_pulsewidth(
            self.cfg.esc_gpio, self._us(self.cfg.esc_center_us, throttle, self.cfg.esc_reverse))

    def neutral(self):
        self.set(0.0, 0.0)

    def close(self):
        if self.pi is None:
            return
        try:
            self.neutral()          # 先回中
            time.sleep(0.3)         # 给电调/舵机时间响应中位
            self.pi.set_servo_pulsewidth(self.cfg.steer_gpio, 0)  # 0 = 停止输出脉冲
            self.pi.set_servo_pulsewidth(self.cfg.esc_gpio, 0)
        finally:
            self.pi.stop()


# ============================================================================
# 感知：地面分割 + 接触点 + 逆透视
# ============================================================================
class Perception:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        w, h = cfg.width, cfg.height
        # 针孔模型内参（假设像素为正方形，主点在图像中心）
        # [标定] 精确做法是用棋盘格标定得到 fx, fy, cx, cy 并替换这里
        self.fx = (w / 2.0) / math.tan(math.radians(cfg.hfov_deg) / 2.0)
        self.fy = self.fx
        self.cx = (w - 1) / 2.0
        self.cy = (h - 1) / 2.0
        self.pitch = math.radians(cfg.cam_pitch_deg)
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel))
        self.scan_cols = np.arange(cfg.scan_step // 2, w, cfg.scan_step)

        # 地面样本区域（底部中间）
        sh = max(1, int(h * cfg.sample_h_frac))
        sw = max(1, int(w * cfg.sample_w_frac))
        x0 = (w - sw) // 2
        self.sample_rect = (x0, h - sh, sw, sh)  # x, y, w, h
        self.ref_hist = None
        self.sample_polluted = False

    def ground_mask(self, bgr):
        cfg = self.cfg
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        x, y, sw, sh = self.sample_rect
        roi = hsv[y:y + sh, x:x + sw]
        # H/S 二维直方图（忽略 V，对光照明暗变化更鲁棒）
        hist = cv2.calcHist([roi], [0, 1], None,
                            [cfg.hist_bins_h, cfg.hist_bins_s], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        if self.ref_hist is None:
            self.ref_hist = hist.copy()
        corr = cv2.compareHist(hist, self.ref_hist, cv2.HISTCMP_CORREL)
        self.sample_polluted = corr < cfg.ref_hist_corr_min
        if self.sample_polluted:
            hist = self.ref_hist          # 样本区被障碍盖住，改用参考地面
        else:
            r = cfg.ref_hist_update       # 缓慢跟随光照/地面变化
            self.ref_hist = (1 - r) * self.ref_hist + r * hist
        bp = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
        _, mask = cv2.threshold(bp, cfg.backproj_thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)  # 先闭：填地面上的小洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)   # 再开：去掉孤立噪点
        return mask

    def contact_points(self, mask):
        """每隔 scan_step 列，从下往上找第一个非地面像素。返回 (u, v) 数组，整列都是地面的列不返回"""
        sub = mask[::-1, self.scan_cols]          # 上下翻转后，第 0 行 = 图像最底行
        non_ground = sub == 0
        has_obs = non_ground.any(axis=0)
        first = np.argmax(non_ground, axis=0)     # 从下往上第一个非地面像素的索引
        v = (self.cfg.height - 1) - first
        u = self.scan_cols
        return np.stack([u[has_obs], v[has_obs]], axis=1).astype(np.float64)

    def pixel_to_ground(self, uv):
        """
        像素 -> 车体坐标 (X 前, Y 左)，假设该点在地面上（Z=0）。
        相机坐标：x 右、y 下、z 前；相机绕 Y 轴向下俯仰 pitch。
        射线在车体系下：Rx = cosθ - yn*sinθ,  Ry = -xn,  Rz = -(sinθ + yn*cosθ)
        与地面 (Z = -h) 求交：t = h / (sinθ + yn*cosθ)
        """
        if len(uv) == 0:
            return np.zeros((0, 2)), np.zeros(0, dtype=bool)
        xn = (uv[:, 0] - self.cx) / self.fx
        yn = (uv[:, 1] - self.cy) / self.fy
        s, c = math.sin(self.pitch), math.cos(self.pitch)
        denom = s + yn * c
        valid = denom > 1e-6                       # 地平线及以上的点射线不会与地面相交，丢弃
        t = np.where(valid, self.cfg.cam_height_m / np.where(valid, denom, 1.0), 0.0)
        X = t * (c - yn * s)
        Y = t * (-xn)
        pts = np.stack([X, Y], axis=1)
        dist = np.hypot(X, Y)
        valid &= (X > 0) & (dist <= self.cfg.max_range_m)
        return pts, valid

    def process(self, bgr):
        mask = self.ground_mask(bgr)
        uv = self.contact_points(mask)
        pts, valid = self.pixel_to_ground(uv)
        return mask, uv[valid], pts[valid]


# ============================================================================
# 规划：人工势场 + 脱困状态机
# ============================================================================
class Planner:
    def __init__(self, cfg: Config, n_scan_cols: int):
        self.cfg = cfg
        self.rep_scale = cfg.rep_norm_cols / max(1, n_scan_cols)
        self.goal_angle = math.radians(cfg.goal_angle_deg)  # 预留：外部（遥控器）可直接改这个值
        self.heading = 0.0          # 低通后的转向角(rad)
        self.escape_until = 0.0     # 倒车结束时间
        self.escape_start = 0.0
        self.escape_steer = 0.0

    def set_goal_angle(self, deg: float):
        """预留接口：遥控器/上位机给目标方向（度，左正）"""
        self.goal_angle = math.radians(deg)

    def forces(self, pts):
        cfg = self.cfg
        fx = cfg.k_att * math.cos(self.goal_angle)
        fy = cfg.k_att * math.sin(self.goal_angle)
        if len(pts):
            d = np.hypot(pts[:, 0], pts[:, 1])
            near = (d < cfg.d0_m) & (d > 1e-3)
            if near.any():
                dn, p = d[near], pts[near]
                mag = cfg.k_rep * (1.0 / dn - 1.0 / cfg.d0_m) / dn ** 2
                # 方向：从障碍指向车 = -p/|p|
                fx += float(np.sum(-mag * p[:, 0] / dn)) * self.rep_scale
                fy += float(np.sum(-mag * p[:, 1] / dn)) * self.rep_scale
        return fx, fy

    def step(self, pts, now=None):
        """返回 dict: steer(-1~1), throttle(-1~1), heading_deg, fx, fy, min_dist, mode"""
        cfg = self.cfg
        now = time.monotonic() if now is None else now
        fx, fy = self.forces(pts)
        min_dist = float(np.min(np.hypot(pts[:, 0], pts[:, 1]))) if len(pts) else float("inf")
        max_steer = math.radians(cfg.max_steer_deg)

        # ---------- 脱困：正在倒车 ----------
        if now < self.escape_until:
            return self._escape_output(now, fx, fy, min_dist)

        # ---------- 是否需要触发倒车 ----------
        blocked = min_dist < cfg.stop_dist_m
        deadlock = abs(fy) < cfg.stuck_fy and fx < cfg.stuck_fx and min_dist < cfg.slow_dist_m
        if blocked or deadlock:
            # 舵打向 Fy 的反方向：倒车时车头会甩向 Fy 一侧（空的一侧），再前进时就朝空处走
            side = 1.0 if fy >= 0 else -1.0
            self.escape_steer = -side
            self.escape_start = now
            extra = 2 * cfg.double_tap_s if cfg.esc_double_tap_reverse else 0.0
            self.escape_until = now + cfg.reverse_time_s + extra
            self.heading = 0.0
            return self._escape_output(now, fx, fy, min_dist)

        # ---------- 正常行驶 ----------
        # Fx 取下限：合力指向后方时 heading 接近 ±90° 再被限幅，即打满舵绕向空的一侧
        raw = math.atan2(fy, max(fx, cfg.min_fx))
        raw = max(-max_steer, min(max_steer, raw))
        a = cfg.steer_lpf_alpha
        self.heading = a * self.heading + (1 - a) * raw

        if min_dist < cfg.stop_dist_m:
            thr = 0.0
        elif min_dist < cfg.slow_dist_m:
            thr = cfg.max_throttle * (min_dist - cfg.stop_dist_m) / (cfg.slow_dist_m - cfg.stop_dist_m)
        else:
            thr = cfg.max_throttle
        thr *= max(cfg.min_turn_factor, math.cos(self.heading))

        return dict(steer=self.heading / max_steer, throttle=thr,
                    heading_deg=math.degrees(self.heading),
                    fx=fx, fy=fy, min_dist=min_dist, mode="DRIVE")

    def _escape_output(self, now, fx, fy, min_dist):
        cfg = self.cfg
        t = now - self.escape_start
        thr = cfg.reverse_throttle
        if cfg.esc_double_tap_reverse:
            # 反向(刹车) -> 回中 -> 真正倒车
            if t < cfg.double_tap_s:
                thr = cfg.reverse_throttle
            elif t < 2 * cfg.double_tap_s:
                thr = 0.0
        return dict(steer=self.escape_steer, throttle=thr,
                    heading_deg=self.escape_steer * cfg.max_steer_deg,
                    fx=fx, fy=fy, min_dist=min_dist, mode="ESCAPE")


# ============================================================================
# 调试显示
# ============================================================================
def draw_debug(frame, mask, uv, perc: Perception, out, fps):
    vis = frame.copy()
    h, w = vis.shape[:2]
    x, y, sw, sh = perc.sample_rect
    cv2.rectangle(vis, (x, y), (x + sw - 1, y + sh - 1), (255, 255, 0), 1)  # 地面样本区

    for u, v in uv.astype(int):
        cv2.circle(vis, (int(u), int(v)), 2, (0, 0, 255), -1)          # 接触点

    # 合力方向箭头（图像中向上=前，向左=左）
    origin = (w // 2, h - 5)
    fx, fy = out["fx"], out["fy"]
    n = math.hypot(fx, fy) + 1e-9
    L = 60
    tip = (int(origin[0] - L * fy / n), int(origin[1] - L * fx / n))
    cv2.arrowedLine(vis, origin, tip, (0, 255, 255), 2, tipLength=0.25)
    # 实际转向方向（绿色）
    hd = math.radians(out["heading_deg"])
    tip2 = (int(origin[0] - 45 * math.sin(hd)), int(origin[1] - 45 * math.cos(hd)))
    cv2.arrowedLine(vis, origin, tip2, (0, 255, 0), 2, tipLength=0.25)

    md = out["min_dist"]
    lines = [
        f"{out['mode']}  FPS {fps:4.1f}" + ("  SAMPLE!" if perc.sample_polluted else ""),
        f"dist {md:4.2f}m" if math.isfinite(md) else "dist  inf",
        f"thr {out['throttle']:+.2f}  hdg {out['heading_deg']:+5.1f}",
        f"F ({fx:+.2f},{fy:+.2f})",
    ]
    for i, s in enumerate(lines):
        cv2.putText(vis, s, (5, 15 + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(vis, s, (5, 15 + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imshow("avoid", vis)
    cv2.imshow("ground mask", mask)
    return (cv2.waitKey(1) & 0xFF) == ord("q")


# ============================================================================
# 主程序
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="RC 车单目视觉避障")
    ap.add_argument("--debug", action="store_true", help="显示调试窗口（按 q 退出）")
    ap.add_argument("--dry-run", action="store_true", help="不输出 PWM，仅调试感知/规划")
    args = ap.parse_args()

    cfg = Config()

    # SIGTERM（如 systemd 停止）也走正常清理流程
    def _on_term(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _on_term)

    act = cam = None
    try:
        act = Actuator(cfg, dry_run=args.dry_run)
        cam = Camera(cfg)
        perc = Perception(cfg)
        plan = Planner(cfg, len(perc.scan_cols))

        period = 1.0 / cfg.loop_hz
        fps, last = 0.0, time.monotonic()
        fail_count = 0
        while True:
            t0 = time.monotonic()
            frame = cam.read()
            if frame is None:
                fail_count += 1
                act.neutral()                     # 取图失败立即停车
                if fail_count > 30:
                    raise RuntimeError("连续取图失败")
                time.sleep(0.05)
                continue
            fail_count = 0

            mask, uv, pts = perc.process(frame)
            out = plan.step(pts)
            act.set(out["steer"], out["throttle"])

            now = time.monotonic()
            dt = now - last
            last = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            if args.debug:
                if draw_debug(frame, mask, uv, perc, out, fps):
                    break
            else:
                md = out["min_dist"]
                print(f"\r{out['mode']:6s} dist={md:5.2f} thr={out['throttle']:+.2f} "
                      f"hdg={out['heading_deg']:+5.1f} fps={fps:4.1f}  ", end="", flush=True)

            sleep = period - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n收到中断，停车退出...")
    finally:
        # 顺序：先回中 -> 关 PWM -> 关摄像头
        if act is not None:
            act.close()
        if cam is not None:
            cam.close()
        cv2.destroyAllWindows()
        print("已安全退出")


if __name__ == "__main__":
    main()
