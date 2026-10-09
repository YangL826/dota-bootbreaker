# -*- coding: utf-8 -*-
"""
破牢之靴 Boot Breaker bot（Claude 雏形 v0.1）
  用 dota_bootbreaker_rec/config.json 里录像时框的区域（F8/F9 可重框，但必须框法一致：
  左上角=游戏窗口左上角，右下角=窗口右下角，跟录像那次一样）
  F6  开始 / 暂停
  F7  存一张当前截图到 debug/（认错时用）
  F10 退出
"""
import os, sys, time, json, math, threading, queue, csv
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
CFGF = os.path.join(HERE, "config.json")

# ---- 场地几何（相对框选区域左上角，936x927 的框量出来的）----
FX0, FX1, FY0, FY1 = 166, 770, 95, 792      # 找球的范围
WALL_L, WALL_R = 175, 757                   # 球心能到的左右极限（碰墙反弹）
HIT_Y = 770                                 # 球心碰到挡板顶的高度
PAD_Y0, PAD_Y1 = 798, 816                   # 挡板红色车顶所在行
PAD_MIN, PAD_MAX = 228, 710                 # 挡板中心活动范围
PAD_W = 120
REPLAY_BTN = (464, 734)                     # “再玩一次”按钮

def detect(img):
    """返回 ball=(x,y,score) 列表, paddle=(cx,width) 或 None"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    f = hsv[FY0:FY1, FX0:FX1]
    H, S, V = f[..., 0], f[..., 1], f[..., 2]
    arc = ((H >= 95) & (H <= 130) & (S >= 120) & (V >= 90)).astype(np.uint8)
    boot = ((H >= 4) & (H <= 20) & (S >= 120) & (V >= 40) & (V <= 150)).astype(np.uint8)
    near = cv2.dilate(arc, np.ones((15, 15), np.uint8))
    comb = cv2.dilate((boot & near) | arc, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(comb)
    cands = []
    if n > 1:
        acs = np.bincount(lab.ravel(), weights=arc.ravel(), minlength=n)
        bcs = np.bincount(lab.ravel(), weights=boot.ravel(), minlength=n)
    for i in range(1, n):
        w, h = st[i, 2], st[i, 3]
        if w > 90 or h > 90:
            continue
        ac = int(acs[i]); bc = int(bcs[i])
        if ac >= 22 and bc >= 15:
            cands.append((cen[i][0] + FX0, cen[i][1] + FY0, ac + bc))
    r = hsv[PAD_Y0:PAD_Y1, FX0:FX1]
    red = (((r[..., 0] >= 160) | (r[..., 0] <= 10)) & (r[..., 1] >= 150) & (r[..., 2] >= 55))
    cols = np.where(red.sum(0) >= 8)[0]
    pad = None
    if len(cols) > 20:
        runs = np.split(cols, np.where(np.diff(cols) > 20)[0] + 1)
        rr = max(runs, key=len)
        if 60 <= rr[-1] - rr[0] <= 160:
            pad = ((rr[0] + rr[-1]) / 2 + FX0, rr[-1] - rr[0])
    return cands, pad

class BallTracker:
    """自上次反弹以来的位置历史 + 线性拟合。
    反弹判定：新点偏离预测 >45px，或最近 3 点的走向与拟合速度夹角 >70°（且速度够大）。"""
    MAXH = 40          # 循环约 57 帧/秒；12 点只够 0.2 秒，拟合噪声大（实测 40 点落点误差 46→20px）
    def __init__(self):
        self.h = []; self.last_seen = 0
        self.black = []          # 静止误检的位置（第五关左上角那种），换关时清空
    def update(self, t, cands):
        if self.black:
            tracking = self.h and t - self.h[-1][0] < 0.3
            pp = self.predict_pos(t) if tracking else None
            def ok(c):
                # 正在跟踪的球经过拉黑位置时不屏蔽（只屏蔽远离预测位置的静止误检）
                if pp and math.hypot(c[0] - pp[0], c[1] - pp[1]) < 40:
                    return True
                return all(math.hypot(c[0] - bx, c[1] - by) > 35 for bx, by in self.black)
            cands = [c for c in cands if ok(c)]
        pick = None
        if cands:
            if self.h and t - self.h[-1][0] < 0.3:
                px, py = self.predict_pos(t)
                d = [math.hypot(c[0] - px, c[1] - py) for c in cands]
                k = int(np.argmin(d))
                if d[k] < 110:
                    pick = cands[k]
            if pick is None and (not self.h or t - self.h[-1][0] >= 0.3):
                pick = max(cands, key=lambda c: c[2])
        if pick is None:
            if self.h and t - self.h[-1][0] > 0.3:
                self.h = []
            return None
        if len(self.h) >= 4:
            px, py = self.predict_pos(t)
            vx, vy = self.vel()
            bounce = math.hypot(pick[0] - px, pick[1] - py) > 45
            if not bounce and len(self.h) >= 5:
                (ta, xa, ya) = self.h[-2]
                cx, cy = pick[0] - xa, pick[1] - ya
                if math.hypot(cx, cy) > 25 and math.hypot(vx, vy) > 150:
                    cos = (cx * vx + cy * vy) / (math.hypot(cx, cy) * math.hypot(vx, vy))
                    bounce = cos < 0.34
            if bounce:
                self.h = self.h[-1:]
        self.h.append((t, pick[0], pick[1])); self.h = self.h[-self.MAXH:]
        self.last_seen = t
        return pick
    def n(self):
        return len(self.h)
    def _fit(self):
        a = np.array(self.h)
        tt = a[:, 0] - a[-1, 0]
        px = np.polyfit(tt, a[:, 1], 1); py = np.polyfit(tt, a[:, 2], 1)
        return px, py
    def vel(self):
        if len(self.h) < 2:
            return 0.0, 0.0
        if len(self.h) == 2:
            (t0, x0, y0), (t1, x1, y1) = self.h
            dt = max(t1 - t0, 1e-3); return (x1 - x0) / dt, (y1 - y0) / dt
        px, py = self._fit(); return px[0], py[0]
    def pos(self):
        if len(self.h) < 3:
            return self.h[-1][1], self.h[-1][2]
        px, py = self._fit(); return px[1], py[1]
    def predict_pos(self, t):
        x, y = self.pos(); vx, vy = self.vel(); dt = t - self.h[-1][0]
        return x + vx * dt, y + vy * dt

def fold(x):
    w = WALL_R - WALL_L
    r = (x - WALL_L) % (2 * w)
    return WALL_L + (2 * w - r if r > w else r)

def landing_x(x, y, vx, vy):
    if vy <= 30:
        return None
    t = (HIT_Y - y) / vy
    if t < 0 or t > 4:
        return None
    return fold(x + vx * t), t

# ---- 键盘：扫描码 SendInput（keyboard 库发的方向键 Dota 收不到）----
SCAN = {"left": (0x4B, True), "right": (0x4D, True), "space": (0x39, False)}
def _send_scan(key, up):
    import ctypes, ctypes.wintypes as wt
    PUL = ctypes.POINTER(ctypes.c_ulong)
    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD), ("time", wt.DWORD), ("dwExtraInfo", PUL)]
    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD), ("dwFlags", wt.DWORD), ("time", wt.DWORD), ("dwExtraInfo", PUL)]
    class _I(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]
    class INPUT(ctypes.Structure):
        _fields_ = [("type", wt.DWORD), ("ii", _I)]
    sc, ext = SCAN[key]
    flags = 0x0008 | (0x0002 if up else 0) | (0x0001 if ext else 0)
    inp = INPUT(type=1, ii=_I(ki=KEYBDINPUT(0, sc, flags, 0, None)))
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
def key_down(k): _send_scan(k, False)
def key_up(k): _send_scan(k, True)
def key_tap(k): key_down(k); time.sleep(0.04); key_up(k)

# ---------------- 以下为实机部分 ----------------
def main():
    import ctypes, mss, keyboard
    try: ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception: pass
    u32 = ctypes.windll.user32
    cfg = json.load(open(CFGF))
    st = {"run": False, "quit": False, "snap": False}
    def cur():
        import ctypes.wintypes as w
        p = w.POINT(); u32.GetCursorPos(ctypes.byref(p)); return p.x, p.y
    def tl(): cfg["left"], cfg["top"] = cur(); print("左上角", cfg["left"], cfg["top"])
    def br():
        cfg["right"], cfg["bottom"] = cur(); json.dump(cfg, open(CFGF, "w"), indent=2)
        print("右下角", cfg["right"], cfg["bottom"], "已保存；尺寸", cfg["right"] - cfg["left"], cfg["bottom"] - cfg["top"], "(应约 936x927)")
    keyboard.add_hotkey("f8", tl); keyboard.add_hotkey("f9", br)
    keyboard.add_hotkey("f6", lambda: (st.update(run=not st["run"]), print("[>] 运行" if st["run"] else "[||] 暂停")))
    keyboard.add_hotkey("f7", lambda: st.update(snap=True))
    keyboard.add_hotkey("f10", lambda: st.update(quit=True))

    def fg_ok():
        h = u32.GetForegroundWindow(); b = ctypes.create_unicode_buffer(256)
        u32.GetWindowTextW(h, b, 256); return "dota" in b.value.lower()

    held = {"left": False, "right": False}
    def hold(k):
        o = "right" if k == "left" else "left"
        if held[o]: key_up(o); held[o] = False
        if k and not held[k]: key_down(k); held[k] = True
    def release_all():
        for k in held:
            if held[k]: key_up(k); held[k] = False

    os.makedirs(os.path.join(HERE, "debug"), exist_ok=True)
    logq = queue.Queue()
    def logger():
        os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
        f = open(os.path.join(HERE, "runs", time.strftime("run_%m%d_%H%M%S.csv")), "w", newline="")
        w = csv.writer(f); w.writerow(["t", "bx", "by", "vx", "vy", "pad", "target", "act", "mode"])
        while True:
            r = logq.get()
            if r is None: break
            w.writerow(r)
            if logq.empty(): f.flush()
        f.close()
    threading.Thread(target=logger, daemon=True).start()

    print(__doc__); print("框选：", cfg)
    tr = BallTracker()
    pad_hist = []
    last_serve = 0; serve_step = 0; no_ball_since = None; no_pad_since = None
    t0 = time.time(); fg_warned = False
    last_replay = 0; games = 0
    BTN = cv2.imread(os.path.join(HERE, "replay_btn.png"))
    GX0, GY0, GX1, GY1 = 160, 90, 780, 830          # 只截场地（含挡板和“再玩一次”按钮）
    canvas = np.zeros((cfg["bottom"] - cfg["top"], cfg["right"] - cfg["left"], 3), np.uint8)
    nloop = 0; tfps = time.time(); static_snapped = False
    level_snap_at = t0 + 2; still_since = None; last_target = None; last_throw = None
    def snap(im, tag):
        try:
            cv2.imwrite(os.path.join(HERE, "debug", f"{tag}_{time.time() - t0:07.1f}.jpg"), im)
        except Exception:
            pass
    with (mss.MSS() if hasattr(mss, "MSS") else mss.mss()) as sct:
        while not st["quit"]:
            if not st["run"]:
                release_all(); time.sleep(0.05); continue
            if not fg_ok():
                release_all()
                if not fg_warned: print("[!] Dota 不在前台，暂停按键"); fg_warned = True
                time.sleep(0.1); continue
            fg_warned = False
            t = time.time()
            mon = {"left": cfg["left"] + GX0, "top": cfg["top"] + GY0, "width": GX1 - GX0, "height": GY1 - GY0}
            canvas[GY0:GY1, GX0:GX1] = np.asarray(sct.grab(mon))[:, :, :3]
            img = canvas
            if st["snap"]:
                cv2.imwrite(os.path.join(HERE, "debug", f"{t:.2f}.png"), img); st["snap"] = False; print("截图已存")
            cands, pad = detect(img)
            ball = tr.update(t, cands)
            # 看门狗：球在半空静止不动 >0.6s = 误检，拉黑这个位置并截图
            still = False
            if ball and tr.n() >= 6 and ball[1] < 735:
                vx_, vy_ = tr.vel(); still = math.hypot(vx_, vy_) < 30
            if not still:
                still_since = None
            else:
                if still_since is None: still_since = t
                if t - still_since > 0.6:
                    still_since = None
                    tr.black.append((ball[0], ball[1])); tr.h = []; ball = None
                    print(f"[!] ({tr.black[-1][0]:.0f},{tr.black[-1][1]:.0f}) 静止误检，已忽略")
                    if not static_snapped: snap(img, "static"); static_snapped = True
            # 换关/重开后存一张关卡全貌，方便以后分析
            if pad and no_pad_since is not None and t - no_pad_since > 1.0:
                tr.black = []; level_snap_at = t + 1.5; static_snapped = False
            if level_snap_at and t > level_snap_at:
                snap(img, "level"); level_snap_at = None
            act = "-"; target = None; mode = ""
            if pad:
                no_pad_since = None
                pad_hist.append((t, pad[0])); pad_hist = pad_hist[-4:]
                pv = 0.0
                if len(pad_hist) >= 2:
                    pv = (pad_hist[-1][1] - pad_hist[0][1]) / max(pad_hist[-1][0] - pad_hist[0][0], 1e-3)
                on_cart = False
                if ball and tr.n() >= 6:
                    vx_, vy_ = tr.vel()
                    on_cart = ball[1] > 735 and math.hypot(vx_, vy_) < 60
                coasting = (not ball) and last_target is not None and t - tr.last_seen < 0.4
                if coasting:
                    # 球短暂丢失（被特效/文字挡住）：继续朝最后的目标走，不要停
                    err = last_target - pad[0]
                    if abs(err) > 15: hold("right" if err > 0 else "left"); act = "coast"
                    else: hold(None); act = "stop"
                    target = last_target; mode = "coast"
                elif ball and not on_cart:
                    no_ball_since = None; serve_step = 0
                    bx, by = tr.pos(); vx, vy = tr.vel()
                    lx = landing_x(bx, by, vx, vy) if tr.n() >= 4 else None
                    if lx:
                        target, mode = lx[0], "catch"
                    else:
                        target, mode = bx, "follow"   # 球往上飞：先跟着球的 x
                    target = min(max(target, PAD_MIN), PAD_MAX)
                    last_target = target
                    err = target - pad[0]
                    brake = 15 + 0.08 * abs(pv)       # 松键后还会滑 20-30px
                    moving_toward = (pv > 50 and err > 0) or (pv < -50 and err < 0)
                    if abs(err) > (brake if moving_toward else 10):
                        hold("right" if err > 0 else "left"); act = "R" if err > 0 else "L"
                    else:
                        hold(None); act = "stop"
                else:
                    if no_ball_since is None: no_ball_since = t
                    if serve_step == 0 and 0 < t - last_serve < 3 and last_throw and t - last_throw < 3:
                        # 刚扔出但还没认出球：挡板先回中间，离哪边都近
                        err = 469 - pad[0]
                        if abs(err) > 20: hold("right" if err > 0 else "left"); act = "center"
                        else: hold(None)
                    else:
                        release_all()
                    # 球不在场上 = 等发球：空格锁车位 → 等瞄准线 → 空格扔出
                    if t - no_ball_since > 0.8:
                        if serve_step == 0 and t - last_serve > 2.5:
                            key_tap("space"); serve_step = 1; last_serve = t; act = "lock"
                        elif serve_step == 1 and t - last_serve > 0.7:
                            key_tap("space"); serve_step = 0; last_serve = t; act = "throw"; last_throw = t
                            no_ball_since = t
            else:
                release_all()
                if no_pad_since is None: no_pad_since = t
                # 看不到挡板：可能是换关黑屏，也可能是游戏结束。只有认出“再玩一次”按钮才点
                if t - no_pad_since > 1.5 and t - last_replay > 3:
                    g = img[650:820, 330:600]
                    r = cv2.matchTemplate(g, BTN, cv2.TM_CCOEFF_NORMED)
                    _, mx, _, loc = cv2.minMaxLoc(r)
                    if mx > 0.8:
                        x = cfg["left"] + 330 + loc[0] + BTN.shape[1] // 2
                        y = cfg["top"] + 650 + loc[1] + BTN.shape[0] // 2
                        u32.SetCursorPos(x, y); time.sleep(0.05)
                        u32.mouse_event(2, 0, 0, 0, 0); time.sleep(0.05); u32.mouse_event(4, 0, 0, 0, 0)
                        games += 1
                        print(f"游戏结束（第 {games} 局），点了再玩一次"); act = "replay"; last_replay = t
                        tr.h = []; serve_step = 0
            logq.put([f"{t - t0:.3f}",
                      f"{ball[0]:.0f}" if ball else "", f"{ball[1]:.0f}" if ball else "",
                      f"{tr.vel()[0]:.0f}" if ball else "", f"{tr.vel()[1]:.0f}" if ball else "",
                      f"{pad[0]:.0f}" if pad else "", f"{target:.0f}" if target is not None else "", act, mode])
            nloop += 1
            if t - tfps > 30:
                print(f"[i] 循环 {nloop / (t - tfps):.0f} 帧/秒"); nloop = 0; tfps = t
            dt = time.time() - t
            if dt < 0.008: time.sleep(0.008 - dt)
    release_all(); logq.put(None); time.sleep(0.2); print("退出")

if __name__ == "__main__":
    main()
