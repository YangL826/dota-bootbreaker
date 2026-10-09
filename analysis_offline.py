# 离线回放：跟踪+落点预测 vs 真实落点
import sys,os,cv2,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import importlib, bb_bot as B
fs=sorted(f for f in os.listdir("frames") if f.endswith(".jpg"))
tr=B.BallTracker(); a=[]
for f in fs:
    t=float(f[:-4]); c,p=B.detect(cv2.imread("frames/"+f)); b=tr.update(t,c)
    L=None
    if b:
        x,y=tr.pos(); vx,vy=tr.vel()
        if tr.n()>=4: L=B.landing_x(x,y,vx,vy)
    a.append((t,b[0] if b else np.nan,b[1] if b else np.nan,L[0] if L else np.nan))
a=np.array(a); ys=a[:,2]
hits=[k for k in range(2,len(a)-2) if ys[k]>740 and ys[k]==np.nanmax(ys[k-2:k+3])]
E={0.6:[],0.4:[],0.25:[],0.1:[]}
for h in hits:
    for T in E:
        ks=[k for k in range(max(0,h-60),h) if not np.isnan(a[k,3]) and a[h,0]-a[k,0]>=T]
        if ks: E[T].append(abs(a[ks[-1],3]-a[h,1]))
for T,v in E.items():
    v=np.array(v); print(f"提前{T}s 落点误差: 中位 {np.median(v):.0f}px  <50px比例 {np.mean(v<50):.0%}  (n={len(v)})")
