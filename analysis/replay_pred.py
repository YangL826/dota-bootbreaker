import sys,math,numpy as np,pandas as pd
sys.path.insert(0,".")
import bb_bot as B
d=pd.read_csv(sys.argv[1]); d=d[d.bx.notna()][["t","bx","by"]].values
def spatial(tr):
    a=np.array(tr.h)
    if len(a)<4 or a[-1,2]-a[0,2]<25: return None   # 需要往下走了一段
    b=np.polyfit(a[:,2],a[:,1],1)                     # x = b0*y + b1
    x=np.polyval(b,a[-1,2]); return B.fold(x+b[0]*(B.HIT_Y-a[-1,2]))
tr=B.BallTracker(); rows=[]
for t,x,y in d:
    p=tr.update(t,[(x,y,100)])
    tp=sp=None
    if p and tr.n()>=4:
        bx,by=tr.pos(); vx,vy=tr.vel()
        L=B.landing_x(bx,by,vx,vy); tp=L[0] if L else None
        if vy>30: sp=spatial(tr)
    rows.append((t,x,y,tp,sp))
r=pd.DataFrame(rows,columns=["t","x","y","tp","sp"])
ys=r.y.values; hits=[k for k in range(3,len(r)-3) if ys[k]>745 and ys[k]==ys[k-3:k+4].max()]
for T in (0.4,0.3,0.2,0.12):
    et=[];es=[]
    for h in hits:
        w=r[(r.t<=r.t[h]-T)&(r.t>r.t[h]-T-0.06)]
        # 只取同一段下落（中间没有反弹）
        if len(w)==0: continue
        k=w.index[-1]
        if not (r.y[k:h+1].diff().dropna()>=-3).all(): continue
        if pd.notna(r.tp[k]): et.append(abs(r.tp[k]-r.x[h]))
        if pd.notna(r.sp[k]): es.append(abs(r.sp[k]-r.x[h]))
    f=lambda v:f"中位{np.median(v):.0f} <40px {np.mean(np.array(v)<40):.0%} n={len(v)}"
    print(f"提前{T}s  速度法: {f(et)}   轨迹法: {f(es)}")
