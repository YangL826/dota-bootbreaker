import pandas as pd,numpy as np,sys
d=pd.read_csv(sys.argv[1]).rename(columns={"pad":"pd"})
print("rows",len(d),"dur",d.t.iloc[-1],"loop dt q",d.t.diff().quantile([.5,.9,.99]).round(4).tolist())
nop=d["pd"].isna().values; trans=[];start=None
for t,n in zip(d.t.values,nop):
    if n and start is None: start=t
    if not n and start is not None:
        if t-start>1: trans.append((start,t))
        start=None
print("levels",len(trans)+1,[round(a) for a,b in trans])
seen=d.dropna(subset=["bx"])
miss=[]
for th in d[d.act=="lock"].t.values:
    prev=seen[seen.t<th]
    if not len(prev): continue
    last=prev.iloc[-1]
    if last.by<700: continue   # 关卡结束/误检，不是掉球
    miss.append(last.t)
print("真实掉球",len(miss))
for m in miss:
    w=d[(d.t>m-1.2)&(d.t<=m+0.05)]
    sp=np.hypot(w.vx,w.vy).dropna()
    last=w.dropna(subset=["bx"]).iloc[-1]
    lost=w.bx.isna().sum()
    tg=w.dropna(subset=["target"])
    print(f"t={m:.1f} 球({last.bx:.0f},{last.by:.0f}) 挡板{last['pd']:.0f} 差{last.bx-last['pd']:+.0f} 球速中位{sp.median():.0f} 丢帧{lost} 最后目标{tg.target.iloc[-1] if len(tg) else None} 模式序列 {''.join(str(x)[0] for x in w['mode'].fillna('-').values[-25:])}")
