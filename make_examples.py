#!/usr/bin/env python3
"""Build a few COMPLETE real predictions (57 VoiceNet + genu + blend + 40 EmoNet)
for example Emolia clips, to demonstrate caption.py. Picks N clips from ONE shard
so only one tar is downloaded."""
import os, sys, json, glob, io, tarfile, random
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
import numpy as np, torch, torch.nn as nn, torchaudio, pandas as pd
from collections import OrderedDict
from transformers import AutoProcessor, WhisperForConditionalGeneration
from huggingface_hub import snapshot_download, hf_hub_download
import compute_baseline as cb

GPU = sys.argv[1] if len(sys.argv) > 1 else "7"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
DEV = "cuda:0"
HERE = os.path.dirname(os.path.abspath(__file__))
random.seed(11)

class MLPHead(nn.Module):
    def __init__(s, D, H, p, out):
        super().__init__(); s.f1=nn.Linear(D,H); s.act=nn.GELU(); s.dp=nn.Dropout(p); s.f2=nn.Linear(H,out)
    def forward(s, x): return s.f2(s.dp(s.act(s.f1(x))))

def load_reg(path):
    ck=torch.load(path,map_location="cpu",weights_only=False); a=ck["arch"]
    if isinstance(a,dict): D,H,p,out=a["D"],a["H"],a.get("p",a.get("dropout",0.0)),a.get("out",1)
    else: D=ck.get("D",768); H=ck["state_dict"]["f1.weight"].shape[0]; p=ck.get("dropout",0.0); out=1
    net=MLPHead(D,H,p,out).eval(); net.load_state_dict(ck["state_dict"])
    mu=torch.tensor(np.asarray(ck["mu"]),dtype=torch.float32); sd=torch.tensor(np.asarray(ck["sd"]),dtype=torch.float32)
    k=len(ck["levels"]) if ck.get("levels") else 7
    return net,mu,sd,k

SEQ_LEN,EMBED_DIM,PROJ=1500,768,64; HIDDEN,DROPS=[64,32,16],[0.0,0.1,0.1,0.1]
class FullEmbeddingMLP(nn.Module):
    def __init__(s,seq,emb,proj,hid,dr):
        super().__init__(); s.flatten=nn.Flatten(); s.proj=nn.Linear(seq*emb,proj)
        L=[nn.ReLU(),nn.Dropout(dr[0])]; cur=proj
        for i,h in enumerate(hid): L+=[nn.Linear(cur,h),nn.ReLU(),nn.Dropout(dr[i+1])]; cur=h
        L.append(nn.Linear(cur,1)); s.mlp=nn.Sequential(*L)
    def forward(s,x): return s.mlp(s.proj(s.flatten(x)))

# pick N clips from one EN shard that are in the embedding set
z=np.load(cb.EMB_NPZ,allow_pickle=True); emb_ids=z["ids"]; emb=z["emb"]; id2row={i:r for r,i in enumerate(emb_ids)}
idx=pd.read_parquet(cb.EMOLIA_INDEX,columns=["key","shard","__emolia_id__","language"]).rename(columns={"__emolia_id__":"eid"})
en=idx[idx.language=="en"]
# choose a shard with many embedded ids
cand_shard=None
for sh in en["shard"].drop_duplicates().sample(frac=1,random_state=3):
    sub=en[en.shard==sh]; have=[r for r in sub.itertuples() if r.eid in id2row]
    if len(have)>=N: cand_shard=sh; rows=random.sample(have,N); break
print("shard:",cand_shard,"clips:",[r.key for r in rows])

# VoiceNet heads
reg={os.path.basename(p)[:-3]:load_reg(p) for p in sorted(glob.glob(cb.VN_REG_DIR+"/*.pt"))}
genu=load_reg.__wrapped__ if False else None
gk=torch.load(cb.GENU_PT,map_location="cpu",weights_only=False); bk=torch.load(cb.BLEND_PT,map_location="cpu",weights_only=False)
def load_gb(ck):
    H=ck["state_dict"]["f1.weight"].shape[0]; net=MLPHead(768,H,0.0,1).eval(); net.load_state_dict(ck["state_dict"])
    return net,torch.tensor(np.asarray(ck["mu"]),dtype=torch.float32),torch.tensor(np.asarray(ck["sd"]),dtype=torch.float32)
gnet,gmu,gsd=load_gb(gk); bnet,bmu,bsd=load_gb(bk)

# EmoNet
wm=WhisperForConditionalGeneration.from_pretrained("laion/BUD-E-Whisper",torch_dtype=torch.float16,attn_implementation="sdpa").to(DEV).eval()
wproc=AutoProcessor.from_pretrained("laion/BUD-E-Whisper")
md=snapshot_download(cb.EMONET_REPO,ignore_patterns=["*.mp3","*.md",".gitattributes"])
avail={os.path.basename(p):p for p in glob.glob(md+"/**/*.pth",recursive=True)}
EMOTIONS=list(cb.load_emonet_taxonomy().keys())
heads={}
for e in EMOTIONS:
    fn=cb.emo_file(e); m=FullEmbeddingMLP(SEQ_LEN,EMBED_DIM,PROJ,HIDDEN,DROPS).to(DEV)
    sd=torch.load(avail[fn],map_location=DEV)
    if any(k.startswith("_orig_mod.") for k in sd): sd=OrderedDict((k.replace("_orig_mod.",""),v) for k,v in sd.items())
    m.load_state_dict(sd); heads[e]=m.eval().half()

# fetch shard, extract our members
tmpdir=os.path.join(HERE,"emonet_tmp"); os.makedirs(tmpdir,exist_ok=True)
hf_hub_download(cb.EMOLIA_AUDIO,cand_shard,repo_type="dataset",local_dir=tmpdir)
tarpath=os.path.join(tmpdir,cand_shard); want={r.key:r for r in rows}; wav={}
with tarfile.open(tarpath) as tf:
    for m in tf:
        if m.name.endswith(".flac") and m.name[:-5] in want:
            w,sr=torchaudio.load(io.BytesIO(tf.extractfile(m).read()))
            if w.dim()==2: w=w.mean(0)
            if sr!=16000: w=torchaudio.functional.resample(w,sr,16000)
            wav[m.name[:-5]]=w.float()
os.remove(tarpath)

os.makedirs(os.path.join(HERE,"examples"),exist_ok=True)
examples=[]
for r in rows:
    x=torch.tensor(emb[id2row[r.eid]],dtype=torch.float32).unsqueeze(0)
    dims={}
    with torch.no_grad():
        for dc,(net,mu,sd,k) in reg.items():
            dims[dc]=round(float(net((x-mu)/sd).squeeze().clamp(0,k-1)),3)
        gv=round(float(gnet((x-gmu)/gsd).squeeze().clamp(0,6)),3)
        bv=round(float(bnet((x-bmu)/bsd).squeeze().clamp(0,10)),3)
        w=wav[r.key]
        inp=wproc([w.numpy()],sampling_rate=16000,return_tensors="pt",padding="max_length",truncation=True)
        enc=wm.get_encoder()(inp.input_features.to(DEV).to(wm.dtype),return_dict=True).last_hidden_state.half()
        emo={e:round(float(h(enc).squeeze().float().cpu()),4) for e,h in heads.items()}
    ex={"id":r.eid,"language":r.language,"dims":dims,"genu":gv,"blend":bv,"emo":emo}
    examples.append(ex)
    json.dump(ex,open(os.path.join(HERE,"examples",f"{r.eid}.json"),"w"),indent=2,ensure_ascii=False)
    print("wrote example",r.eid)
json.dump(examples,open(os.path.join(HERE,"examples","examples.json"),"w"),indent=2,ensure_ascii=False)
print("DONE",len(examples),"examples")
