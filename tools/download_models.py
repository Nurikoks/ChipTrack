"""Download official model snapshots with pinned revision and LFS SHA256 verification."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('root',nargs='?',default=str(Path(__file__).resolve().parents[1]))
parser.add_argument('--vlm',default='Qwen/Qwen3-VL-2B-Instruct')
parser.add_argument('--vlm-dir',default='vlm')
parser.add_argument('--skip-detector',action='store_true')
args=parser.parse_args()
ROOT=Path(args.root)
REPOS={**({} if args.skip_detector else {'detector':'PekingU/rtdetr_v2_r18vd'}),args.vlm_dir:args.vlm}

def download(item):
    repo,revision,folder,f=item
    name=f['rfilename']; dst=folder/name
    dst.parent.mkdir(parents=True,exist_ok=True)
    expected=f.get('lfs',{}).get('sha256')
    if dst.exists() and (not expected or hashlib.file_digest(dst.open('rb'),'sha256').hexdigest()==expected):
        return name
    url=f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
    partial=dst.with_name(dst.name+'.partial')
    print('Downloading',repo,name,flush=True)
    with urllib.request.urlopen(url,timeout=90) as response, partial.open('wb') as output:
        h=hashlib.sha256(); size=0
        while chunk:=response.read(4*1024*1024):
            output.write(chunk); h.update(chunk); size+=len(chunk)
    if expected and h.hexdigest()!=expected: raise ValueError('SHA256 mismatch: '+name)
    partial.replace(dst)
    print('Verified',name,size,'bytes',flush=True)
    return name

for key,repo in REPOS.items():
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}?blobs=true',timeout=60) as r:
        meta=json.load(r)
    folder=ROOT/'models'/key
    folder.mkdir(parents=True,exist_ok=True)
    files=[f for f in meta['siblings'] if f['rfilename'].endswith(('.json','.safetensors','.txt','.model')) and '/' not in f['rfilename']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(download,[(repo,meta['sha'],folder,f) for f in files]))
    (folder/'chiptrick-source.json').write_text(json.dumps({'repo':repo,'revision':meta['sha'],'files':files},indent=2),encoding='utf-8')
    print('READY',key,repo,meta['sha'],flush=True)

# Pin the pose asset by content hash even though Google's URL contains "latest".
pose_url='https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task'
pose_sha='4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad'
folder=ROOT/'models'/'pose';folder.mkdir(parents=True,exist_ok=True)
dst=folder/'pose_landmarker_full.task'
if not dst.exists() or hashlib.sha256(dst.read_bytes()).hexdigest()!=pose_sha:
    with urllib.request.urlopen(pose_url,timeout=90) as response:
        data=response.read()
    if hashlib.sha256(data).hexdigest()!=pose_sha:raise ValueError('Pose asset SHA256 mismatch')
    dst.write_bytes(data)
(folder/'chiptrick-source.json').write_text(json.dumps({'url':pose_url,'sha256':pose_sha,'package':'mediapipe==1.1.1'},indent=2),encoding='utf-8')
print('READY pose',pose_sha,flush=True)
