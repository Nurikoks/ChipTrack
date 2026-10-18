import time
from types import SimpleNamespace
import numpy as np


def test_failed_check_cannot_be_clean(monkeypatch):
    from app.pipeline import analyze, temporal, video
    clip=SimpleNamespace(frames=[np.zeros((20,20,3),dtype=np.uint8)]*3,times=[0.,1.,2.],duration=2.,width=20,height=20)
    monkeypatch.setattr(video,'sample',lambda *a,**kw:clip)
    monkeypatch.setattr(temporal,'extract',lambda *a:([],[],[]))
    monkeypatch.setattr(temporal,'findings',lambda *a:([],[]))
    a=analyze.Analyzer.__new__(analyze.Analyzer)
    a.vlm=SimpleNamespace(calls=0)
    a.det=lambda frames:[[],[],[]]
    a._people=lambda *args:[[],[],[]]
    a._room_people=lambda *args:[[],[],[]]
    a._scene=lambda *args:{k:(0,0,10,10) for k in ['desk','safe','door','laptop']}
    a._timeline=lambda *args:{'n':np.zeros(3),'attended':np.zeros(3),'beside_safe':np.zeros(3),'t':np.array([0,1,2])}
    def failing(*args): raise RuntimeError('failed visual check')
    a._check_departures=failing
    for name in ['_check_changeover','_check_entries','_check_service','_check_periodic']:
        setattr(a,name,lambda *args:[])
    result=a.analyze('unused',SimpleNamespace(deadline=time.time()+300))
    assert result['verdict']=='needs_review'
    assert result['partial'] is True
    assert any(c['status']=='failed' for c in result['checks'])
