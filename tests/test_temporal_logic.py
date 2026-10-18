from types import SimpleNamespace
import numpy as np


def test_return_uses_earlier_receipt_even_when_current_window_has_both_scores():
    from app.pipeline.temporal import findings
    clip=SimpleNamespace(times=list(range(25)),frames=[np.zeros((8,8,3),dtype=np.uint8)]*25)
    an=SimpleNamespace(scene={},vlm=SimpleNamespace(p_sequence=lambda *args:.96),trace=[],people=[],
                       hands=SimpleNamespace(check=lambda *args:{'status':'contact','time':21,'index':21}))
    observations=[{'start':6.,'end':12.,'context':'service','scores':{'cash_in':.91,'cash_out':.2}},
                  {'start':18.,'end':24.,'context':'service','scores':{'cash_in':.92,'cash_out':.95}}]
    found,issues=findings(an,clip,observations)
    assert len(found)==1 and found[0].type_id=='T14_CASH_RETURNED_TO_CLIENT'
    assert found[0].t==21 and issues==[]


def test_payout_without_prior_receipt_is_not_automatically_return_violation():
    from app.pipeline.temporal import findings
    clip=SimpleNamespace(times=list(range(7)),frames=[np.zeros((8,8,3),dtype=np.uint8)]*7)
    an=SimpleNamespace(scene={},vlm=SimpleNamespace(p_sequence=lambda *args:.96))
    found,issues=findings(an,clip,[{'start':0.,'end':6.,'context':'service','scores':{'cash_out':.98}}])
    assert found==[] and len(issues)==1
