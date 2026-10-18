"""Context and ordered visual events. No clip names, IDs, or ground-truth inputs."""
import numpy as np
from .models import VLM,crop
from ..telemetry import emit

QUESTIONS = {
    "cash_in": "Is the visitor opposite the cashier transferring BANKNOTES into the cashier's possession in these frames? Exclude documents and objects other than money.",
    "cash_out": "Is the cashier transferring BANKNOTES into the visitor's possession in these frames? Exclude returning an ID card or document.",
    "machine": "Is the cashier feeding banknotes through the banknote counting machine in these frames? Hand counting alone does not count.",
    "band": "Is the cashier wrapping a rubber band around a stack of banknotes?",
    "cash_safe": "Is the cashier putting banknotes or a cash collection bag inside the safe?",
    "mixed": "Are two previously separate piles of BANKNOTES visibly combined into a single pile?",
}


def extract(an, clip, tl, budget):
    contexts, observations = [], []
    issues = []
    span = 6.0
    starts = np.arange(0, max(clip.times[-1], 0.1), span)
    for start in starts:
        if not budget.ok(12):
            issues.append("Временные события проверены не до конца: истёк бюджет анализа")
            break
        end = min(float(start + span), clip.times[-1])
        idx = sorted(set(int(np.argmin(abs(np.asarray(clip.times)-t))) for t in np.linspace(start, end, 4)))
        s, e = idx[0], idx[-1]
        images = [VLM.to_pil(clip.frames[i]) for i in idx]
        visitor = an._ask_full(clip, idx[len(idx)//2], "visitor_seated")
        at_safe = float(np.mean(tl["beside_safe"][s:e+1]))
        context = "service" if visitor >= .65 else "safe" if at_safe > .4 else "unknown"
        if context == "unknown":
            if float(np.mean(tl['n'][s:e+1])) < .2:
                context = 'empty_room'
            elif tl["n"][s] == 0 and tl["n"][e] >= 1:
                context = "arrival"
            elif tl["n"][s] >= 1 and tl["n"][e] == 0:
                context = "departure"
            elif np.mean(tl["n"][s:e+1]) >= .7 and visitor < .3:
                context = "work_without_visible_client"
        contexts.append({"start": round(float(start), 1), "end": round(end, 1), "context": context,
                         "score": round(visitor if context == "service" else at_safe if context == "safe" else .5, 3)})
        keys = ["machine", "mixed"] if np.mean(tl['attended'][s:e+1]) > .2 else []
        if visitor >= .35:
            keys += ['cash_in','cash_out','band']
        if at_safe > .1:
            keys.append("cash_safe")
        obs = {"start": round(float(start), 1), "end": round(end, 1), "context": context, "scores": {}}
        for key in keys:
            if not budget.ok(5):
                issues.append("Часть временных признаков не проверена")
                break
            focused = [crop(clip.frames[i],an.scene['desk'],margin=.55,min_side=640) for i in idx] if key!='cash_safe' and 'desk' in an.scene else images
            p = an.vlm.p_sequence(focused, QUESTIONS[key])
            obs["scores"][key] = round(p, 4)
            an.trace.append((round((float(start)+end)/2, 1), key, round(p, 4)))
        if max(obs['scores'].get('cash_in',0),obs['scores'].get('cash_out',0))>=.85 and np.mean(tl['n'][s:e+1])>=1.5:
            context='service'
            obs['context']=context
            contexts[-1].update(context=context,score=round(max(obs['scores'].get('cash_in',0),obs['scores'].get('cash_out',0)),3))
        observations.append(obs)
        emit('inference','temporal_window',start=obs['start'],end=obs['end'],context=context,scores=obs['scores'])
    return contexts, observations, issues


def findings(an, clip, observations):
    from .analyze import Finding
    from .video import nearest_index
    out, issues = [], []
    received = None
    for o in observations:
        p = o["scores"]
        if o["context"] in ("departure", "arrival"):
            received = None
        if p.get("cash_out", 0) >= .85:
            if received and received["start"] < o["start"] and o["start"]-received["end"] < 90:
                # Confirm the ordered operation using earlier and later frames together.
                times = [received["start"], received["end"], o["start"], o["end"]]
                ids = [nearest_index(clip.times, t) for t in times]
                images=[crop(clip.frames[i],an.scene['desk'],margin=.55,min_side=640) if 'desk' in an.scene else VLM.to_pil(clip.frames[i]) for i in ids]
                score = an.vlm.p_sequence(images,
                    "Do these ordered frames show the SAME visitor first giving banknotes to the cashier, then the cashier returning banknotes to that visitor? Both directions must be visible. A normal cash payout without an earlier receipt is not this event.")
                emit('inference','cash_return_verification',receipt_start=received['start'],return_start=o['start'],score=round(score,4))
                if score >= .8:
                    try:
                        geometry=an.hands.check(clip,o['start'],o['end'],an.people,an.scene) if getattr(an,'hands',None) else {'status':'unknown'}
                    except Exception as exc:
                        geometry={'status':'unknown','reason':type(exc).__name__}
                    emit('inference','handover_geometry',start=o['start'],end=o['end'],result=geometry)
                    if geometry['status']=='contact':
                        an.trace.append((geometry['time'],'hand_contact',1.))
                        receipt_index=nearest_index(clip.times,received['start']+(received['end']-received['start'])*2/3)
                        out.append(Finding("T14_CASH_RETURNED_TO_CLIENT", geometry['time'], min(score, p["cash_out"]),
                            "В последовательности кадров посетитель передаёт банкноты кассиру, затем получает их обратно; сближение рук подтверждено на соседних кадрах",
                            [(receipt_index,None),(geometry['index'],None),(ids[3],None)]))
                    elif geometry['status']=='unknown':
                        issues.append(f"{o['start']:.1f}–{o['end']:.1f} с: возможная передача денег; руки клиента и кассира недостаточно видны для подтверждения")
                    else:
                        an.trace.append(((o['start']+o['end'])/2,'hand_contact',0.))
            else:
                issues.append(f"{o['start']:.1f}–{o['end']:.1f} с: видна передача денег клиенту, но предшествующий приём денег не установлен")
        # Do not replace an earlier receipt before checking the current return.
        # Both direction scores can be high in a window containing several movements.
        if p.get('cash_in',0)>=.75 and received is None:
            received=o
        if p.get("mixed", 0) >= .75:
            ids = [nearest_index(clip.times, t) for t in np.linspace(max(0,o['start']-30),o['end'],6)]
            q = "Is the cashier visibly taking a counted bundle out of the counting machine and combining it with a separate remaining pile of banknotes that has not passed through the machine in this sequence? Sorting a bundle into separate piles does not count. Answer no if the action is unclear."
            images=[crop(clip.frames[i],an.scene['desk'],margin=.4,min_side=640) if 'desk' in an.scene else VLM.to_pil(clip.frames[i]) for i in ids]
            score = an.vlm.p_sequence(images,q)
            emit('inference','mixing_verification',start=o['start'],end=o['end'],score=round(score,4))
            if score >= .9:
                out.append(Finding('T13_CASH_MIXED',(o['start']+o['end'])/2,min(score,p['mixed']),
                    'Последовательность кадров указывает на объединение пачки из счётного аппарата с отдельно лежавшим остатком банкнот; принадлежность пачек операциям требует проверки',
                    [(ids[0],None),(ids[-2],None),(ids[-1],None)]))
            elif score>=.6 and p['mixed']>=.85:
                issues.append(f"{o['start']:.1f}–{o['end']:.1f} с: объединение пачек требует проверки принадлежности разным операциям")
    return out, issues
