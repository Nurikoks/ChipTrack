"""Development-only probe for handover evidence, not an inference shortcut by video ID."""
import csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import config
from app.pipeline import video
from app.pipeline.models import VLM,crop
labels={r['video_id']:r for r in csv.DictReader((config.ROOT/'data/labels_enriched.csv').open(encoding='utf-8-sig'))}
baseline={r['video_id']:r for r in map(json.loads,(config.WORK_DIR/'reports/baseline/results.jsonl').read_text(encoding='utf-8').splitlines())}
vlm=VLM(config.VLM_ID)
questions={
 'receive':'Does the visitor reach out and TAKE an item from the cashier, so that the item moves into the visitor\'s hand? The visitor merely watching the cashier handle a bundle is not a handover.',
 'cash_receive':'Does the visitor visibly receive a bundle of BANKNOTES from the cashier? The visitor must actually take possession of the banknotes, not just watch the cashier.',
 'visitor_hands':'Do the visitor\'s hands move across the desk toward the cashier to receive something in this sequence?',
}
rows=[]
for vid,start,end in [('X-1',18,24),('X-4',30,36),('X-3',12,18),('X-5',30,36)]:
    clip=video.sample(config.ROOT/'videos'/labels[vid]['file'])
    ids=[video.nearest_index(clip.times,t) for t in np.linspace(start,min(end,clip.times[-1]),5)]
    images=[crop(clip.frames[i],baseline[vid]['debug']['scene']['desk'],margin=.55,min_side=640) for i in ids]
    for key,q in questions.items():
        score=vlm.p_sequence(images,q)
        row={'video':vid,'start':start,'end':end,'question':key,'score':round(score,5)}
        rows.append(row);print(json.dumps(row),flush=True)
        (config.WORK_DIR/'reports/probe-handover.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
