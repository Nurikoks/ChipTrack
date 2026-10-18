"""Re-evaluate both runs against the same corrected labels and event intervals."""
import csv
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.metrics import evaluate
from app.pipeline.rules import TYPE_NAMES

folder=ROOT/'runtime'/'reports'
baseline=[json.loads(x) for x in (folder/'baseline'/'results.jsonl').read_text(encoding='utf-8').splitlines()]
current=[json.loads(x) for x in (folder/'current'/'results.jsonl').read_text(encoding='utf-8').splitlines()]
by_id={r['video_id']:r for r in current}
old=[]
for r in baseline:
    reference=by_id.get(r['video_id'])
    if not reference:continue
    old.append({**r,'events':reference['events'],'verdict':r['pred_verdict']})
metrics={'baseline':evaluate(old),'current':evaluate(current),'comparison_scope':'Одни и те же видео разработки, исправленная разметка, сопоставление одного предсказания одному событию по типу и интервалу с допуском 2 секунды. Это не независимый тест новой камеры.'}
(folder/'comparison.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# ChipTRick: измеренное сравнение','',metrics['comparison_scope'],'',
       '| Показатель | Исходный анализатор | Текущий |','|---|---:|---:|']
names={'videos':'Уникальных видео','tp':'Нарушающих видео с находкой (TP)','fp':'Нормальных видео с ложной находкой (FP)','fn':'Нарушающих видео без находки (FN)','tn':'Нормальных видео с завершённым чистым ответом (TN)',
       'recall':'Recall наличия нарушения','precision':'Precision наличия нарушения','event_tp':'Правильный тип в нужном интервале','event_fp':'Лишние события или неверный тип/время','event_fn':'Не найденные размеченные события','event_recall':'Recall типа + времени','event_precision':'Precision типа + времени','needs_review':'Неопределённый или неполный результат','coverage':'Доля определённых полных результатов'}
for k,name in names.items():
    values=[metrics[v][k] for v in ('baseline','current')]
    if k in ('recall','precision','event_recall','event_precision','coverage'):
        values=['—' if x is None else f'{x*100:.1f}%' for x in values]
    lines.append(f"| {name} | {values[0]} | {values[1]} |")
lines+=['','Наличие любой находки на нарушающем видео засчитывается в первый показатель даже при неверном типе. Поэтому для задачи классификации важнее строки «тип + время». Неопределённые положительные видео остаются в знаменателе recall.',
        '','## По типам','','| Тип | Размечено событий | Правильно: исходный | Правильно: текущий |','|---|---:|---:|---:|']
for tid,d in metrics['current']['per_type'].items():
    lines.append(f"| {TYPE_NAMES.get(tid,tid)} | {d['events']} | {metrics['baseline']['per_type'].get(tid,{}).get('matched',0)} | {d['matched']} |")
seconds=[r['seconds'] for r in current if 'seconds' in r]
if seconds:
    lines+=['',f"Время анализа: в среднем {sum(seconds)/len(seconds):.1f} с, максимум {max(seconds):.1f} с. Без ожидания очереди и первоначальной загрузки весов."]
lines+=['','## Каждый ролик','','| Видео | Эталон | Исходный | Текущий | Причины просмотра |','|---|---|---|---|---|']
old_by={r['video_id']:r for r in old}
def fmt(events):return '<br>'.join(f"{e.get('t',0):.1f} с — {TYPE_NAMES.get(e['type'],e['type'])}" for e in events) or 'Нет находок'
for r in current:
    b=old_by.get(r['video_id'],{})
    lines.append(f"| {r['video_id']} | {fmt(r['events']) if r['events'] else 'Нет нарушения'} | {fmt(b.get('preds',[]))} | {r['verdict']}: {fmt(r['preds'])} | {'; '.join(r.get('review_reasons',[]))} |")
(folder/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
errors=['# Конкретные ошибки текущей версии','','Ошибки посчитаны по размеченным типам и интервалам. «Требует просмотра» не означает, что нарушение найдено.','',
        '| Видео | Пропущено событий | Лишних/неверных событий | Результат |','|---|---:|---:|---|']
for r in current:
    m=evaluate([r])
    if m['event_fn'] or m['event_fp'] or m['needs_review']:
        errors.append(f"| {r['video_id']} | {m['event_fn']} | {m['event_fp']} | {r['verdict']} |")
errors+=['','## Приоритеты следующей доработки','',
         '1. Документы: отличать удостоверение от пачки банкнот по нескольким кадрам и области рук; отдельно учитывать, попало ли начало обслуживания в ролик.',
         '2. Ключ и сейф: подтверждать состояние замка и дверцы на последовательности кадров; измерять ложные находки на нормальных роликах.',
         '3. Дверь: улучшать наблюдение состояния двери при разных ракурсах и перекрытиях.',
         '4. Смешение и пересчёт: размечать наблюдаемые действия и границы операций. Отсутствие действия в коротком фрагменте не доказывает его пропуск.',
         '5. Контекст: уменьшать ненужные направления нормальных роликов на просмотр, сохраняя явную неопределённость при неполном анализе.',
         '', 'Числа по каждому типу, ожидаемые и найденные таймкоды приведены в COMPARISON.md. Эти задачи не считаются решёнными только потому, что реализованы сайт и механизм обучения.']
(folder/'ERRORS.md').write_text('\n'.join(errors)+'\n',encoding='utf-8')
print(json.dumps(metrics,ensure_ascii=False,indent=2))
