"""Align labels and rules registry with the team's verified Excel (ForteBank_видео_и_нормы_НБРК.xlsx).

Usage: python apply_excel_types.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Excel "Описание из исходной таблицы" -> our type ids (Excel is the ground truth)
EXCEL = {
    "X-1": ("T14_CASH_ISSUED_WITHOUT_RECOUNT", "дала деньги (некую сумму) клиенту"),
    "X-2": ("T12_ID_NOT_CHECKED", "не проверяла документы клиента"),
    "X-4": ("T05_CASH_EXPOSED_NEXT_CLIENT", "не положила стопку в ящик"),
    "X-11": ("T02_PC_UNLOCKED_UNATTENDED", "не выключила ноутбук"),
    "X-13": ("T15_CASH_NOT_COUNTED", "не посчитала деньги"),
    "X-15": ("T01_UNATTENDED_VALUABLES", "не убрала пакет со стула"),
    "X-17": ("T12_ID_NOT_CHECKED", "клиент не дал своё удостоверение"),
    "X-18": ("T06_TWO_CLIENTS_AT_ONCE", "обслуживает двоих клиентов сразу"),
    "X-20": ("T03_SAFE_LEFT_OPEN", "сейф не закрыла"),
    "X-22": ("T02_PC_UNLOCKED_UNATTENDED", "не выключила ноутбук"),
    "X-24": ("T01_UNATTENDED_VALUABLES", "не положила журнал в сейф (оставила на столе)"),
    "X-26": ("T01_UNATTENDED_VALUABLES", "не положила деньги в сейф (оставила на столе)"),
    "X-28": ("T02_PC_UNLOCKED_UNATTENDED", "не выключила ноутбук"),
    "X-30": ("T07_CASH_ROOM_DOOR_NOT_CLOSED", "не закрыла за собой дверь, клиент зашёл без спроса"),
    "X-32": ("T07_CASH_ROOM_DOOR_NOT_CLOSED", "не закрыла за собой дверь, клиент зашёл без спроса"),
    "X-33": ("T07_CASH_ROOM_DOOR_NOT_CLOSED", "не закрыла за собой дверь"),
    "X-34": ("T12_ID_NOT_CHECKED", "не проверяла документы клиента"),
    "X-38": ("T05_CASH_EXPOSED_NEXT_CLIENT", "не положила деньги в сейф, второй клиент вошёл, на столе другие деньги"),
    "X-40": ("T10_PERSONAL_ITEMS_AT_DESK", "личные вещи на столе (куртка, сумка)"),
    "X-44": ("T13_CASH_MIXED", "объединила посчитанные деньги с остатком"),
    "X-46": ("T11_FOOD_DRINK_AT_DESK", "напитки на столе"),
    "X-47": ("T13_CASH_MIXED", "объединила деньги двух клиентов"),
}

lab = pd.read_csv(ROOT / "data" / "labels_enriched.csv")
for vid, (t, d) in EXCEL.items():
    m = lab["video_id"] == vid
    lab.loc[m, "type_candidate"] = t
    lab.loc[m, "confidence"] = "ground_truth_excel"
    lab.loc[m, "notes"] = "Excel (эталон): " + d + " | Phase 1 note: " + lab.loc[m, "notes"].astype(str)
lab.to_csv(ROOT / "data" / "labels_enriched.csv", index=False, encoding="utf-8")

p = ROOT / "data" / "rules_registry.json"
reg = json.loads(p.read_text(encoding="utf-8"))
reg["meta"]["version"] = "0.2"
reg["documents"]["NBRK_34"] = {
    "title": "Требования к безопасности и беспрерывности работы информационных систем банков, филиалов банков-нерезидентов Республики Казахстан и организаций, осуществляющих отдельные виды банковских операций",
    "issuer": "Правление Национального Банка Республики Казахстан",
    "act": "постановление от 28 января 2016 года № 34",
    "registration": "зарегистрировано в Министерстве юстиции РК 25 февраля 2016 года № 13256",
    "status": "действует (обновлённый), проверено на adilet 2026-10-17",
    "short_ru": "Требования к безопасности информационных систем банков (пост. Правления НБРК от 28.01.2016 № 34)",
    "url": "https://adilet.zan.kz/rus/docs/V1600013256",
    "mirror_url": "https://zakon.uchet.kz/rus/docs/V1600013256"}
ids = {r["id"] for r in reg["rules"]}
new_rules = [
    {"id": "R34_P4_1", "doc": "NBRK_34", "clause": "п. 4, пп. 1)", "clause_verified": True, "verified": True,
     "quote_short": "установку и функционирование программного или программно-аппаратного комплекса защиты от несанкционированного доступа",
     "paraphrase_ru": "На рабочих местах банк обеспечивает защиту от несанкционированного доступа с идентификацией и аутентификацией работника.",
     "governs_observable_behavior": "Включённый ноутбук на рабочем месте остаётся без присмотра (норма требует защиты доступа, а не выключения — косвенная связь)."},
    {"id": "R34_P5", "doc": "NBRK_34", "clause": "п. 5", "clause_verified": True, "verified": True,
     "quote_short": "порядок хранения и использования технических средств, паролей или другой информации, предоставляющих доступ к рабочему месту",
     "paraphrase_ru": "Порядок хранения и использования средств доступа к рабочему месту устанавливается внутренними документами банка.",
     "governs_observable_behavior": "Доступ к рабочему месту оставлен открытым (включённый компьютер без присмотра)."},
    {"id": "R231_P19_2", "doc": "NBRK_231", "clause": "п. 19, пп. 2)", "clause_verified": True, "verified": True,
     "quote_short": "принимает банкноты с полистным пересчетом, монеты с пересчетом по кружкам",
     "paraphrase_ru": "При приёме кассир пересчитывает банкноты полистно и проверяет их подлинность и платёжность.",
     "governs_observable_behavior": "Принятые деньги убраны без видимого пересчёта."},
]
reg["rules"] += [r for r in new_rules if r["id"] not in ids]
tr = reg["type_rules"]
tr.pop("T04_KEY_LEFT_IN_SAFE", None)
tr.pop("T08_COUNT_WITHOUT_CLIENT", None)
tr.pop("T06_UNAUTHORIZED_PERSON_CASH_ZONE", None)
tr["T02_PC_UNLOCKED_UNATTENDED"] = {"rules": ["R34_P4_1", "R34_P5", "ARDFM53_P71_9"], "grounding": "weak",
    "note": "Нормы требуют защиты от несанкционированного доступа, а не выключения ноутбука (косвенная связь)."}
tr["T06_TWO_CLIENTS_AT_ONCE"] = {"rules": ["R231_P20_2", "R231_P20_1"], "grounding": "weak",
    "note": "Само присутствие двух клиентов пунктом не запрещено; важно, что на рабочем месте только деньги текущего клиента."}
tr["T07_CASH_ROOM_DOOR_NOT_CLOSED"] = {"rules": ["R14_P8", "R14_P16_1"], "grounding": "medium",
    "note": "Нормы об устройстве закрывающихся входов; режим закрытия — внутренний порядок доступа."}
tr["T12_ID_NOT_CHECKED"] = {"rules": ["R231_P19_1"], "grounding": "medium",
    "note": "Условная: идентификация может быть цифровой или вне кадра."}
tr["T13_CASH_MIXED"] = {"rules": ["R231_P20_2"], "grounding": "medium",
    "note": "Деньги текущего клиента отделяются от ранее принятых (п. 20, ч. 2)."}
tr["T14_CASH_ISSUED_WITHOUT_RECOUNT"] = {"rules": ["R231_P31_4", "R231_P35"], "grounding": "strong", "note": ""}
tr["T15_CASH_NOT_COUNTED"] = {"rules": ["R231_P19_2", "R231_P20_1"], "grounding": "medium",
    "note": "Условная: пересчёт мог быть вне выбранных кадров."}
p.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
print("labels and registry updated;", len(reg["rules"]), "rules")
