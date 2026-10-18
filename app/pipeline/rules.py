"""Rule mapper: violation type -> verified registry rules -> Russian description text.

Only rules present in data/rules_registry.json can ever be cited.
"""
import json
from functools import lru_cache

from .. import config

TYPE_NAMES = {
    "T01_UNATTENDED_VALUABLES": "Ценности/документы оставлены без присмотра на рабочем месте",
    "T02_PC_UNLOCKED_UNATTENDED": "Ноутбук оставлен включённым без присмотра",
    "T03_SAFE_LEFT_OPEN": "Сейф оставлен открытым",
    "T04_KEY_LEFT_IN_SAFE": "Ключи оставлены в замочной скважине сейфа",
    "T05_CASH_EXPOSED_NEXT_CLIENT": "Наличные предыдущей операции оставлены открыто при подходе следующего клиента",
    "T06_TWO_CLIENTS_AT_ONCE": "Обслуживание двух клиентов одновременно",
    "T07_CASH_ROOM_DOOR_NOT_CLOSED": "Дверь кассового помещения не закрыта",
    "T10_PERSONAL_ITEMS_AT_DESK": "Личные вещи на рабочем месте кассира",
    "T11_FOOD_DRINK_AT_DESK": "Напитки на рабочем месте кассира",
    "T12_ID_NOT_CHECKED": "Не проверены документы клиента",
    "T13_CASH_MIXED": "Смешение денег разных операций/клиентов",
    "T14_CASH_RETURNED_TO_CLIENT": "Возврат принятых денег клиенту",
    "T15_CASH_NOT_COUNTED": "Принятые деньги не пересчитаны",
    "VISUAL_ANOMALY": "Визуальная аномалия относительно штатной процедуры",
}

# "why it may be a violation" — phrased as a possibility, never as a conclusion
WHY = {
    "T01_UNATTENDED_VALUABLES": "это может указывать на то, что ценности оставлены вне запираемых устройств без присмотра",
    "T02_PC_UNLOCKED_UNATTENDED": "это может указывать на то, что доступ к рабочему месту оставлен незащищённым",
    "T03_SAFE_LEFT_OPEN": "это может указывать на то, что сейф рабочего места оставлен открытым",
    "T04_KEY_LEFT_IN_SAFE": "это может указывать на нарушение порядка хранения ключей от сейфа (ключи доступны посторонним)",
    "T05_CASH_EXPOSED_NEXT_CLIENT": "это может указывать на то, что ранее принятые наличные не убраны в запираемое устройство",
    "T06_TWO_CLIENTS_AT_ONCE": "это может указывать на одновременное обслуживание двух клиентов, при котором деньги разных клиентов находятся на рабочем месте",
    "T07_CASH_ROOM_DOOR_NOT_CLOSED": "это может указывать на то, что вход в кассовое помещение не закрыт",
    "T10_PERSONAL_ITEMS_AT_DESK": "это может указывать на хранение личных вещей в кассовом помещении",
    "T11_FOOD_DRINK_AT_DESK": "это может указывать на внос в кассовое помещение предметов вне разрешённого перечня",
    "T12_ID_NOT_CHECKED": "это может указывать на то, что визуальная идентификация клиента по документу не проводилась (если она не была цифровой)",
    "T13_CASH_MIXED": "это может указывать на то, что деньги текущей операции не отделены от ранее принятых",
    "T14_CASH_RETURNED_TO_CLIENT": "это соответствует указанному командой отклонению от процедуры приёма наличных: деньги переданы обратно клиенту; точное внутреннее основание требует документа банка",
    "T15_CASH_NOT_COUNTED": "это может указывать на то, что принятые наличные не пересчитаны полистно",
    "VISUAL_ANOMALY": "это отличается от штатной процедуры и требует проверки человеком",
}

GROUNDING_PREFIX = {"strong": "Правило", "medium": "Возможное нарушение требований",
                    "weak": "Косвенно связано с требованием", "none": ""}


@lru_cache(maxsize=1)
def registry() -> dict:
    return json.loads(config.RULES_PATH.read_text(encoding="utf-8"))


def rule_refs(type_id: str) -> tuple[list[dict], str]:
    reg = registry()
    entry = reg["type_rules"].get(type_id) or reg.get("extension_types", {}).get(type_id) \
        or {"rules": [], "grounding": "none"}
    by_id = {r["id"]: r for r in reg["rules"]}
    refs = []
    for rid in entry["rules"][:2]:  # cite at most the two most relevant clauses
        r = by_id[rid]
        d = reg["documents"][r["doc"]]
        refs.append({"id": rid, "document": d["short_ru"], "document_full": d["title"],
                     "act": d["act"], "clause": r["clause"], "url": d["url"],
                     "verified": bool(r["verified"] and r["clause_verified"]),
                     "summary": r["paraphrase_ru"]})
    return refs, entry["grounding"]


def rule_text(type_id: str) -> str:
    refs, grounding = rule_refs(type_id)
    if not refs:
        return "Нормативное основание не верифицировано; требуется оценка проверяющим."
    head = refs[0]
    txt = (f"{GROUNDING_PREFIX[grounding]}: {head['document']}, {head['clause']} — "
           f"{head['summary']} Источник: {head['url']}")
    if len(refs) > 1:
        txt += f"; см. также {refs[1]['clause']}."
    if grounding == "weak":
        txt += " (основание косвенное/по толкованию)"
    return txt


def build_violation(type_id: str, timecode: float, observed: str, frames: list[dict],
                    confidence: float) -> dict:
    """observed: factual Russian sentence(s) about what is visible (no intent, no identity)."""
    refs, grounding = rule_refs(type_id)
    desc = f"{observed.rstrip('.')}; {WHY[type_id]}. {rule_text(type_id)}"
    return {
        "type": TYPE_NAMES[type_id],
        "timecode_sec": round(float(timecode), 1),
        "description": desc,
        "evidence_frames": frames[:3],
        # --- additional keys (never replace the required ones) ---
        "type_id": type_id,
        "confidence": round(float(confidence), 3),
        "rule": {"grounding": grounding, "references": refs},
    }
