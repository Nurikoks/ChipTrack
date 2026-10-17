"""Rule mapper: violation type -> verified registry rules -> Russian description text.

Only rules present in data/rules_registry.json can ever be cited.
"""
import json
from functools import lru_cache

from .. import config

TYPE_NAMES = {
    "T01_UNATTENDED_VALUABLES": "Ценности/документы оставлены без присмотра на рабочем месте",
    "T02_PC_UNLOCKED_UNATTENDED": "Рабочее место покинуто без блокировки компьютера",
    "T03_SAFE_LEFT_OPEN": "Сейф оставлен открытым без присмотра",
    "T04_KEY_LEFT_IN_SAFE": "Ключ оставлен в замке сейфа",
    "T05_CASH_EXPOSED_NEXT_CLIENT": "Наличные предыдущей операции оставлены открыто при подходе следующего клиента",
    "T06_UNAUTHORIZED_PERSON_CASH_ZONE": "Постороннее лицо в кассовой зоне",
    "T07_CASH_ROOM_DOOR_NOT_CLOSED": "Дверь кассового помещения не закрыта",
    "T08_COUNT_WITHOUT_CLIENT": "Пересчёт наличных клиента в его отсутствие",
    "T09_COLLECTOR_NO_DOC_CHECK": "Приём сумки у инкассатора без видимой проверки документов",
    "T10_PERSONAL_ITEMS_AT_DESK": "Личные вещи на рабочем месте кассира",
    "T11_FOOD_DRINK_AT_DESK": "Употребление напитков/пищи на рабочем месте",
    "VISUAL_ANOMALY": "Визуальная аномалия относительно штатной процедуры",
}

# "why it may be a violation" — phrased as a possibility, never as a conclusion
WHY = {
    "T01_UNATTENDED_VALUABLES": "это может указывать на то, что ценности оставлены вне запираемых устройств без присмотра",
    "T02_PC_UNLOCKED_UNATTENDED": "это может указывать на то, что рабочая станция оставлена без блокировки",
    "T03_SAFE_LEFT_OPEN": "это может указывать на то, что сейф рабочего места оставлен открытым без присмотра",
    "T04_KEY_LEFT_IN_SAFE": "это может указывать на нарушение порядка хранения ключей от сейфа",
    "T05_CASH_EXPOSED_NEXT_CLIENT": "это может указывать на то, что ранее принятые наличные не убраны в запираемое устройство",
    "T06_UNAUTHORIZED_PERSON_CASH_ZONE": "это может указывать на доступ постороннего лица в кассовую зону",
    "T07_CASH_ROOM_DOOR_NOT_CLOSED": "это может указывать на то, что вход в кассовое помещение не закрыт",
    "T08_COUNT_WITHOUT_CLIENT": "это может указывать на пересчёт наличных клиента не в его присутствии",
    "T09_COLLECTOR_NO_DOC_CHECK": "это может указывать на приём сумки без проверки препроводительного документа",
    "T10_PERSONAL_ITEMS_AT_DESK": "это может указывать на хранение личных вещей в кассовом помещении",
    "T11_FOOD_DRINK_AT_DESK": "это может указывать на внос в кассовое помещение предметов вне разрешённого перечня",
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
