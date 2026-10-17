# Normative sources and the source → rule → video mapping (Phase 2)

Machine-readable version: [data/rules_registry.json](../data/rules_registry.json). The service may cite **only** rules from that file.

## 1. Sources found and verified

Every clause below was string-matched against the official text on **adilet.zan.kz** on 2026-10-17. The full text was first extracted from the zakon.uchet.kz mirror, then each quote was confirmed on the rendered adilet page.

| Key | Document | Act | Status | URL |
|---|---|---|---|---|
| NBRK_231 | Rules on cash operations and on collection of banknotes, coins and valuables in second-tier banks… (Правила осуществления кассовых операций и операций по инкассации банкнот, монет и ценностей в банках второго уровня…) | NBRK Board Resolution of 29.11.2019 No. 231, reg. MoJ No. 19680 | In force (edition with amendments to 12.07.2026) | https://adilet.zan.kz/rus/docs/V1900019680 |
| NBRK_14 | Rules on organizing security and arrangement of premises of second-tier banks… (Правила организации охраны и устройства помещений банков второго уровня…) | NBRK Board Resolution of 24.02.2020 No. 14, reg. MoJ No. 20075 | In force (paras. 11, 13.3, 14, 16 apply in a special wording while suspended until 01.01.2027) | https://adilet.zan.kz/rus/docs/V2000020075 |
| ARDFM_53 | Information-security requirements for banks… (Требования к обеспечению информационной безопасности банков…) | ARDFM Board Resolution of 03.04.2026 No. 53, reg. MoJ No. 38336 | In force; **replaced** NBRK Resolution No. 48 of 27.03.2018, which no longer applies | https://adilet.zan.kz/rus/docs/V2600038336 |

Checked and **rejected**:
* NBRK Resolution No. 48 of 2018 (information security): it lost force on 03.04.2026.
* The 1999/2001 cash rules: they lost force on 01.01.2020.
* An unverified prg.kz page on cash-room technical requirements: its jurisdiction and status could not be confirmed.

## 2. How a rule is matched to a video fragment

```
detected type (visible fact)  →  type_rules[type] in the registry  →  rule(s): document + clause + verified wording  →  text in description
```

1. Detection outputs only a **visible fact** (for example, "banknotes on the desk, the cashier walks away").
2. The type is looked up in `type_rules` to get rule ids and a grounding level.
3. The description is assembled as: *what is visible* + *why it may be a violation*, then «Правило: <short_ru>, <clause>».
   - Grounding **strong**: the rule is cited as is.
   - Grounding **medium** or **weak**: the text adds «возможное нарушение требований…» or «косвенно связано с…», and notes when the specific requirement is set by the bank's internal documents.
   - Grounding **none**: «нормативное основание не верифицировано». No document is named.
4. The service never states guilt or intent, and never relies on anything not visible (ledger postings, reconciliations, documents it cannot read).

Example description (T05):
> На видео после ухода клиента на столе кассира остаётся пачка банкнот; в это время в помещение входит и подходит к столу другой человек. Это может указывать на хранение ранее принятых наличных вне запираемых устройств. Правило: Правила кассовых операций и инкассации (пост. Правления НБРК от 29.11.2019 № 231), п. 20, ч. 2.

## 3. Mapping per type

| Type | Rule (verified) | What the rule says (short) | Visible fragment it is matched to | Grounding |
|---|---|---|---|---|
| T01 Valuables or documents left unattended | NBRK_231 para. 20 part 2; para. 3 subpara. 7; para. 2 subpara. 17 (documents count as valuables); para. 22 (only if a client is present) | Only the current client's cash may be at the workplace; all previously accepted cash is kept in lockable devices | Banknotes, collection bag or documents on the desk; cashier walks away | **medium** (strong for cash, weaker for documents) |
| T02 PC left unlocked, unattended | ARDFM_53 para. 71 subpara. 9; NBRK_231 para. 2 subpara. 17 | After 30 minutes of inactivity the system ends the session or locks the workstation | Screen shows working content with nobody at the desk | **weak / indirect**. There is no explicit "lock when leaving" requirement |
| T03 Safe left open | NBRK_231 para. 20 part 2; para. 80 | Cash is kept in lockable devices; key handling is set by internal documents | Safe door open, cashier away from it | **medium** |
| T04 Key left in safe lock | NBRK_231 para. 80 (para. 76 for vault or safe-room doors) | The order for keeping keys to workplace safes is set by internal documents that must exclude unauthorized use | Key in the lock, cashier away or out of the room | **medium**. The detail lives in internal bank documents |
| T05 Previous cash exposed when the next client approaches | NBRK_231 para. 20 part 2 | Only the current client's cash may be at the workplace | Banknote stack on the desk while the next person comes up | **strong** |
| T06 Unauthorized person in the cash zone | NBRK_14 para. 8; para. 16 subpara. 1; NBRK_231 para. 75 (vault or safe room only) | The cash unit is isolated by a closing entrance; each cash room has a lockable door | A non-cashier on the cashier's side; an item handed to a third person | **medium** |
| T07 Cash-room door not closed | NBRK_14 para. 8; para. 16 subpara. 1 | Single closing service entrance; door that locks | Cashier enters and leaves the door open | **medium** |
| T08 Client cash counted without the client | NBRK_231 para. 20 part 1; para. 31 subpara. 4 | Cash is accepted and counted in the client's presence; cash being issued is counted in the client's presence | Client leaves, then the cash is counted | **strong** |
| T09 Collector bag accepted without a document check | NBRK_231 para. 24 (+ para. 101) | Acceptance from collectors is based on an accompanying document; the cashier checks it, the bag's integrity and its seal | No document handover and no bag inspection visible | **strong** (the description must say "not visible", not "not checked") |
| T10 Personal items at the desk | NBRK_231 para. 12 part 1 | Workers' personal items are kept outside the cash unit | Coat and handbag on the cash desk | **strong** |
| T11 Food or drink at the desk | NBRK_231 para. 12 part 2 (interpretation) | Only medicines, medical devices and phones without a camera may be brought into the cash unit | Cashier drinks from a cup at the desk with cash on it | **weak**. Interpretation: food and drink are not on the allowed list; no explicit ban was found |
| T12 Cash handed to a client at the desk | none | No clause found that forbids a handover the video can show; the registry keeps an empty rule list | Banknotes passed across the desk | **none**. The description says the basis is not verified and the reviewer must judge |
| VISUAL_ANOMALY | none | | | **none** |

Extension rules, kept ready for unseen test types:
* Using a camera phone in the cash room: para. 12 part 2.
* Cashier leaving the workplace while a client is still being served: para. 22.
* Cash issued without being counted in front of the client: para. 31 subpara. 4.
* Counting without a controlling worker: para. 42 (weak; the second person's role cannot be seen on video).

## 4. Summary of grounding
* **Strong (4):** T05, T08, T09, T10.
* **Medium (5):** T01, T03, T04, T06, T07.
* **Weak (2):** T02 (only an automatic-lock requirement exists), T11 (interpretation of para. 12).
* **None:** T12 (cash handover at the desk) and the generic VISUAL_ANOMALY path.
* No type had to be dropped. T02 and T11 stay, but their descriptions must use the softened «косвенно/толкование» wording.
