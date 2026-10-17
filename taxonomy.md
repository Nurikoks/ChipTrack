# Violation taxonomy, draft v0.1 (Phase 1, for review)

Status: **draft, waiting for team confirmation.** Every type below comes from frames we actually looked at. Rule citations are deliberately missing; they are Phase 2 work and will be added only from verified Republic of Kazakhstan (RK) sources.

Evidence sheets: `docs/taxonomy_frames/<TYPE>.jpg`. Each row is one clip: 3 frames (before, at the label, after) plus an optional zoom tile. Orange captions are violation clips, green captions are normal clips used as hard negatives.
Per-clip details: [data/labels_enriched.csv](data/labels_enriched.csv).

## 1. What the data looks like

| Fact | Evidence |
|---|---|
| 42 label rows, 42 files, all open. H.264 High, 464×848 portrait, ~29.8–30 fps (slightly variable), 11.6–142.5 s, median 22.7 s, 2.3–27.4 MB | `data/video_meta.csv` |
| **2 byte-identical pairs:** X-4 = X-47 (labelled violation @57 s and violation @68 s, so one clip with two events) and X-5 = X-48 (both normal) | MD5 in `data/video_meta.csv` |
| So there are 40 unique clips and 21 unique violation clips with 22 events. Cross-validation must group X-4/X-47 and X-5/X-48 together | |
| One fixed high corner camera, but **at least 4 room set-ups**: (A) cashier seated with her back to camera, client chair opposite, safe on the right; (B) cashier facing the camera, laptop, safe right of the desk; (C) desk with the counting machine on the right and no laptop; (D) same as C plus a blue chair. The cash-room door (glass) is always top-left | overview sheets |
| Actors: 2 female cashiers (pink cardigan / white blouse), 2–3 male clients/collectors. The green fabric bag is the cash-collection bag | |
| Label timestamps fall inside the event, usually at the **moment the cashier walks away** (state violations) or at the **peak of the action** (action violations) | see "label pattern" per type |

## 2. Timecode convention (proposed)

* **Action violations** (drinking, handing an item to a third party, putting personal items down): `timecode` = the frame where the action is most visible.
* **State / omission violations** (safe left open, key left in lock, cash or PC left unattended, door not closed): `timecode` = the **moment of abandonment**, i.e. the first frame where the unsafe state exists **and** the responsible person turns away or walks off. This matches 9 of the 10 state labels (X-11, X-15, X-20, X-22, X-24, X-26, X-28, X-32, X-33). X-13 is the exception: there the label falls on the safe interaction.
* Evidence frames: frame 1 is the abandonment/action frame; frame 2 is a later frame showing the unsafe state persisting (person absent); optional frame 3 is a zoom on the object (key, screen, banknotes).

## 3. Candidate types

Names are in Russian because they go into the API `type` field and the UI.

### T01 «Ценности/документы оставлены без присмотра на рабочем месте» (UNATTENDED_VALUABLES)
* **Definition:** the cashier leaves the workplace (walks away or out of the room) while cash, a cash-collection bag or documents stay openly on the desk.
* **Visible cues:** banknotes, green bag or document folder on the desk; no person at the desk; the cashier walking away or the room empty afterwards.
* **Examples:** X-26 @9 (banknotes), X-15 @15 (collection bag), X-24 @8 (folder plus cup; medium confidence).
* **Label pattern:** the moment the cashier stands up or turns to leave.
* **Hard negatives:** X-25, X-27, X-12 (desk cleared before leaving). A cup alone left on the desk is not enough.

### T02 «Рабочее место покинуто без блокировки компьютера» (PC_UNLOCKED_UNATTENDED)
* **Definition:** the cashier leaves while the computer shows working content (an open spreadsheet) instead of a lock screen, a black screen or a closed lid.
* **Visible cues:** a bright white or spreadsheet-like screen with nobody at the desk. In X-11 the screen keeps glowing after the light is switched off.
* **Examples:** X-28 @7, X-11 @18.
* **Hard negatives:** X-23 and X-29 (lock-screen wallpaper), X-25 and X-27 (black screen), X-12 (lid closed). **Caution:** an open laptop is normal; only *visible working content* separates the classes. The screen is small, so this type is angle-sensitive.

### T03 «Сейф оставлен открытым без присмотра» (SAFE_LEFT_OPEN)
* **Definition:** the safe door stays open after the cashier moves away from it (back to the desk or out of the room).
* **Visible cues:** the safe door panel is swung open and nobody's hand is at the safe.
* **Example:** X-20 @5 (door open with the key in it while she sits at the desk).
* **Hard negatives:** X-21 (closed before sitting down); X-35, X-39 and X-8, where the safe is open *while the cashier is at it*. That is normal, even with a client present.

### T04 «Ключ оставлен в замке сейфа» (KEY_LEFT_IN_SAFE)
* **Definition:** a key stays in the safe lock while the cashier is away from the safe or leaves the room.
* **Visible cues:** a small object sticking out of the lock on the safe door; zoom tiles in the sheet.
* **Examples:** X-22 @14 (key visible from 5 s to 20.5 s, cashier leaves); X-13 @47 (key in the lock at 44 s and 58–65 s, cashier leaves the room at 58–62 s).
* **Hard negatives:** X-23, X-14, X-21 (keyhole empty).
* **Risk:** the key is only a few pixels at 464×848 and needs a high-resolution crop of the safe. Expect lower recall on a new camera angle.

### T05 «Наличные предыдущей операции оставлены открыто при подходе следующего клиента» (CASH_EXPOSED_NEXT_CLIENT)
* **Definition:** after a client leaves, banknotes stay openly on the desk while the next person enters and approaches.
* **Visible cues:** a stack of banknotes on the desk, a person entering through the door and walking to the desk, and the cashier not putting the cash away.
* **Examples:** X-4 @57, X-38 @33. X-47 @68 (the same file as X-4) may be a continuation: the old stack is still there during the second client's operation (low confidence).
* **Hard negatives:** X-5, X-19, X-39 (desk clear at changeover).

### T06 «Постороннее лицо в кассовой зоне» (UNAUTHORIZED_PERSON_CASH_ZONE)
* **Definition:** a person who is not the cashier comes to the cashier's side of the desk, or joins an ongoing operation, and/or receives an item from the cashier.
* **Visible cues:** a second standing person next to the cashier, or at the desk corner on the cashier side, while cash is being handled; an item passing from the cashier to that person.
* **Examples:** X-30 @13 (stands beside the cashier during counting), X-18 @20 (third person enters during service and receives an item).
* **Hard negatives:** a client seated *opposite* the cashier (all client clips); the collector standing at the desk during a regular handover (X-16).
* **Note:** roles are inferred only from position and behaviour (who sits opposite, who carries the bag). No identification of people.

### T07 «Дверь кассового помещения не закрыта» (CASH_ROOM_DOOR_NOT_CLOSED)
* **Definition:** the cashier enters the cash room and goes to the workplace without closing the door behind her.
* **Visible cues:** the door stays ajar or open after the person passes; no "turn back and close" motion. In X-33 an outsider then appears in the open doorway.
* **Examples:** X-32 @5, X-33 @6.
* **Hard negative:** X-31 (cashier turns back and closes the door at 8–10 s).

### T08 «Пересчёт наличных клиента в его отсутствие» (COUNT_WITHOUT_CLIENT)
* **Definition:** the client hands over cash and leaves before it is counted; the cashier counts it after the client has gone.
* **Visible cues:** the client's departure overlaps with the cash being loaded into or run through the counting machine; the client chair is empty during the count.
* **Example:** X-34 @13.
* **Hard negatives:** X-3, X-35, X-37, X-39 (client stays seated during the count).

### T09 «Приём сумки у инкассатора без видимой проверки документов» (COLLECTOR_NO_DOC_CHECK)
* **Definition:** the collector's bag is accepted with no visible exchange or check of a document beforehand, and the collector leaves at once.
* **Visible cues:** the bag goes straight onto the desk; no paper or card passes between them; the cashier does not examine anything before taking the bag.
* **Example:** X-17 @7.
* **Hard negative / reference procedure:** X-16. The collector hands a document (9 s), the cashier examines it (10–14 s) and returns it, and only then is the bag handed over (16 s).
* **Note:** this is a *visible absence* of a check. The description must say only "no document handover is visible", never "the documents were not checked".

### T10 «Личные вещи на рабочем месте кассира» (PERSONAL_ITEMS_AT_DESK)
* **Definition:** outerwear or a personal bag is put on the cash desk.
* **Visible cues:** a coat or handbag on the desk next to the counting machine.
* **Example:** X-40 @17.
* **Hard negatives:** none observed in normal clips.

### T11 «Употребление напитков/пищи на рабочем месте при работе с наличными» (FOOD_DRINK_AT_DESK)
* **Definition:** the cashier drinks or eats at the desk while cash is on it.
* **Visible cues:** a cup raised to the mouth; banknotes on the desk.
* **Example:** X-46 @11.
* **Hard negative:** X-25 (cup on the desk, not drinking).

### T12 «Передача банкнот клиенту у кассового стола» (CASH_HANDOVER_AT_DESK)
* **Definition:** banknotes pass across the desk between the cashier and a visitor. Type confirmed by the team for X-1 @22.
* **Visible cues:** a hand holding banknotes stretched across the desk toward the other person; hand-to-hand passing.
* **Example:** X-1 @22 (cashier hand-counts notes from the machine, then passes them to the seated client at 22.5–23 s). X-2 @10 is also caught (client lays an item on the desk at 9 s), but its type is still unresolved.
* **Hard negatives:** none separable. The VLM scores normal client transactions (X-3, X-5, X-19, X-35, X-37, X-39) as high as X-1, so T12 is recall-only for now. No verified rule is cited.

### Generic path «Визуальная аномалия относительно штатной процедуры» (VISUAL_ANOMALY)
This is reserved for unseen types in the test (FAQ #9). It will be reported only with a strictly factual description and the rule field marked "not verified". The verdict policy for this path is Phase 3.

## 4. Unresolved: needs human review (3 events)

I could not find a visible element that separates these clips from normal ones. I won't guess.
X-1 @22 was resolved by the team as T12 (see §3) and is no longer listed here.

| Clip @ label | What is visible | Closest normal clip | Question for the team |
|---|---|---|---|
| **X-2 @10** | Client puts a blue item (banknotes or an ID?) on the desk (9 s); cashier picks it up (10.5 s), handles it (11–13.5 s), loads the machine (14 s) | X-3, X-37 | Is the item cash or a document? What is the violation? |
| **X-44 @16** | Cashier alone takes notes from the counter (12–13.5 s) and hand-counts them (14–20 s) | X-43, X-45 (almost identical) | What separates X-44 from X-43/X-45? |
| **X-47 @68** | Second client's operation while the first client's stack is still on the desk (64–70 s) | X-5 | Is it T05 continuing, or "cash of two clients mixed"? |

Please watch these three clips at normal speed (motion is easier to read than stills) and tell me what you see. If they turn out to be 1–2 new types, I'll add them to the list.

## 5. Implications for the architecture (preview, not decided)

* 8 of the 11 types are **state + presence** problems: an object is in an unsafe state (safe door, key, screen, banknotes, bag, door) **and** the responsible person is away. That suggests a per-frame "scene state" (objects + person positions relative to desk, safe and door) plus a temporal state machine, not frame-level action recognition.
* Cues that survive a camera change: open safe door, banknotes or bag on the desk, a person next to the cashier, the door gap, a cup at the mouth (pose). Fragile cues: key in lock and screen content (tiny), so they need high-resolution crops around detected objects.
* **Many violations differ from normal clips only by one small detail** (key, screen content, a turn-back motion). A local VLM checklist on targeted crops is the likely verifier. A trained classifier on 21 positives would overfit to room layout.
