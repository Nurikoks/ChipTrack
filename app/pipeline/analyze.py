"""End-to-end analysis: video -> perception -> scene -> events -> VLM checklist -> violations.

Design rules (see docs/architecture.md):
* No fixed pixel zones: desk / safe / laptop / door are grounded per video by the VLM.
* Events are relations between people and those anchors over time (relative, angle-agnostic).
* Every reported finding is a visible fact verified by a constrained yes/no VLM question.
* Deadline-aware: checks run in priority order; when time runs out the best partial result is returned.
"""
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .. import config
from . import video
from .models import VLM, Detector, crop
from .rules import build_violation
from ..telemetry import emit

log = logging.getLogger("forte.analyze")

# ---- yes/no questions (English: best accuracy for the VLM; never shown to users) ----
Q = {
    "money": "Are there banknotes (paper money) lying on this part of the table?",
    "money2": "Is there a stack of money or a pile of banknotes here?",
    "bag": "Is there a green fabric bag lying on the desk?",
    "bag_alone": "Is a green fabric bag lying on the desk by itself, not held in anyone's hands?",
    "docs": "Are there papers, a document folder or a tablet left lying on the desk?",
    "screen": "This is a close-up of a laptop. Is its screen showing a spreadsheet or document with text and table rows (not a lock screen, wallpaper or black screen)?",
    "lid_open": "This is a close-up of a laptop. Is the laptop lid open so that its screen is visible?",
    "lock_screen": "This is a close-up of a laptop. Is its screen showing a lock screen, a photo wallpaper or a black screen?",
    "safe_open": "This is a close-up of a small metal safe. Is the safe door open?",
    "key": "This is a close-up of a safe. Is a metal KEY inserted into the lock, with its shaft or bow projecting from the door? A dark empty keyhole alone is NOT a key.",
    "personal": "Is there a coat, a jacket or a woman's handbag (personal belongings, not a green cash bag) lying on the desk?",
    "drinking": "Is the person at the desk drinking from a cup right now?",
    "cup_in_hand": "Is the person holding a cup, a mug or a glass in their hand (not paper or banknotes)?",
    "banknotes_in_hand": "Is the cashier holding paper banknotes in her hands?",
    "door_open": "This is a close-up of a glass door. Is the door standing open (not closed)?",
    "carry_bag": "Is a person carrying a green fabric cash-collection bag?",
    "doc_handover": "Is someone holding out a small sheet of paper or an ID card toward another person (not a bag)?",
    "occupied": "Is a person sitting at the desk or standing right at it, at the workplace?",
    "counting": "Is the person at the desk counting banknotes or putting money into a banknote counting machine?",
    "extra_person": "Is there a standing person right next to the seated worker, on the same side of the desk as the worker (not a visitor sitting opposite)?",
    "visitor_seated": "Is anyone sitting on the visitor side of the desk, across from the worker?",
    "at_safe": "Is a person standing at the small metal safe and using it right now?",
    "id_shown": "Is the visitor showing an identity card or passport and is the cashier inspecting it? Banknotes, cash bundles, receipts and ordinary sheets of paper are NOT identity documents.",
    "handover": "Is the visitor handing money, a bag or a package to the worker or putting it on the desk?",
    "two_visitors": "Are there two different visitors at the worker's desk at the same time (besides the worker)?",
    "merging": "Is the worker putting one stack of banknotes on top of another stack, combining two piles of money into one?",
}
# decision thresholds on P(yes) (set on the training probe set; recall first)
TH = {"money": 0.3, "bag": 0.6, "docs": 0.55, "screen": 0.75, "safe_open": 0.4, "key": 0.5,
      "personal": 0.7, "drinking": 0.6, "door_open": 0.72, "carry_bag": 0.6, "doc_handover": 0.7,
      "counting": 0.6, "extra_person": 0.6, "visitor_seated": 0.5, "at_safe": 0.5, "occupied": 0.5,
      "id_shown": 0.6, "handover": 0.5, "two_visitors": 0.85, "merging": 0.6}


@dataclass
class Finding:
    type_id: str
    t: float                      # timecode (s)
    conf: float
    observed: str                 # factual Russian sentence
    frames: list[tuple[int, object]] = field(default_factory=list)  # (frame index, highlight box or None)


class Budget:
    def __init__(self, deadline: float):
        self.deadline = deadline
        self.skipped = 0

    def left(self) -> float:
        return self.deadline - time.time()

    def ok(self, need: float = 2.0) -> bool:
        result = self.left() > need
        if not result:
            self.skipped += 1
        return result


class Analyzer:
    def __init__(self):
        self.det = Detector(config.DETECTOR_ID)
        self.vlm = VLM(config.VLM_ID, four_bit=config.VLM_4BIT)
        try:
            from .hands import HandContact
            self.hands=HandContact(config.POSE_PATH)
        except Exception as exc:
            self.hands=None
            emit('inference','hand_detector_unavailable',error=str(exc))
        self.trace = []
        self.scene, self.room_tracks = {}, []

    # ------------------------------------------------------------------ main
    def analyze(self, path, job) -> dict:
        t0 = time.time()
        budget = Budget(job.deadline - 20)  # keep 20 s for encoding the report
        self.vlm.calls, self.trace = 0, []
        self.ambiguities=[]
        partial = False
        issues, coverage = [], []

        job.stage, job.progress = "decoding", 0.05
        clip = video.sample(path, config.SAMPLE_FPS)

        job.stage, job.progress = "perception", 0.15
        dets = self.det(clip.frames)
        raw = self._people(dets, clip)

        job.stage, job.progress = "scene", 0.3
        scene = self._scene(clip, raw)
        self.scene = scene
        people = self._room_people(raw, scene, clip)
        self.people=people

        job.stage, job.progress = "checks", 0.4
        tl = self._timeline(clip, people, scene)
        findings: list[Finding] = []
        from . import temporal
        contexts, observations = [], []
        try:
            emit("inference", "stage", stage="temporal_events", job_id=getattr(job, "id", "evaluation"))
            contexts, observations, temp_issues = temporal.extract(self, clip, tl, budget)
            issues.extend(temp_issues)
            if budget.ok(8):
                ff, ii = temporal.findings(self, clip, observations)
                findings.extend(ff)
                issues.extend(ii)
        except Exception as exc:
            partial = True
            issues.append(f"Временные события: {type(exc).__name__}")
            log.exception("temporal extraction failed")
        checks = [self._check_departures, self._check_changeover, self._check_entries, self._check_service,
                  self._check_periodic]
        for i, chk in enumerate(checks):
            if not budget.ok(8):
                partial = True
                log.warning("deadline: skipping %s and later checks", chk.__name__)
                break
            try:
                previous_skips = budget.skipped
                emit("inference", "stage", stage=chk.__name__, job_id=getattr(job, "id", "evaluation"))
                findings += chk(clip, people, scene, tl, budget)
                coverage.append({"check": chk.__name__, "status": "partial" if budget.skipped > previous_skips else "completed"})
                partial |= budget.skipped > previous_skips
            except Exception as exc:
                partial = True
                coverage.append({"check": chk.__name__, "status": "failed", "reason": type(exc).__name__})
                log.exception("check %s failed; continuing", chk.__name__)
            job.progress = 0.4 + 0.5 * (i + 1) / len(checks)

        job.stage = "report"
        partial |= budget.skipped > 0
        issues.extend(self.ambiguities)
        if clip.duration - clip.times[-1] > max(2, 2 / config.SAMPLE_FPS):
            partial = True
            issues.append("Видео длиннее проанализированного отрезка")
        if not scene:
            issues.append("Не удалось определить объекты кассового рабочего места")
        for anchor, label in (("desk", "рабочий стол"), ("safe", "сейф"), ("laptop", "экран рабочего места"), ("door", "дверь")):
            if anchor not in scene:
                issues.append(f"Не удалось локализовать {label}; связанные проверки недоступны для этого ракурса")
        if any(o['scores'].get('cash_safe', 0) >= .7 for o in observations) and not any(o['scores'].get('machine', 0) >= .6 for o in observations):
            issues.append("Видно перемещение денег к сейфу, но полный машинный пересчёт по выбранным кадрам не подтверждён; проверьте T15")
        if any(c["context"] == "unknown" for c in contexts):
            issues.append("Контекст части ролика не определён")
        from .features import make_windows, apply_classifier
        windows = make_windows(clip.duration, self.trace, contexts, tl)
        findings=self._merge(findings)
        base_predictions = [{'type': f.type_id, 't': f.t, 'conf': f.conf} for f in findings]
        base_review_required=bool(partial or issues)
        base_verdict = 'violation' if base_predictions else 'needs_review' if partial or issues else 'no_violation'
        findings, classifier_issues = apply_classifier(windows, getattr(job, "version", None), clip,findings)
        issues.extend(classifier_issues)
        findings = self._merge(findings)
        violations = []
        for f in findings:
            frames = [{"timestamp_sec": round(clip.times[idx], 1),
                       "image_base64": video.to_b64_jpeg(clip.frames[idx], box=box)} for idx, box in f.frames[:3]]
            violations.append(build_violation(f.type_id, f.t, f.observed, frames, f.conf))
        log.info("analysis done in %.1fs, vlm calls=%d, findings=%d, partial=%s",
                 time.time() - t0, self.vlm.calls, len(violations), partial)
        out = {"verdict": "violation" if violations else "needs_review" if partial or issues else "no_violation", "violations": violations,
               "partial": partial, "video_duration_sec": round(clip.duration, 1),
               "processing_time_sec": round(time.time() - t0, 1), "review_reasons": list(dict.fromkeys(issues)),
               "contexts": contexts, "events": observations, "checks": coverage,
               "confidence_kind": "uncalibrated_score", "_windows": windows,
               "_trace": self.trace, "feature_schema": "events-v1",
               "_base_predictions": base_predictions, "_base_verdict": base_verdict,
               "_base_review_required":base_review_required}
        from ..versions import engine_fingerprint
        out['_extractor'] = engine_fingerprint()
        if config.DEBUG:
            out["debug"] = {"scene": {k: [int(v) for v in b] for k, b in scene.items()},
                            "n_people": [round(float(x), 1) for x in tl["n"]],
                            "attended": [int(x) for x in tl["attended"]],
                            "entries": [(round(clip.times[i], 1), b, a) for i, b, a in self._count_events(tl, True)],
                            "exits": [(round(clip.times[i], 1), b, a) for i, b, a in self._count_events(tl, False)],
                            "trace": self.trace}
        return out

    # ------------------------------------------------------------ perception
    def _people(self, dets, clip):
        """Per-frame person boxes, dropping people seen through/reflected in glass (small, far)."""
        heights = [d["box"][3] - d["box"][1] for fr in dets for d in fr if d["label"] == "person" and d["score"] > 0.5]
        ref = np.percentile(heights, 80) if heights else clip.height
        return [[d for d in fr if d["label"] == "person" and d["score"] > 0.45
                 and (d["box"][3] - d["box"][1]) >= 0.45 * ref] for fr in dets]

    @staticmethod
    def _iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
        return ix * iy / u if u > 0 else 0.0

    def _track(self, raw, max_gap=4):
        """Greedy IoU tracker -> list of tracks, each {frame_idx: box}."""
        tracks, active = [], []   # active: (track_id, last_idx)
        for i, ps in enumerate(raw):
            used = set()
            for p in sorted(ps, key=lambda d: -d["score"]):
                best, bj = 0.25, None
                for k, (tid, last) in enumerate(active):
                    if i - last > max_gap or k in used:
                        continue
                    s = self._iou(p["box"], tracks[tid][last])
                    if s > best:
                        best, bj = s, k
                if bj is None:
                    tracks.append({i: p["box"]})
                    active.append((len(tracks) - 1, i))
                    used.add(len(active) - 1)
                else:
                    tid, _ = active[bj]
                    tracks[tid][i] = p["box"]
                    active[bj] = (tid, i)
                    used.add(bj)
            active = [(t, l) for t, l in active if i - l <= max_gap]
        return tracks

    def _room_people(self, raw, scene, clip):
        """Keep only tracks that are physically in the room: the track reaches the workplace
        (desk/safe zone) or its feet go below the door's lower edge. People seen through the glass
        wall never do either. Gaps inside a kept track are filled (occlusions)."""
        desk, safe, door = scene.get("desk"), scene.get("safe"), scene.get("door")
        out = [[] for _ in raw]
        self.room_tracks = []
        for tr in self._track(raw):
            if len(tr) < 3:
                continue                      # 1-2 frame blips
            in_room = False
            for b in tr.values():
                if self._near(b, desk, 0.3, 0.6) or (safe is not None and self._cover(b, safe) > 0.25):
                    in_room = True
                elif door is not None and b[3] > door[3] + 0.12 * (door[3] - door[1]):
                    in_room = True
                if in_room:
                    break
            if not (desk or door):
                in_room = True
            if not in_room:
                continue
            self.room_tracks.append(tr)
            idx = sorted(tr)
            for a, b in zip(idx, idx[1:] + [idx[-1]]):
                for j in range(a, b if b > a else a + 1):
                    out[j].append({"box": tr[a], "score": 1.0, "label": "person"})
        return out

    @staticmethod
    def _cover(pbox, anchor):
        """Fraction of `anchor` covered by the person box."""
        ix = max(0, min(pbox[2], anchor[2]) - max(pbox[0], anchor[0]))
        iy = max(0, min(pbox[3], anchor[3]) - max(pbox[1], anchor[1]))
        a = (anchor[2] - anchor[0]) * (anchor[3] - anchor[1])
        return ix * iy / a if a > 0 else 0.0

    def _scene(self, clip, people):
        """Ground the static anchors once per video on the least-occluded frames."""
        counts = [len(p) for p in people]
        cand = sorted(range(len(clip.frames)), key=lambda i: (counts[i], i))[:3]
        scene = {}
        for name, query in [("desk", "desk"), ("safe", "small grey metal safe"), ("laptop", "laptop"),
                            ("door", "glass door")]:
            for i in cand:
                box = self.vlm.ground(VLM.to_pil(clip.frames[i]), query)
                if box and self._plausible(name, box, clip):
                    scene[name] = box
                    break
        log.info("scene anchors: %s", {k: [int(v) for v in b] for k, b in scene.items()})
        return scene

    @staticmethod
    def _plausible(name, box, clip):
        w, h = box[2] - box[0], box[3] - box[1]
        area = w * h / (clip.width * clip.height)
        return {"desk": 0.02 < area < 0.6, "safe": 0.002 < area < 0.2, "laptop": 0.001 < area < 0.15,
                "door": 0.02 < area < 0.7}[name]

    # -------------------------------------------------------------- timeline
    def _timeline(self, clip, people, scene):
        """Workplace = desk ∪ safe. A person attends it when the box centre is within the anchor
        expanded relative to its own size (no absolute pixel thresholds)."""
        n = np.array([len(p) for p in people], float)
        att = np.zeros(len(people), bool)
        near_safe = np.zeros(len(people), bool)
        beside_safe = np.zeros(len(people), bool)
        for i, ps in enumerate(people):
            safe = scene.get("safe")
            if scene.get("desk") is not None:
                att[i] = any(self._near(p["box"], scene["desk"], 0.12, 0.6) or
                             (safe is not None and self._cover(p["box"], safe) > 0.3) for p in ps)
            else:
                att[i] = len(ps) > 0
            # strict: person box covers part of the safe (used to decide "nobody at the open safe")
            near_safe[i] = safe is not None and any(self._cover(p["box"], safe) > 0.2 for p in ps)
            # loose: standing beside the safe still counts as being at the workplace (departure logic)
            beside_safe[i] = safe is not None and any(
                self._cover(p["box"], self._expand(safe, 0.5, clip)) > 0.05 and p["box"][3] > safe[1] for p in ps)
        k = 3  # ~1.5 s median smoothing at 2 fps
        sm = lambda a: np.array([np.median(a[max(0, j - k // 2): j + k // 2 + 1]) for j in range(len(a))])
        return {"n": sm(n), "attended": sm(att.astype(float)) > 0.5,
                "near_safe": sm(near_safe.astype(float)) > 0.5, "beside_safe": sm(beside_safe.astype(float)) > 0.5,
                "t": np.asarray(clip.times)}

    @staticmethod
    def _expand(b, m, clip):
        w, h = b[2] - b[0], b[3] - b[1]
        return (max(0, b[0] - m * w), max(0, b[1] - m * h), min(clip.width, b[2] + m * w),
                min(clip.height, b[3] + m * h))

    @staticmethod
    def _near(pbox, anchor, mx, my):
        """Person box centre inside `anchor` expanded by mx*width and my*height."""
        if anchor is None:
            return False
        cx, cy = (pbox[0] + pbox[2]) / 2, (pbox[1] + pbox[3]) / 2
        w, h = anchor[2] - anchor[0], anchor[3] - anchor[1]
        return anchor[0] - mx * w <= cx <= anchor[2] + mx * w and anchor[1] - my * h <= cy <= anchor[3] + my * h

    @staticmethod
    def _segments(flags, val=True, min_len=4):
        """Runs of `flags == val` lasting >= min_len frames -> list of (start_idx, end_idx)."""
        out, s = [], None
        for i, f in enumerate(list(flags) + [not val]):
            if f == val and s is None:
                s = i
            elif f != val and s is not None:
                if i - s >= min_len:
                    out.append((s, i - 1))
                s = None
        return out

    def _count_events(self, tl, up: bool, min_len=4):
        """Entries (up=True) / exits: (index, count before, count after).

        Preferred: room tracks that start (entry) or end (exit) at the grounded door — robust to people
        standing up or turning (which only re-start a track mid-room). Fallback without a door anchor:
        sustained changes of the person count."""
        door = (self.scene or {}).get("door")
        n = tl["n"]
        if door is not None and self.room_tracks:
            ev = []
            min_track = int(5 * config.SAMPLE_FPS)   # passers-by behind the glass are brief
            for tr in self.room_tracks:
                idx = sorted(tr)
                if len(idx) < max(min_len, min_track):
                    continue
                i = idx[0] if up else idx[-1]
                if (up and i < 2) or (not up and i > len(n) - 3):
                    continue                  # present from the start / until the end
                if not self._near(tr[i], door, 0.35, 0.25):
                    continue
                # continuation of another track at the same place (person stood up / turned): not an entry/exit
                win = range(max(0, i - 4), i) if up else range(i + 1, min(len(n), i + 5))
                if any(self._iou(tr[i], o[j]) > 0.2 for o in self.room_tracks if o is not tr for j in win if j in o):
                    continue
                ev.append((i, int(n[max(0, i - 2)]), int(n[min(len(n) - 1, i + 3)])))
            return sorted(ev)
        ev = []
        for i in range(1, len(n)):
            ch = n[i] > n[i - 1] if up else n[i] < n[i - 1]
            if ch and all((n[j] >= n[i]) if up else (n[j] <= n[i]) for j in range(i, min(len(n), i + min_len))):
                ev.append((i, int(n[i - 1]), int(n[i])))
        return ev

    def _ask_crop(self, clip, i, box, key, margin=0.25):
        p = self.vlm.p_yes(crop(clip.frames[i], box, margin), Q[key])
        self.trace.append((round(clip.times[i], 1), key, round(p, 3)))
        return p

    def _ask_full(self, clip, i, key):
        p = self.vlm.p_yes(VLM.to_pil(clip.frames[i]), Q[key])
        self.trace.append((round(clip.times[i], 1), key, round(p, 3)))
        return p

    def _money(self, clip, i, desk):
        """Banknotes on the desk: two overlapping half-desk tiles, each upscaled (small stacks are
        only a few pixels), two phrasings; returns the max P(yes)."""
        x1, y1, x2, y2 = desk
        w = x2 - x1
        best = 0.0
        for tb in ((x1, y1, x1 + 0.6 * w, y2), (x1 + 0.4 * w, y1, x2, y2)):
            img = crop(clip.frames[i], tb, 0.1, min_side=512)
            for key in ("money", "money2"):
                p = self.vlm.p_yes(img, Q[key])
                self.trace.append((round(clip.times[i], 1), key, round(p, 3)))
                best = max(best, p)
                if best >= 0.9:
                    return best
        return best

    def _idx(self, clip, i):
        return int(min(len(clip.frames) - 1, max(0, i)))

    # ---------------------------------------------------------------- checks
    def _check_departures(self, clip, people, scene, tl, budget):
        """Workplace (desk ∪ safe) left unattended >= 2 s -> what was left behind?"""
        out = []
        gaps = [g for g in self._segments(tl["attended"], False, min_len=4) if g[0] > 0 and tl["attended"][g[0] - 1]]
        # 2-D geometry cannot separate "at the desk" from "at the door behind the desk" in some layouts,
        # so a VLM occupancy scan (every ~2 s) proposes departures too.
        step = max(1, int(round(2.0 * config.SAMPLE_FPS)))
        occ = []
        for i in range(0, len(clip.frames), step):
            if not budget.ok(20):
                break
            occ.append((i, bool(tl["beside_safe"][i]) or self._ask_full(clip, i, "occupied") >= TH["occupied"]))
        for (i0, o0), (i1, o1), (i2, o2) in zip(occ, occ[1:], occ[2:]):
            if o0 and not o1 and not o2 and not any(s <= i1 <= e + step for s, e in gaps):
                gaps.append((i1, min(len(clip.frames) - 1, i2 + step)))
        gaps = sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)[:4]
        for s, e in sorted(gaps):
            if not budget.ok(6):
                break
            t_leave = max(0.0, clip.times[s] - 1.0)  # convention: the moment the worker gets up / turns away
            leave_idx = self._idx(clip, s - 2)       # person still visible walking away
            iv = min(e, s + 4)                       # ~2 s later: nobody at the workplace
            if "desk" in scene:
                pm = self._money(clip, iv, scene["desk"])
                pb = self._ask_crop(clip, iv, scene["desk"], "bag")
                if pm >= TH["money"] or pb >= TH["bag"]:
                    what = "остаётся зелёная инкассаторская сумка" if pb >= TH["bag"] and pb > pm \
                        else "остаются банкноты"
                    out.append(Finding("T01_UNATTENDED_VALUABLES", t_leave, max(pm, pb),
                                       f"На видео работник отходит от рабочего места, на столе {what}; "
                                       f"рядом со столом никого нет", [(leave_idx, None), (iv, scene["desk"])]))
                elif self._ask_crop(clip, iv, scene["desk"], "docs") >= TH["docs"]:
                    out.append(Finding("T01_UNATTENDED_VALUABLES", t_leave, 0.6,
                                       "На видео работник отходит от рабочего места, на столе остаются документы/папка; "
                                       "рядом со столом никого нет", [(leave_idx, None), (iv, scene["desk"])]))
            if "laptop" in scene:
                ps = self._ask_crop(clip, iv, scene["laptop"], "screen", margin=0.4)
                if ps >= TH["screen"] and (
                        self._ask_crop(clip, iv, scene["laptop"], "lid_open", margin=0.4) < 0.5 or
                        self._ask_crop(clip, iv, scene["laptop"], "lock_screen", margin=0.4) >= 0.5):
                    ps = 0.0   # closed lid or lock screen: not working content
                if ps >= TH["screen"]:
                    out.append(Finding("T02_PC_UNLOCKED_UNATTENDED", t_leave, ps,
                                       "На видео работник отходит от рабочего места, экран компьютера продолжает "
                                       "показывать рабочее содержимое (таблицу/документ), а не экран блокировки",
                                       [(leave_idx, None), (iv, scene["laptop"])]))
            if "safe" in scene:
                po = self._ask_crop(clip, iv, scene["safe"], "safe_open", margin=0.3)
                if po >= TH["safe_open"] and e - s >= 4:
                    # still open at the end of the absence, with nobody at the safe -> left open
                    pe = self._ask_crop(clip, e, scene["safe"], "safe_open", margin=0.3)
                    if pe >= TH["safe_open"] and not tl["beside_safe"][e]:
                        out.append(Finding("T03_SAFE_LEFT_OPEN", t_leave, min(po, pe),
                                           "На видео работник отходит от рабочего места, дверца сейфа остаётся открытой",
                                           [(leave_idx, None), (e, scene["safe"])]))
                # T04: key left in the safe lock (the key is only a few pixels: low recall, see limitations)
                pk = self._ask_crop(clip, iv, scene["safe"], "key", margin=0.15)
                if pk >= TH["key"]:
                    out.append(Finding("T04_KEY_LEFT_IN_SAFE", t_leave, pk * 0.8,
                                       "На видео работник отходит от рабочего места; в замке сейфа виден ключ",
                                       [(leave_idx, None), (iv, scene["safe"])]))
        return out

    def _check_changeover(self, clip, people, scene, tl, budget):
        """Someone left, then a new person enters while cash lies openly on the desk."""
        out = []
        if "desk" not in scene:
            return out
        exits = [i for i, _, _ in self._count_events(tl, up=False)]
        for i, before, after in self._count_events(tl, up=True):
            if not budget.ok(4) or after < 2:
                continue
            if not any(0 < clip.times[i] - clip.times[j] < 40 for j in exits if j < i):
                continue
            p, j = 0.0, i
            for dk in (2, 4, 6):              # a hand may cover the stack at any single instant
                jj = self._idx(clip, i + dk)
                pj = self._money(clip, jj, scene["desk"])
                if pj > p:
                    p, j = pj, jj
                if p >= TH["money"]:
                    break
            if p >= TH["money"]:
                out.append(Finding("T05_CASH_EXPOSED_NEXT_CLIENT", clip.times[i], p,
                                   "На видео после ухода предыдущего посетителя на столе открыто лежат банкноты, "
                                   "в это время в помещение входит и подходит к столу другой человек",
                                   [(i, None), (j, scene["desk"])]))
        return out

    def _check_entries(self, clip, people, scene, tl, budget):
        """Entries: collector without document (T09), extra person beside the worker (T06), door left open (T07)."""
        out = []
        for i, before, after in self._count_events(tl, up=True):
            if not budget.ok(6):
                break
            # T06: two different visitors at the desk at the same time (a new one arrives while the first
            # is still being served), confirmed twice ~1.5 s apart
            if after >= 3:
                k, k2 = self._idx(clip, i + 3), self._idx(clip, i + 6)
                p1 = self._ask_full(clip, k, "two_visitors")
                if p1 >= TH["two_visitors"]:
                    p2 = self._ask_full(clip, k2, "two_visitors")
                    if p2 >= TH["two_visitors"]:
                        out.append(Finding("T06_TWO_CLIENTS_AT_ONCE", clip.times[k], min(p1, p2),
                                           "На видео у рабочего стола одновременно находятся два посетителя, работник "
                                           "взаимодействует с обоими", [(k, None), (k2, None)]))
            # T07: room was empty, someone enters and walks on without turning back to the door.
            # Closing the door is visible as the person returning to the door zone right after entering
            # (relative motion, camera-angle independent); the glass-door state itself is ambiguous.
            if before == 0 and "door" in scene:
                door = scene["door"]
                horizon = min(len(people), i + int(10 * config.SAMPLE_FPS))
                near = [any(self._near(p["box"], door, 0.35, 0.25) for p in people[k]) for k in range(i, horizon)]
                left = next((k for k, f in enumerate(near) if not f), None)
                if left is not None and horizon - i >= int(6 * config.SAMPLE_FPS):
                    returned = any(near[left + 1:])
                    if not returned:
                        pd_ = self._ask_crop(clip, self._idx(clip, i + left + 4), door, "door_open", margin=0.05)
                        if pd_ >= 0.5:
                            t_pass = clip.times[self._idx(clip, i + left)]
                            out.append(Finding("T07_CASH_ROOM_DOOR_NOT_CLOSED", t_pass, 0.5 + 0.3 * pd_,
                                               "На видео человек входит в пустое помещение и сразу проходит к рабочему "
                                               "месту, не возвращаясь к двери, чтобы закрыть её; дверь после входа "
                                               "выглядит открытой",
                                               [(self._idx(clip, i + left), None),
                                                (self._idx(clip, i + left + 4), door)]))
            elif "door" in scene and budget.ok(5):
                # Occupied rooms also need a door-state check after an entry.
                a, b = self._idx(clip, i + int(3*config.SAMPLE_FPS)), self._idx(clip, i + int(5*config.SAMPLE_FPS))
                pa = self._ask_crop(clip, a, scene["door"], "door_open", margin=.05)
                pb = self._ask_crop(clip, b, scene["door"], "door_open", margin=.05)
                if min(pa,pb) >= .9 and b > a:
                    out.append(Finding("T07_CASH_ROOM_DOOR_NOT_CLOSED", clip.times[a], min(pa,pb),
                        "После входа человека дверь кассового помещения выглядит открытой на двух последовательных проверках",
                        [(a,scene["door"]),(b,scene["door"])]))
        return out

    def _check_service(self, clip, people, scene, tl, budget):
        """T12: a visitor is served (hands over money / a bag) without any visible ID document.
        Service episodes = >= 5 s with someone besides the worker in the room (no reliance on door
        entries); the first ~20 s of each episode are scanned every second, stopping once an ID is seen."""
        out = []
        min_len = int(5 * config.SAMPLE_FPS)
        segs = self._segments(tl["n"] >= 2, True, min_len=min_len)
        segs = sorted(segs, key=lambda g: g[1] - g[0], reverse=True)[:3]
        step = max(1, int(round(config.SAMPLE_FPS)))
        for s, e in sorted(segs):
            end = min(e + 1, s + int(20 * config.SAMPLE_FPS))
            id_max, hand_at = 0.0, None
            inspected = []
            for k in range(s, end, step):
                if not budget.ok(6):
                    break
                id_max = max(id_max, self._ask_full(clip, k, "id_shown"))
                inspected.append(k)
                if id_max >= TH["id_shown"]:
                    break
                if hand_at is None and (k - s) % (2 * step) == 0 and \
                        self._ask_full(clip, k, "handover") >= TH["handover"]:
                    hand_at = k
            # the episode must have been scanned for a while (>= 8 s) before claiming "no ID seen"
            scanned = len(inspected) >= 8 and clip.times[inspected[-1]] - clip.times[inspected[0]] >= 7
            # a visitor must really be there (the person count can include a stray track)
            mid = self._idx(clip, (s + min(end, e)) // 2)
            if id_max < TH["id_shown"] and scanned and budget.ok(4) and \
                    self._ask_full(clip, mid, "visitor_seated") >= TH["visitor_seated"]:
                k = hand_at if hand_at is not None else self._idx(clip, s + int(6 * config.SAMPLE_FPS))
                out.append(Finding("T12_ID_NOT_CHECKED", clip.times[k], 0.5 + 0.4 * (1 - id_max),
                                   "На видео посетитель обслуживается у рабочего стола; предъявления или проверки "
                                   "удостоверения личности в кадрах обслуживания не видно",
                                   [(k, None), (self._idx(clip, k + 2 * step), None)]))
        return out

    def _check_periodic(self, clip, people, scene, tl, budget):
        """Sparse scan: money mixing, personal items, drinking, safe open while nobody is at the safe."""
        out = []
        # T13 (money mixing) is not detected: the VLM could not separate it (X-44 0.007 vs normal X-45 0.23)
        step = max(1, int(round(4.0 * config.SAMPLE_FPS)))   # every ~4 s
        runs = []
        for i in range(0, len(clip.frames), step):
            if not budget.ok(3):
                break
            if tl["n"][i] >= 1 and self._ask_full(clip, i, "drinking") >= TH["drinking"]:
                p1 = self.trace[-1][2]
                j = self._idx(clip, i + 2)    # confirm ~1 s later: a hand near the face is not drinking
                p2 = self._ask_full(clip, j, "drinking")
                cup = self._ask_full(clip, i, "cup_in_hand")
                paper=0.
                if p2>=TH['drinking'] and cup>=.7 and 'desk' in scene:
                    paper=max(self._ask_crop(clip,k,scene['desk'],'banknotes_in_hand',margin=.3) for k in (i,j,self._idx(clip,i+4)))
                    if paper>=.8:
                        self.ambiguities.append(f'{clip.times[i]:.1f} с: предмет в руках похож и на банкноты, и на чашку; питьё не подтверждено')
                if p2 >= TH["drinking"] and cup >= .7 and paper < .8:
                    out.append(Finding("T11_FOOD_DRINK_AT_DESK", clip.times[i], min(p1, p2, cup),
                                       "На видео работник за рабочим столом пьёт из чашки", [(i, None), (j, None)]))
            if self._ask_full(clip, i, "personal") >= TH["personal"]:
                p1 = self.trace[-1][2]
                if "desk" not in scene or self._ask_crop(clip, i, scene["desk"], "bag") < TH["bag"]:
                    out.append(Finding("T10_PERSONAL_ITEMS_AT_DESK", clip.times[i], p1,
                                       "На видео на рабочем столе лежат верхняя одежда и/или сумка", [(i, None)]))
            if "safe" in scene:
                po = self._ask_crop(clip, i, scene["safe"], "safe_open", margin=0.3)
                unattended = False
                if po >= TH["safe_open"] and not tl["near_safe"][i]:
                    unattended = self._ask_full(clip, i, "at_safe") < TH["at_safe"]
                runs.append((i, unattended, po))
        # Open with nobody at it on two consecutive scans -> verify duration every ~2 s.
        # Normal use (take/put something from the chair) closes within ~6 s; flag >= 9 s,
        # or still open at the end of the clip after >= 5 s.
        dense = max(1, int(round(2.0 * config.SAMPLE_FPS)))
        done_until = -1
        for (i1, f1, p1), (i2, f2, p2) in zip(runs, runs[1:]):
            if not (f1 and f2) or i1 <= done_until or not budget.ok(4):
                continue
            last_open, k, at_end = i2, i2 + dense, False
            while k < len(clip.frames) and budget.ok(3):
                if self._ask_crop(clip, k, scene["safe"], "safe_open", margin=0.3) < TH["safe_open"]:
                    break
                last_open, k = k, k + dense
            else:
                at_end = k >= len(clip.frames)
            done_until = last_open
            dur = clip.times[last_open] - clip.times[i1]
            if dur >= 9.0 or (at_end and dur >= 5.0):
                out.append(Finding("T03_SAFE_LEFT_OPEN", clip.times[i1], min(p1, p2),
                                   f"На видео дверца сейфа остаётся открытой не менее {dur:.0f} с, при этом у сейфа "
                                   f"никого нет", [(i1, scene["safe"]), (last_open, scene["safe"])]))
        return out

    # ------------------------------------------------------------- aggregate
    @staticmethod
    def _merge(findings, window=12.0):
        """One finding per type per ~12 s window (keep the most confident, earliest timecode)."""
        out = []
        for f in sorted(findings, key=lambda f: (f.type_id, f.t)):
            if out and out[-1].type_id == f.type_id and f.t - out[-1].t < window:
                if f.conf > out[-1].conf:
                    f.t = min(f.t, out[-1].t)
                    out[-1] = f
                continue
            out.append(f)
        return sorted(out, key=lambda f: f.t)
