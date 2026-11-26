"""Profile the Track B labels: integrity checks, value sets, null structure, constraints, priors.

Usage:
    python -I tools/analyze_labels.py --data data --out work
Writes work/label_profile.txt and work/priors.json (majority non-null value per feature,
the fallback answers for the reliability shell).
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

TRACK_B_FEATURES = (
    "driver_gender", "driver_tshirt_color", "driver_emotion", "driver_phone", "driver_safety_belt",
    "passenger_codriver", "passenger_codriver_gender", "passenger_codriver_tshirt_color",
    "passenger_codriver_emotion", "codriver_safety_belt",
    "passenger_rear_seat_left", "passenger_rear_left_gender", "passenger_rear_left_tshirt_color",
    "passenger_rear_left_emotion", "passenger_rear_left_safety_belt",
    "passenger_rear_seat_right", "passenger_rear_right_gender", "passenger_rear_right_tshirt_color",
    "passenger_rear_right_emotion", "passenger_rear_right_safety_belt",
    "phone_codriver_seat", "phone_codriver_seat_color",
    "colabottle_codriver_seat", "colacan_codriver_seat",
    "suitcase", "suitcase_color", "suitcase_location", "suitcase_pose",
    "baby_seat", "baby_seat_orientation", "baby",
)
ALLOWED = {
    "gender": {"MALE", "FEMALE"}, "tshirt_color": {"BLACK", "WHITE"}, "emotion": {"HAPPY", "SERIOUS"},
    "phone_codriver_seat_color": {"BLACK", "WHITE"}, "suitcase_color": {"ANTRACITE", "YELLOW"},
    "suitcase_location": {"CO_DRIVER_SEAT", "REAR_SEAT"}, "suitcase_pose": {"UPWARDS", "FLAT"},
    "baby_seat_orientation": {"FRONT_FACING", "REAR_FACING"},
}
# child -> parent: the child should be non-null exactly when the parent is YES
PARENT = {}
for seat, flag in (("passenger_codriver", "passenger_codriver"),
                   ("passenger_rear_left", "passenger_rear_seat_left"),
                   ("passenger_rear_right", "passenger_rear_seat_right")):
    for attr in ("gender", "tshirt_color", "emotion"):
        PARENT[f"{seat}_{attr}"] = flag
PARENT.update({
    "codriver_safety_belt": "passenger_codriver",
    "passenger_rear_left_safety_belt": "passenger_rear_seat_left",
    "passenger_rear_right_safety_belt": "passenger_rear_seat_right",
    "phone_codriver_seat_color": "phone_codriver_seat",
    "suitcase_color": "suitcase", "suitcase_location": "suitcase", "suitcase_pose": "suitcase",
    "baby_seat_orientation": "baby_seat", "baby": "baby_seat",
})


def allowed(feature):
    for key, values in ALLOWED.items():
        if feature.endswith(key):
            return values
    return {"YES", "NO"}


def norm(v):
    return None if v is None else str(v).strip().upper()


def load(labels_dir: Path):
    out = {}
    for p in sorted(labels_dir.glob("*.json")):
        out[p.stem] = json.loads(p.read_text())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    main_labels = load(args.data / "main" / "labels")
    ext_labels = load(args.data / "extended" / "labels")
    num = lambda s: int(re.search(r"(\d+)$", s).group(1))
    groups = {
        "twins_00000-00029": {k: v for k, v in main_labels.items() if num(k) < 30},
        "main_00030-00059": {k: v for k, v in main_labels.items() if 30 <= num(k) < 60},
        "main_00060+": {k: v for k, v in main_labels.items() if num(k) >= 60},
        "extended": ext_labels,
    }
    lines = []
    say = lines.append
    say(f"scenes: main={len(main_labels)} extended={len(ext_labels)}")

    # 1. Integrity: environment and occupancy of 00030-00059 (the withheld photos' stems).
    say("\n== environment by group ==")
    for g, labs in groups.items():
        env = Counter(str(l.get("env")) for l in labs.values())
        driver_only = sum(1 for l in labs.values() if all(norm(l.get(f)) == "NO" for f in
                          ("passenger_codriver", "passenger_rear_seat_left", "passenger_rear_seat_right")))
        say(f"{g:20} n={len(labs):5}  env={dict(env.most_common())}  driver_only={driver_only}")

    # 2. Value sets, nulls, out-of-vocabulary values.
    say("\n== value counts per feature (twins | main_00030+ | extended) ==")
    bad = []
    for f in TRACK_B_FEATURES:
        row = []
        for g in ("twins_00000-00029", "main_00030-00059", "main_00060+", "extended"):
            c = Counter(norm(l.get(f, "<MISSING>")) for l in groups[g].values())
            row.append(dict(sorted(c.items(), key=lambda kv: str(kv[0]))))
            for v in c:
                if v is not None and v not in allowed(f):
                    bad.append((g, f, v, c[v]))
        say(f"{f:34} twins={row[0]}\n{'':34} m30-59={row[1]}\n{'':34} m60+={row[2]}\n{'':34} ext={row[3]}")
    say(f"\nout-of-vocabulary values: {bad if bad else 'none'}")

    # 3. Null structure: child non-null <=> parent YES.
    say("\n== parent/child null consistency ==")
    all_labs = {**{f"m:{k}": v for k, v in main_labels.items()}, **{f"e:{k}": v for k, v in ext_labels.items()}}
    for child, parent in PARENT.items():
        viol = Counter()
        for l in all_labs.values():
            p, c = norm(l.get(parent)), norm(l.get(child))
            if p == "YES" and c is None:
                viol["parent YES, child null"] += 1
            elif p != "YES" and c is not None:
                viol[f"parent {p}, child set"] += 1
        say(f"{child:34} <- {parent:26} {dict(viol) if viol else 'consistent'}")
    for f in ("driver_gender", "driver_tshirt_color", "driver_emotion", "driver_phone", "driver_safety_belt",
              "passenger_codriver", "passenger_rear_seat_left", "passenger_rear_seat_right",
              "phone_codriver_seat", "colabottle_codriver_seat", "colacan_codriver_seat", "suitcase", "baby_seat"):
        nulls = sum(1 for l in all_labs.values() if norm(l.get(f)) is None)
        if nulls:
            say(f"top-level feature {f} is null in {nulls} scenes")

    # 4. Co-driver seat co-occurrence (what can share that seat?).
    say("\n== co-driver seat co-occurrence (all scenes) ==")
    combo = Counter()
    for l in all_labs.values():
        key = (f"pax={norm(l.get('passenger_codriver'))}",
               f"phone={norm(l.get('phone_codriver_seat'))}",
               f"bottle={norm(l.get('colabottle_codriver_seat'))}",
               f"can={norm(l.get('colacan_codriver_seat'))}",
               f"suitcase@{norm(l.get('suitcase_location'))}",
               f"baby_seat={norm(l.get('baby_seat'))}")
        combo[key] += 1
    pax_with = defaultdict(Counter)
    for l in all_labs.values():
        pax = norm(l.get("passenger_codriver"))
        for f in ("phone_codriver_seat", "colabottle_codriver_seat", "colacan_codriver_seat", "baby_seat"):
            pax_with[f][(pax, norm(l.get(f)))] += 1
        pax_with["suitcase_location"][(pax, norm(l.get("suitcase_location")))] += 1
        pax_with["baby_seat_vs_suitcase_loc"][(norm(l.get("baby_seat")), norm(l.get("suitcase_location")))] += 1
        pax_with["rear_pax_vs_baby_seat"][((norm(l.get("passenger_rear_seat_left")), norm(l.get("passenger_rear_seat_right"))), norm(l.get("baby_seat")))] += 1
    for f, c in pax_with.items():
        say(f"{f:28} {dict(sorted(c.items(), key=str))}")

    # 5. Priors: majority non-null value per feature over every labelled scene.
    priors, margins = {}, {}
    for f in TRACK_B_FEATURES:
        c = Counter(norm(l.get(f)) for l in all_labs.values() if norm(l.get(f)) is not None)
        (top, n_top), total = c.most_common(1)[0], sum(c.values())
        priors[f] = top
        margins[f] = round(n_top / total, 4)
    (args.out / "priors.json").write_text(json.dumps(priors, indent=2))
    say("\n== priors (majority non-null value, share) ==")
    for f in TRACK_B_FEATURES:
        say(f"{f:34} {priors[f]:15} {margins[f]}")
    exp = sum(margins[f] * sum(1 for l in all_labs.values() if norm(l.get(f)) is not None) for f in TRACK_B_FEATURES)
    tot = sum(1 for l in all_labs.values() for f in TRACK_B_FEATURES if norm(l.get(f)) is not None)
    say(f"\npriors-only accuracy on all labelled features: {exp / tot:.4f} ({tot} features)")

    (args.out / "label_profile.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
