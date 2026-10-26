#!/usr/bin/env python3
"""Generate 1000 training examples for a helpful, honest, cautious small model.

Every numeric, date, unit, logic and code answer is COMPUTED here (not typed by
hand), so the dataset cannot contain arithmetic slips. Each example:
  - decomposes the problem into steps
  - verifies the claim with a check that uses only the prompt or stated knowledge
  - flags uncertainty honestly
  - refuses to guess or fabricate when the information is not available
  - shows learning from a previous mistake (the "mistakes" category)

Output: cautious_assistant_1000.jsonl  (one {"user":..., "assistant":...} per line)
Run:    python3 training_data/generate_cautious_dataset.py
"""
import contextlib
import datetime as dt
import io
import itertools
import json
import os
import random
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

random.seed(1014)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cautious_assistant_1000.jsonl")

SEEN = set()


def n(x):
    """Format numbers readably."""
    if isinstance(x, int):
        return f"{x:,}"
    return f"{x:,.2f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)


def usd(d):
    q = Decimal(d).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"${q:,}"


def collect(gen, target, label):
    """Call gen() until `target` unique (by user text) examples exist."""
    out, tries = [], 0
    while len(out) < target:
        tries += 1
        if tries > target * 400:
            raise RuntimeError(f"{label}: could not produce {target} unique examples (got {len(out)})")
        item = gen()
        if item is None:
            continue
        u, a = item
        if u in SEEN:
            continue
        SEEN.add(u)
        out.append({"user": u, "assistant": a})
    return out


def collect_pool(pool, target, label):
    """Draw unique examples from a pre-built list of (user, assistant)."""
    random.shuffle(pool)
    out = []
    for u, a in pool:
        if u in SEEN:
            continue
        SEEN.add(u)
        out.append({"user": u, "assistant": a})
        if len(out) == target:
            return out
    raise RuntimeError(f"{label}: pool has only {len(out)} unique, need {target}")


# ---------------------------------------------------------------- arithmetic
ITEMS = ["bolts", "notebooks", "bottles", "tiles", "batteries", "envelopes", "candles", "bricks", "mugs", "cables"]
PEOPLE = ["Maya", "Omar", "Priya", "Lena", "Diego", "Aiko", "Tomas", "Ngozi", "Sven", "Rosa", "Kofi", "Hana"]


def g_boxes():
    a, b = random.randint(3, 25), random.randint(4, 48)
    total = a * b
    c = random.randint(1, total - 1)
    it = random.choice(ITEMS)
    ans = total - c
    u = f"A warehouse has {a} crates with {b} {it} in each crate. If {n(c)} {it} are shipped out, how many are left?"
    s = (f"I'll break this into steps.\n"
         f"Step 1: total {it} = {a} × {b} = {n(total)}.\n"
         f"Step 2: subtract the shipped ones: {n(total)} − {n(c)} = {n(ans)}.\n"
         f"Verification: {n(ans)} + {n(c)} = {n(ans + c)}, which equals the starting total of {n(total)}.\n"
         f"Answer: {n(ans)} {it} are left. This assumes every crate held exactly {b} and nothing was lost or damaged, "
         f"since the question doesn't say otherwise.")
    return u, s


def g_distance():
    sp = random.choice(range(30, 121, 5))
    t = random.choice([1.5, 2, 2.5, 3, 3.5, 4, 5, 6])
    d = sp * t
    who = random.choice(PEOPLE)
    u = f"{who} drives at a steady {sp} km/h for {t:g} hours. How far does {who} travel?"
    s = (f"Step 1: use distance = speed × time.\n"
         f"Step 2: {sp} × {t:g} = {d:g} km.\n"
         f"Verification: divide back: {d:g} ÷ {sp} = {d / sp:g} hours, which matches.\n"
         f"Answer: {d:g} km. This assumes the speed really is constant; with traffic or stops the real distance would differ.")
    return u, s


def g_avg_speed():
    v1 = random.choice([30, 40, 50, 60, 80])
    v2 = random.choice([20, 40, 60, 90, 120])
    if v1 == v2:
        return None
    d = random.choice([60, 120, 240])
    t1, t2 = Fraction(d, v1), Fraction(d, v2)
    avg = Fraction(2 * d) / (t1 + t2)
    naive = (v1 + v2) / 2
    u = f"I drive {d} km out at {v1} km/h and the same {d} km back at {v2} km/h. What's my average speed for the whole trip?"
    s = (f"The tempting answer is the simple mean, {naive:g} km/h, but that's wrong because I spend different amounts of time at each speed.\n"
         f"Step 1: time out = {d} ÷ {v1} = {float(t1):.4f} h; time back = {d} ÷ {v2} = {float(t2):.4f} h.\n"
         f"Step 2: total distance = {2 * d} km; total time = {float(t1 + t2):.4f} h.\n"
         f"Step 3: average speed = total distance ÷ total time = {float(avg):.2f} km/h.\n"
         f"Verification: {float(avg):.2f} lies between {min(v1, v2)} and {max(v1, v2)} and is closer to the slower speed, "
         f"which is what we expect because more time is spent going slowly.\n"
         f"Answer: about {float(avg):.2f} km/h (this assumes constant speeds and no stops).")
    return u, s


def g_workers():
    nw, days = random.randint(2, 12), random.randint(2, 20)
    total = nw * days
    divs = [m for m in range(1, 25) if total % m == 0 and m != nw]
    if not divs:
        return None
    m = random.choice(divs)
    ans = total // m
    u = f"{nw} workers can finish a job in {days} days. How many days would {m} workers need, working at the same rate?"
    s = (f"Step 1: measure the job in worker-days: {nw} × {days} = {total} worker-days.\n"
         f"Step 2: with {m} workers, days = {total} ÷ {m} = {ans}.\n"
         f"Verification: {m} × {ans} = {m * ans} worker-days, equal to {total}.\n"
         f"Answer: {ans} days. Caveat: this assumes all workers are equally productive and the work splits perfectly; "
         f"real jobs often have coordination overhead, so treat it as an idealised estimate.")
    return u, s


def g_mean_target():
    k = random.randint(3, 5)
    scores = [random.randint(55, 98) for _ in range(k)]
    target = random.choice([80, 85, 90])
    need = target * (k + 1) - sum(scores)
    sc = ", ".join(map(str, scores))
    u = f"My test scores are {sc}. What do I need on the next test to average {target}?"
    head = (f"Step 1: sum so far = {sum(scores)} over {k} tests.\n"
            f"Step 2: to average {target} over {k + 1} tests I need a total of {target} × {k + 1} = {target * (k + 1)}.\n"
            f"Step 3: needed next score = {target * (k + 1)} − {sum(scores)} = {need}.\n")
    if need > 100:
        tail = (f"Verification and caveat: a {need} is above 100, so if 100 is the maximum score this is impossible. "
                f"Even a 100 would give an average of {(sum(scores) + 100) / (k + 1):.2f}. "
                f"I'm not going to pretend it's reachable; unless extra credit exists, you'd need a lower target or more tests.")
    elif need <= 0:
        tail = (f"Verification: even a 0 gives an average of {sum(scores) / (k + 1):.2f}, which is already at or above {target}. "
                f"So any score works.")
    else:
        tail = (f"Verification: ({sum(scores)} + {need}) ÷ {k + 1} = {(sum(scores) + need) / (k + 1):g}, which hits the target.\n"
                f"Answer: you need {need} on the next test.")
    return u, head + tail


def g_change():
    q, p = random.randint(2, 9), random.choice(range(125, 1250, 25))
    total = q * p
    bill = next(b for b in (2000, 5000, 10000, 20000) if b >= total)
    change = bill - total
    it = random.choice(["sandwiches", "notebooks", "pens", "coffees", "tickets", "magazines"])
    u = f"{q} {it} cost ${p / 100:.2f} each. If I pay with a ${bill // 100} bill, how much change do I get?"
    s = (f"Step 1: cost = {q} × ${p / 100:.2f} = ${total / 100:.2f}.\n"
         f"Step 2: change = ${bill / 100:.2f} − ${total / 100:.2f} = ${change / 100:.2f}.\n"
         f"Verification: ${change / 100:.2f} + ${total / 100:.2f} = ${(change + total) / 100:.2f}, equal to the bill I paid with.\n"
         f"Answer: ${change / 100:.2f}. I've assumed no tax or discounts, since none were mentioned.")
    return u, s


def g_recipe():
    base = Fraction(random.choice([1, 2, 3, 3, 5]), random.choice([2, 3, 4, 8]))
    if base > 3:
        return None
    n0, n1 = random.choice([12, 6, 8, 24]), random.choice([18, 30, 36, 15, 40, 9])
    if n0 == n1:
        return None
    res = base * Fraction(n1, n0)
    unit = random.choice(["cups of flour", "cups of sugar", "teaspoons of vanilla", "cups of milk"])
    u = f"A recipe needs {base} {unit} for {n0} servings. How much do I need for {n1} servings?"
    s = (f"Step 1: scale factor = {n1} ÷ {n0} = {Fraction(n1, n0)} (≈ {n1 / n0:.3f}).\n"
         f"Step 2: new amount = {base} × {Fraction(n1, n0)} = {res} (≈ {float(res):.3f}) {unit}.\n"
         f"Verification: dividing back, {res} ÷ {Fraction(n1, n0)} = {res / Fraction(n1, n0)}, the original amount.\n"
         f"Answer: {res} {unit} (about {float(res):.2f}). Note: baking doesn't always scale linearly (leavening, pan size, "
         f"cooking time), so I'm only certain about the arithmetic, not the final taste.")
    return u, s


# --------------------------------------------------------------------- units
PAIRS = [
    ("miles", "kilometers", 1.609344, "1 mile = 1.609344 km (exact by definition)"),
    ("pounds", "kilograms", 0.45359237, "1 pound = 0.45359237 kg (exact by definition)"),
    ("inches", "centimeters", 2.54, "1 inch = 2.54 cm (exact by definition)"),
    ("feet", "meters", 0.3048, "1 foot = 0.3048 m (exact by definition)"),
    ("US gallons", "liters", 3.785411784, "1 US gallon = 3.785411784 L (exact by definition)"),
    ("US fluid ounces", "milliliters", 29.5735295625, "1 US fluid ounce = 29.5735295625 mL (exact, 1/128 of a US gallon)"),
]


def g_unit():
    if random.random() < 0.22:
        return g_temp()
    a, b, f, note = random.choice(PAIRS)
    x = random.choice([random.randint(2, 500), random.randint(2, 300) + 0.5])
    if random.random() < 0.5:
        res, src, dst, fac = x * f, a, b, f
        step = f"{x:g} × {f:g}"
        back = res / f
    else:
        res, src, dst, fac = x / f, b, a, 1 / f
        step = f"{x:g} ÷ {f:g}"
        back = res * f
    r = round(res, 2)
    u = random.choice([f"Convert {x:g} {src} to {dst}.", f"How many {dst} is {x:g} {src}?", f"What is {x:g} {src} in {dst}?"])
    s = (f"Step 1: recall the defining relationship: {note}.\n"
         f"Step 2: compute {step} = {res:.4f}, which rounds to {r:g} {dst}.\n"
         f"Verification: converting {r:g} {dst} back gives about {back:.2f} {src}, matching {x:g} within rounding.\n"
         f"Answer: about {r:g} {dst}. The factor is exact; only the rounding to two decimals loses precision.")
    return u, s


def g_temp():
    if random.random() < 0.5:
        c = random.choice([random.randint(-40, 45), random.randint(-20, 40) + 0.5])
        f = c * 9 / 5 + 32
        u = random.choice([f"Convert {c:g}°C to Fahrenheit.", f"What is {c:g} degrees Celsius in °F?"])
        s = (f"Step 1: formula °F = °C × 9/5 + 32.\n"
             f"Step 2: {c:g} × 9/5 = {c * 9 / 5:g}; adding 32 gives {f:g}°F.\n"
             f"Verification with anchor points: 0°C = 32°F and 100°C = 212°F. {c:g}°C sits "
             f"{'between' if 0 <= c <= 100 else 'outside'} those anchors, and {f:g}°F is "
             f"{'between 32 and 212' if 0 <= c <= 100 else 'outside the 32–212 range'}, consistent. "
             f"(Also −40 is the same on both scales.)\n"
             f"Answer: {f:g}°F.")
    else:
        f = random.choice([random.randint(-20, 110), random.randint(0, 100) + 0.5])
        c = (f - 32) * 5 / 9
        u = random.choice([f"Convert {f:g}°F to Celsius.", f"What is {f:g} degrees Fahrenheit in °C?"])
        s = (f"Step 1: formula °C = (°F − 32) × 5/9.\n"
             f"Step 2: {f:g} − 32 = {f - 32:g}; × 5/9 = {c:.2f}°C.\n"
             f"Verification: converting back, {c:.2f} × 9/5 + 32 = {c * 9 / 5 + 32:.2f}°F, matching {f:g}.\n"
             f"Answer: about {c:.1f}°C (rounded to one decimal).")
    return u, s


# ------------------------------------------------------------ percent/finance
def g_discount():
    price = random.choice([20, 35, 48, 60, 75, 120, 199, 250, 400])
    d = random.choice([10, 15, 20, 25, 30, 40, 50])
    tax = random.choice([0, 5, 7, 8, 10])
    p = Decimal(price)
    disc = p * (100 - d) / 100
    fin = disc * (100 + tax) / 100
    if tax:
        u = f"An item costs {usd(p)} with {d}% off, then {tax}% sales tax is added to the sale price. What do I pay?"
        s = (f"Step 1: discounted price = {usd(p)} × (1 − {d / 100:g}) = {usd(disc)}.\n"
             f"Step 2: add tax on the discounted price: {usd(disc)} × {1 + tax / 100:g} = {usd(fin)}.\n"
             f"Verification: discount amount {usd(p - disc)} + price paid before tax {usd(disc)} = {usd(p)}; "
             f"tax amount {usd(fin - disc)} is {tax}% of {usd(disc)}.\n"
             f"Answer: {usd(fin)}. I assumed tax applies after the discount, as stated; some stores compute it differently, so check the receipt.")
    else:
        u = f"A jacket is {usd(p)} and is {d}% off. What's the sale price?"
        s = (f"Step 1: discount = {usd(p)} × {d}% = {usd(p - disc)}.\n"
             f"Step 2: sale price = {usd(p)} − {usd(p - disc)} = {usd(disc)}.\n"
             f"Verification: {usd(disc)} ÷ {usd(p)} = {float(disc / p):.2f}, i.e. {100 - d}% of the original, as expected.\n"
             f"Answer: {usd(disc)}.")
    return u, s


def g_pct_change():
    old = random.randint(20, 400)
    new = random.randint(20, 400)
    if old == new:
        return None
    ch = (new - old) / old * 100
    rev = (old - new) / new * 100
    word = "increase" if new > old else "decrease"
    u = f"A value goes from {old} to {new}. What is the percent change?"
    s = (f"Step 1: change = {new} − {old} = {new - old}.\n"
         f"Step 2: percent change = change ÷ original = {new - old} ÷ {old} = {ch:.2f}%, a {word}.\n"
         f"Verification: {old} × (1 + {ch / 100:.4f}) = {old * (1 + ch / 100):.2f}, which recovers {new}.\n"
         f"Answer: {abs(ch):.2f}% {word}. Note the asymmetry: going back from {new} to {old} is a "
         f"{abs(rev):.2f}% {'decrease' if new > old else 'increase'}, not {abs(ch):.2f}%, because the base differs.")
    return u, s


def g_compound():
    P = random.choice([1000, 2500, 5000, 10000])
    r = random.choice([3, 4, 5, 6, 7])
    t = random.randint(2, 30)
    val = Decimal(P) * (Decimal(1) + Decimal(r) / 100) ** t
    simple = Decimal(P) * (1 + Decimal(r) / 100 * t)
    u = f"If I invest {usd(P)} at {r}% annual interest compounded yearly, how much will I have after {t} years?"
    s = (f"Step 1: formula A = P × (1 + r)^t with P = {P}, r = {r / 100:g}, t = {t}.\n"
         f"Step 2: (1 + {r / 100:g})^{t} = {float((1 + Decimal(r) / 100) ** t):.4f}; times {P} = {usd(val)}.\n"
         f"Verification: simple (non-compounded) interest would give {usd(simple)}; compounding should be larger, and {usd(val)} is.\n"
         f"Answer: about {usd(val)}. This is arithmetic only: it assumes a fixed rate, no fees, no taxes and no withdrawals. "
         f"Real returns are not guaranteed, and I can't predict what any real investment will earn.")
    return u, s


def g_tip():
    bill = Decimal(random.choice([32.50, 48.00, 75.20, 120.00, 86.40, 64.80, 150.00]))
    tip = random.choice([15, 18, 20])
    ppl = random.choice([2, 3, 4, 5, 6])
    total = bill * (100 + tip) / 100
    each = total / ppl
    u = f"Our bill is {usd(bill)}. We want to tip {tip}% and split it evenly among {ppl} people. How much does each person pay?"
    s = (f"Step 1: tip = {usd(bill)} × {tip}% = {usd(bill * tip / 100)}.\n"
         f"Step 2: total = {usd(bill)} + {usd(bill * tip / 100)} = {usd(total)}.\n"
         f"Step 3: each = {usd(total)} ÷ {ppl} = {usd(each)}.\n"
         f"Verification: {ppl} × {usd(each)} = {usd(each * ppl)} versus a total of {usd(total)}; "
         f"any gap of a cent or two is just rounding to whole cents.\n"
         f"Answer: about {usd(each)} each, assuming the tip is on the pre-tax bill shown.")
    return u, s


def g_what_pct():
    b = random.choice([40, 50, 80, 120, 150, 200, 250, 360])
    a = random.randint(1, b - 1)
    u = f"What percent of {b} is {a}?"
    s = (f"Step 1: percent = part ÷ whole × 100 = {a} ÷ {b} × 100.\n"
         f"Step 2: {a} ÷ {b} = {a / b:.4f}, so the percent is {a / b * 100:.2f}%.\n"
         f"Verification: {a / b * 100:.2f}% of {b} = {a / b * 100 / 100 * b:.2f}, which returns {a}.\n"
         f"Answer: {a / b * 100:.2f}%.")
    return u, s


# --------------------------------------------------------------------- dates
ANCHOR = dt.date(2000, 1, 1)  # a Saturday
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def fmt(d):
    return d.strftime("%B %-d, %Y")


def rand_date(lo=1995, hi=2040):
    s, e = dt.date(lo, 1, 1), dt.date(hi, 12, 31)
    return s + dt.timedelta(days=random.randint(0, (e - s).days))


def g_weekday():
    d = rand_date()
    D = (d - ANCHOR).days
    r = D % 7
    wd = DAYS[(5 + r) % 7]
    assert wd == d.strftime("%A")
    u = random.choice([f"What day of the week is {fmt(d)}?", f"What weekday falls on {fmt(d)}?"])
    s = (f"Step 1: use a known anchor: January 1, 2000 was a Saturday.\n"
         f"Step 2: days from the anchor to {fmt(d)} = {D:,} (negative would mean earlier).\n"
         f"Step 3: {D:,} mod 7 = {r}, so move {r} day(s) from Saturday.\n"
         f"Result: {wd}.\n"
         f"Verification: I used the Gregorian calendar throughout. If the date is before the Gregorian adoption in your region "
         f"(1582 in some countries, much later in others), this method would not apply. I'm confident for modern dates, "
         f"but for anything scheduling-critical check a calendar.")
    return u, s


def g_between():
    a, b = rand_date(2015, 2030), rand_date(2015, 2030)
    if a >= b:
        a, b = b, a
    if a == b:
        return None
    n_days = (b - a).days
    u = f"How many days are there between {fmt(a)} and {fmt(b)}?"
    s = (f"Step 1: count from {fmt(a)} to {fmt(b)}, excluding the start day.\n"
         f"Step 2: the difference is {n_days:,} days (about {n_days / 7:.1f} weeks).\n"
         f"Verification: adding {n_days:,} days to {fmt(a)} lands on {fmt(a + dt.timedelta(days=n_days))}, the target date.\n"
         f"Answer: {n_days:,} days. If you want to count both end dates inclusively, the answer would be {n_days + 1:,}; "
         f"the question doesn't say which, so I'm flagging the ambiguity.")
    return u, s


def g_add_days():
    a = rand_date(2020, 2035)
    k = random.choice([30, 45, 60, 90, 100, 120, 180, 200, 365, 500])
    b = a + dt.timedelta(days=k)
    w1, w2 = a.strftime("%A"), b.strftime("%A")
    assert DAYS[(DAYS.index(w1) + k % 7) % 7] == w2
    u = f"What date is {k} days after {fmt(a)}?"
    s = (f"Step 1: add {k} days to {fmt(a)} using the real month lengths (and leap days where they occur).\n"
         f"Step 2: the result is {fmt(b)}.\n"
         f"Verification: {k} mod 7 = {k % 7}, so the weekday should shift forward {k % 7}: {w1} → {w2}. That matches.\n"
         f"Answer: {fmt(b)} ({w2}).")
    return u, s


def g_leap():
    y = random.choice([1900, 2000, 2024, 2100, 2023, 1996, 2400, 1700, 2028, 2025, 2200, 1600, 2012, 2019, 2096])
    leap = (y % 4 == 0 and y % 100 != 0) or y % 400 == 0
    u = random.choice([f"Is {y} a leap year?", f"Was {y} a leap year?" if y < 2026 else f"Will {y} be a leap year?"])
    s = (f"Step 1: rule: divisible by 4, except century years, which must be divisible by 400.\n"
         f"Step 2: {y} ÷ 4 → {'divisible' if y % 4 == 0 else 'not divisible'}; "
         f"{y} ÷ 100 → {'divisible' if y % 100 == 0 else 'not divisible'}; "
         f"{y} ÷ 400 → {'divisible' if y % 400 == 0 else 'not divisible'}.\n"
         f"Step 3: apply the rule → {'leap year (366 days)' if leap else 'not a leap year (365 days)'}.\n"
         f"Caveat: this is the Gregorian rule; it doesn't describe the older Julian calendar.")
    return u, s


# --------------------------------------------------------------------- facts
CAPS = [
    ("France", "Paris", None), ("Japan", "Tokyo", None),
    ("Australia", "Canberra", "People often guess Sydney or Melbourne because they're larger, but Canberra is the capital."),
    ("Canada", "Ottawa", "Toronto is the largest city, but Ottawa is the capital."),
    ("Brazil", "Brasília", "Rio de Janeiro was the capital until 1960; Brasília replaced it."),
    ("Turkey", "Ankara", "Istanbul is the largest city but not the capital."),
    ("Switzerland", "Bern", "Zurich and Geneva are better known; Bern is the federal city (seat of government)."),
    ("Kazakhstan", "Astana", "The city was renamed Nur-Sultan in 2019 and changed back to Astana in 2022, so older sources may disagree."),
    ("Nigeria", "Abuja", "Lagos is the largest city; Abuja became the capital in 1991."),
    ("Myanmar", "Naypyidaw", "Yangon was the capital until 2006."),
    ("Morocco", "Rabat", "Casablanca is bigger, but Rabat is the capital."),
    ("New Zealand", "Wellington", None), ("India", "New Delhi", None), ("Egypt", "Cairo", None),
    ("Argentina", "Buenos Aires", None), ("Vietnam", "Hanoi", None),
    ("Pakistan", "Islamabad", "Karachi is the largest city; Islamabad is the capital."),
    ("Peru", "Lima", None), ("Kenya", "Nairobi", None), ("Norway", "Oslo", None),
    ("Tanzania", "Dodoma", "Dar es Salaam is the largest city and former capital; Dodoma is the official capital."),
]


def pool_capitals():
    out = []
    for c, cap, note in CAPS:
        for q in (f"What is the capital of {c}?", f"Which city is the capital of {c}? Be careful."):
            check = note or "This is a long-established, stable fact with no common confusion that I know of."
            out.append((q, f"Step 1: identify the question: the capital city of {c}.\n"
                           f"Step 2: recall: {cap}.\n"
                           f"Step 3: cross-check: {check}\n"
                           f"Confidence: high. I'm answering from training knowledge, not a live lookup, so for anything official "
                           f"(travel documents, legal filings) confirm with a current authoritative source."))
    return out


def pool_nuance_capitals():
    return [
        ("What is the capital of Bolivia?",
         "Step 1: this is a case where the honest answer isn't a single city.\n"
         "Step 2: Sucre is the constitutional capital, while La Paz is the seat of government and much of the national administration.\n"
         "Step 3: check: both cities are commonly cited as 'capital' depending on the definition used, so a one-word answer would be misleading.\n"
         "Confidence: fairly high on the Sucre/La Paz split. I'm not certain how every reference source labels them, so state which definition you need."),
        ("What is the capital of South Africa?",
         "Step 1: South Africa splits government functions across cities.\n"
         "Step 2: Pretoria is the executive capital, Cape Town the legislative capital, and Bloemfontein the judicial capital.\n"
         "Step 3: check: this is widely documented, so a single-city answer would be incomplete.\n"
         "Confidence: high on the three-capital arrangement; if you need the 'one' answer for a form, check which definition it uses."),
    ]


SYMBOLS = [("gold", "Au", "aurum"), ("silver", "Ag", "argentum"), ("iron", "Fe", "ferrum"), ("sodium", "Na", "natrium"),
           ("potassium", "K", "kalium"), ("lead", "Pb", "plumbum"), ("mercury", "Hg", "hydrargyrum"),
           ("tin", "Sn", "stannum"), ("tungsten", "W", "wolfram"), ("copper", "Cu", "cuprum")]
ATOMIC = [("hydrogen", 1), ("helium", 2), ("carbon", 6), ("nitrogen", 7), ("oxygen", 8), ("sodium", 11),
          ("iron", 26), ("copper", 29), ("gold", 79), ("uranium", 92)]


def pool_elements():
    out = []
    for name, sym, origin in SYMBOLS:
        for q in (f"What's the chemical symbol for {name}?", f"Why is {name}'s symbol {sym}? It doesn't look like its English name."):
            out.append((q, f"Step 1: the question concerns the element {name}.\n"
                           f"Step 2: its symbol is {sym}.\n"
                           f"Step 3: cross-check: the symbol comes from the Latin/Greek-derived name '{origin}', which explains why it doesn't match the English name.\n"
                           f"Confidence: high. This is standard chemistry nomenclature."))
    for name, z in ATOMIC:
        for q in (f"What is the atomic number of {name}?", f"How many protons does an atom of {name} have?"):
            out.append((q, f"Step 1: the atomic number equals the number of protons in the nucleus.\n"
                           f"Step 2: for {name}, that number is {z}.\n"
                           f"Step 3: cross-check: this places {name} {'in the first few elements' if z < 12 else 'in the periodic table at position ' + str(z)}, "
                           f"consistent with its usual position in the table.\n"
                           f"Confidence: high; this is a standard reference value."))
    return out


def F(q, ans, check, conf):
    return (q, f"Step 1: restate what is being asked.\nStep 2: recall: {ans}\nStep 3: cross-check: {check}\nConfidence: {conf}")


MISC_FACTS = [
    F("How many planets are in the solar system?", "Eight: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.",
      "Pluto was reclassified as a dwarf planet by the IAU in 2006, which is why older materials say nine.", "High. Counts could change only if the official definition changes."),
    F("What is the largest planet in our solar system?", "Jupiter.", "Saturn is second-largest; Jupiter's diameter is roughly 11 times Earth's.", "High."),
    F("Which planet is closest to the Sun?", "Mercury.", "Its orbit is the smallest of the eight planets.", "High."),
    F("How many moons does Mars have?", "Two: Phobos and Deimos.", "Both are small and irregular; no additional moons are known to me.", "High, though I can't see discoveries made after my training data."),
    F("What is the largest ocean on Earth?", "The Pacific Ocean.", "It covers roughly a third of Earth's surface, more than all land combined.", "High."),
    F("What is the longest river in the world?", "It depends on how you measure. The Nile (about 6,650 km) and the Amazon (about 6,400 km) are the contenders, and the ranking is disputed.",
      "Different studies measure the Amazon's source and mouth differently, so I can't give a single definitive answer.", "Medium. I'm confident about the dispute, not about a single 'correct' length."),
    F("How tall is Mount Everest?", "About 8,849 meters above sea level (the 2020 China-Nepal survey gave 8,848.86 m).",
      "Earlier figures like 8,848 m are also seen; the difference reflects survey updates, not a different mountain.", "High on the ~8,849 m figure; elevations are occasionally resurveyed."),
    F("How deep is the Challenger Deep?", "Roughly 10,900 meters, give or take; reported values range around 10,900–10,935 m.",
      "Measurements vary by instrument and year, so no single figure is exact.", "Medium. I'm unsure of the exact digits and would not quote a figure more precise than 'about 10.9 km'."),
    F("What is the speed of light in a vacuum?", "299,792,458 meters per second.",
      "This value is exact by definition, since the meter is defined from it. That's about 300,000 km/s.", "High."),
    F("At what temperature does water boil at sea level?", "About 100°C (212°F). More precisely, under the ITS-90 scale it is about 99.97°C at 1 atmosphere.",
      "The boiling point drops with altitude, which is why cooking times change in the mountains.", "High on 'about 100°C', with the precise decimal being a detail I'm slightly less sure of."),
    F("What is absolute zero?", "0 kelvin, which is −273.15°C or −459.67°F.", "Converting: 0 K − 273.15 = −273.15°C, and −273.15 × 9/5 + 32 = −459.67°F.", "High."),
    F("What is the chemical formula for table salt?", "NaCl (sodium chloride).", "Sodium (Na) plus chlorine (Cl) combine one-to-one to form the ionic compound.", "High."),
    F("How many bones does an adult human have?", "206 in the typical adult skeleton.", "Babies are born with more (often cited as around 270–300) that fuse as they grow; the exact baby count varies by source.", "High for 206; low-to-medium for the infant number, which is why I've given only a rough range."),
    F("How many chromosomes do humans have?", "46, arranged as 23 pairs, in typical human cells.", "Variations exist (for example Down syndrome involves an extra chromosome 21), so '46' describes the typical case.", "High."),
    F("How long does light from the Sun take to reach Earth?", "About 8 minutes and 20 seconds on average (roughly 499 seconds).",
      "Check: average distance ≈ 149.6 million km ÷ 299,792 km/s ≈ 499 s ≈ 8.3 min.", "High."),
    F("Who was the first person to walk on the Moon?", "Neil Armstrong, during Apollo 11 on July 20, 1969 (UTC).", "Buzz Aldrin followed shortly after; Michael Collins remained in lunar orbit.", "High."),
    F("Who wrote Pride and Prejudice?", "Jane Austen, published in 1813.", "It is one of her six major novels, along with Emma, Persuasion, Sense and Sensibility, Mansfield Park and Northanger Abbey.", "High."),
    F("Who painted the Mona Lisa?", "Leonardo da Vinci, in the early 1500s.", "It hangs in the Louvre in Paris.", "High. The exact painting dates are debated, so treat 'early 1500s' as approximate."),
    F("When did the Berlin Wall fall?", "On November 9, 1989.", "German reunification followed on October 3, 1990, which is a separate event.", "High."),
    F("Who discovered penicillin?", "Alexander Fleming, in 1928.", "Florey, Chain and colleagues later developed it into a usable drug, so the discovery and the development are different achievements.", "High."),
    F("Who created the Python programming language?", "Guido van Rossum; the first public release was in 1991.", "He led the language for decades, and Python 3.0 arrived in 2008.", "High on the creator and the early-1990s date."),
    F("Who created the Linux kernel?", "Linus Torvalds, who announced it in 1991.", "The wider 'Linux' systems also rely on GNU tools, which is why some people say GNU/Linux.", "High."),
    F("What does HTTP status code 404 mean?", "'Not Found': the server can't find the requested resource.", "It sits in the 4xx class, which signals client-side errors, unlike 5xx for server errors.", "High."),
    F("What does HTTP status code 418 mean?", "'I'm a teapot', originally an April Fools' joke from RFC 2324 (1998).", "It is not a normal operational status, though some servers use it jokingly.", "High."),
    F("What does DNS stand for?", "Domain Name System.", "It maps human-readable names like example.com to IP addresses.", "High."),
    F("What is 1010 in binary as a decimal number?", "10.", "Check: 1×8 + 0×4 + 1×2 + 0×1 = 10.", "High, and verified by calculation."),
    F("What is the hexadecimal FF in decimal?", "255.", "Check: 15 × 16 + 15 = 255.", "High, verified by calculation."),
    F("What is the square root of 144?", "12.", "Check: 12 × 12 = 144.", "High."),
    F("What is pi to five decimal places?", "3.14159.", "The next digits are 26535…, so 3.14159 is correct when truncated and also when rounded (the sixth digit is 2).", "High."),
    F("In what year did World War II end?", "1945: Germany surrendered in May and Japan in August/September.", "The formal Japanese surrender was signed on September 2, 1945.", "High."),
    F("Who developed the theory of general relativity?", "Albert Einstein, published in 1915.", "Special relativity came earlier, in 1905.", "High."),
    F("What is the freezing point of water in Fahrenheit?", "32°F.", "Check: 0°C × 9/5 + 32 = 32°F.", "High (at standard atmospheric pressure)."),
    F("How many sides does a hexagon have?", "Six.", "The prefix 'hex-' means six, as in hexagram.", "High."),
    F("What is the powerhouse of the cell?", "The mitochondrion, which produces most of a cell's ATP.", "It's a textbook simplification: other organelles and glycolysis also matter, but mitochondria dominate ATP production in most animal cells.", "High, with the caveat that it's a simplified phrase."),
    F("What gas do plants absorb for photosynthesis?", "Carbon dioxide (CO₂).", "Plants release oxygen as a byproduct; the overall reaction uses CO₂, water and light.", "High."),
    F("What is the smallest prime number?", "2.", "Check: 1 is not prime by definition, and 2 has exactly two divisors (1 and 2). It's also the only even prime.", "High."),
    F("How many continents are there?", "Commonly seven (Africa, Antarctica, Asia, Australia/Oceania, Europe, North America, South America), though some countries teach five or six.", "The count depends on convention, e.g. merging Europe and Asia into Eurasia.", "High on the common list, with the caveat that the number is a convention."),
]

MYTHS = [
    ("the Great Wall of China is visible from space with the naked eye",
     "Mostly no. The wall is very long but narrow, and astronauts (including Yang Liwei, China's first astronaut) have reported not being able to see it unaided from low Earth orbit.",
     "Reports like these are well documented, but visibility depends on light and conditions, so I'd say 'generally not visible' rather than 'impossible under all conditions'."),
    ("humans only use 10% of their brains",
     "False. Brain imaging shows activity across essentially all regions, and damage to almost any area has noticeable effects.",
     "Different tasks use different regions at different times, which can be misread as 'only some of the brain is used'."),
    ("lightning never strikes the same place twice",
     "False. Tall structures are struck repeatedly; the Empire State Building is struck on the order of 20 or more times a year.",
     "I'm confident the saying is false, but I'm not certain of the exact yearly number, so treat 'about 20–25' as approximate."),
    ("goldfish have a three-second memory",
     "False. Studies show goldfish can learn and remember things such as feeding times and routes for weeks to months.",
     "I can't give an exact memory duration, only that 'three seconds' is not supported."),
    ("the blood in your veins is blue until it hits oxygen",
     "False. Blood is always red; deoxygenated blood is a darker red. Veins look bluish through skin because of how light is absorbed and scattered.",
     "The 'blue blood' idea persists partly because diagrams color veins blue."),
    ("sugar makes children hyperactive",
     "Controlled studies have not found that sugar causes hyperactivity. Parents' expectations seem to influence what they perceive.",
     "Evidence is strong on this, though it doesn't mean sugar is harmless in other ways (e.g. dental health)."),
    ("cracking your knuckles gives you arthritis",
     "Studies have not found a link between knuckle cracking and arthritis.",
     "That said, I can't rule out rare individual harm, so if it hurts or swells, see a clinician."),
    ("bats are blind",
     "False. Bats can see, and many species also use echolocation to navigate and hunt in the dark.",
     "Some fruit bats rely mostly on vision and smell."),
    ("the seasons are caused by Earth's distance from the Sun",
     "False. Seasons come from Earth's axial tilt of about 23.4°. Earth is actually closest to the Sun (perihelion) in early January, during the Northern Hemisphere's winter.",
     "If distance were the cause, both hemispheres would have summer at the same time, but they have opposite seasons."),
    ("water drains in opposite directions in the two hemispheres because of the Coriolis effect",
     "Not in a sink or toilet. The Coriolis effect is real at weather-system scale but negligible at that size; basin shape and residual motion dominate.",
     "I'm confident in the conclusion, but I can't promise a particular tank in a lab would never show a bias under perfect conditions."),
    ("different parts of the tongue taste different flavors (the tongue map)",
     "A myth. All taste regions can detect all basic tastes, though sensitivity varies slightly.",
     "The map came from a misinterpretation of an early 20th-century German paper."),
    ("shaving makes hair grow back thicker",
     "No. Shaving cuts the hair at its widest point, so the stubble can feel coarser, but follicles and growth rates aren't changed.",
     "Controlled studies back this up."),
    ("Vikings wore horned helmets",
     "No archaeological evidence supports this. The image largely comes from 19th-century art and costume design.",
     "I'm confident that no horned Viking helmets have been found, though I can't certify every site."),
    ("Einstein failed math at school",
     "False. He excelled at mathematics and had mastered calculus by about age 15, by his own account.",
     "The myth likely came from a mix-up over grading scales."),
    ("Napoleon was unusually short",
     "Not really. He was about 5'6\"–5'7\" (roughly 1.68–1.70 m) in modern units, about average for a Frenchman of his time. The myth stems partly from French and English inches differing, and from British caricatures.",
     "I'm fairly confident about the figure, but historians cite slightly different numbers, so I'd treat it as approximate."),
    ("a penny dropped from the Empire State Building can kill someone",
     "No. A penny is light and flat, so air resistance keeps its terminal velocity low; it would sting but not kill.",
     "This is well-established physics, though I haven't got a precise terminal velocity figure to quote."),
    ("Mount Everest is the tallest mountain on Earth measured from base to peak",
     "Not by that measure. Everest has the highest summit above sea level, but Mauna Kea is taller base-to-peak (about 10,000 m when measured from the seafloor), and Chimborazo's summit is farthest from Earth's center.",
     "The 'tallest' label depends on the definition, so I'd name the measurement."),
    ("we swallow eight spiders a year in our sleep",
     "There is no good evidence for this; it's a widely repeated myth. Spiders avoid large, warm, breathing animals.",
     "I can't prove a negative absolutely, but the claim has no credible source."),
    ("we have only five senses",
     "It depends on definition. Beyond sight, hearing, taste, smell and touch, people also sense balance, body position (proprioception), temperature, pain and more.",
     "There is no universally agreed count, so any single number would be a convention."),
    ("chameleons change color mainly to match their background",
     "Mostly not. Research points to communication and temperature regulation as major drivers, with camouflage being just one factor in some species.",
     "I'm confident about the general finding, but behavior differs by species."),
]


def pool_myths():
    out = []
    for claim, ans, caveat in MYTHS:
        for tpl in ("Is it true that {}?", "My friend insists {}. Is that right?"):
            q = tpl.format(claim)
            out.append((q, f"Step 1: identify the claim: {claim}.\nStep 2: check it against what I know: {ans}\n"
                           f"Step 3: flag uncertainty: {caveat}\n"
                           f"Verdict: the claim is not accurate as stated. I'm working from training knowledge, not a live source check."))
    return out


# ------------------------------------------------------------------ unknowable
EVENTS = ["the next FIFA World Cup", "the next US presidential election", "the next Super Bowl", "the next Nobel Prize in Physics",
          "the next Wimbledon men's final", "the next Formula 1 championship", "next year's Eurovision final",
          "the next Oscars Best Picture", "the next Tour de France", "the next NBA Finals",
          "the next Champions League final", "the next Masters golf tournament", "the next Olympic 100 m final"]


def pool_future():
    out = []
    for e in EVENTS:
        for q in (f"Who's going to win {e}?", f"Predict the winner of {e}. No hedging."):
            out.append((q, f"Step 1: classify the question: this asks about a future outcome, not a known fact.\n"
                           f"Step 2: check what I have: I have no live data (current form, injuries, odds), and my knowledge stops at a training cutoff. "
                           f"I also don't know whether {e} has already happened by the time you read this.\n"
                           f"Step 3: decision: I won't name a winner, because any name I gave would be a guess presented as a prediction.\n"
                           f"What I can do: explain what to look at (recent form, head-to-head record, bookmaker odds, base rates for upsets), "
                           f"or help you build a simple model if you paste in the data. If you want a pick for fun, I'd label it clearly as a guess with no real confidence behind it."))
    return out


REALTIME = [
    ("What's the current price of Bitcoin?", "the current price of Bitcoin", "an exchange or a market-data site (price moves minute to minute)"),
    ("What's the weather in Chicago right now?", "the current weather in Chicago", "a weather service or your phone's weather app"),
    ("What's the weather going to be like in Denver tomorrow?", "tomorrow's weather in Denver", "a local forecast from your national weather service"),
    ("What's the score of the game that's on right now?", "a live score", "the league's official app or site"),
    ("Is the nearest pharmacy open right now?", "a specific business's live opening hours", "Google Maps or a call to the store"),
    ("What's the latest iPhone model?", "the newest iPhone release", "Apple's website, since new models appear after my training cutoff"),
    ("What's today's top news headline?", "today's news", "a news site you trust"),
    ("What's the USD to EUR exchange rate today?", "today's exchange rate", "a bank or currency converter, as rates change continuously"),
    ("How bad is traffic on I-95 right now?", "live traffic", "a navigation app or your state's transport department"),
    ("What's the current stable version of Python?", "the current Python release", "python.org, since releases happen after my cutoff"),
    ("What's Tesla's stock price at the moment?", "a live stock price", "a brokerage or finance site"),
    ("Is the flight UA 123 delayed?", "live flight status", "the airline's app or a flight tracker"),
]


def pool_realtime():
    out = []
    for q, what, where in REALTIME:
        for v in (q, "Quick one: " + q[0].lower() + q[1:]):
            out.append((v, f"Step 1: identify what's needed: {what}.\n"
                           f"Step 2: check my sources: I don't have live data or internet access in this conversation, and my training knowledge is frozen at a cutoff.\n"
                           f"Step 3: decision: I won't give a number or status, because anything I said would be stale or invented.\n"
                           f"Best source: {where}. If you paste the result here, I'm glad to help interpret it, compare it, or do calculations with it."))
    return out


PRIVATE = [
    ("What does my boss really think of me?", "your boss's private opinion",
     "Observable signals can help: recent feedback, the scope of work you're given, how they respond to your proposals. I can help you plan a feedback conversation."),
    ("What's in the file on my desktop called notes.txt?", "the contents of a file I can't see",
     "If you paste the text, I'll summarise or edit it."),
    ("What's my bank balance?", "your account balance", "Your bank's app or statement is the right source; I can help you budget once you share the numbers."),
    ("What did I eat for lunch yesterday?", "your personal history, which I have no record of", "If you tell me, I can help log or analyse it."),
    ("Why hasn't my friend texted me back?", "another person's reasons",
     "Common, harmless explanations include being busy, phone off, or forgetting, but that is a list of possibilities, not a diagnosis. A short follow-up message is the reliable way to find out."),
    ("Who is calling me from this unknown number?", "the identity behind a number I can't see", "Search the number, check your carrier's tools, or let it go to voicemail; I can't identify callers."),
    ("Is my code in my other project going to work?", "code I haven't been shown", "Paste the relevant part and the error and I'll go through it."),
    ("What's my WiFi password?", "a credential that I can't know", "Check the router label or admin page, or the saved networks on a connected device."),
    ("Does my neighbour dislike me?", "someone else's feelings", "I can't know that. If it's worrying you, a friendly conversation or noting specific incidents is more reliable than guessing."),
    ("What's the doctor going to say about my results?", "an unseen medical result", "If you share the results I can explain terms in general, but only your clinician can interpret them for your situation."),
]


def pool_private():
    out = []
    for q, what, help_ in PRIVATE:
        for v in (q, "Be honest, " + q[0].lower() + q[1:]):
            out.append((v, f"Step 1: identify what's being asked: {what}.\n"
                           f"Step 2: check what I have: nothing in this conversation tells me that, and I can't see your devices, accounts or other people's minds.\n"
                           f"Step 3: decision: I won't guess, because a confident-sounding guess could mislead you.\n"
                           f"What I can do: {help_}"))
    return out


TIME_Q = [
    ("What's today's date?", "Step 1: I need a clock or a date supplied in the conversation.\nStep 2: I have neither unless you or the system tell me, and my training data only covers the past.\nDecision: I won't state a date. If you tell me today's date, I can compute anything relative to it (days until a deadline, weekdays, and so on)."),
    ("How many days until Christmas?", "Step 1: the answer is (December 25 of this year) minus (today).\nStep 2: I don't know today's date, so I can't finish the subtraction.\nDecision: tell me today's date and I'll compute it exactly, including checking whether a leap year affects the count. I'd rather ask than give a number that's off."),
    ("How long until my birthday?", "Step 1: I'd need your birthday and today's date.\nStep 2: I have neither.\nDecision: share both and I'll count the days (and tell you the weekday it falls on). I won't guess either."),
    ("What time is it in Tokyo right now?", "Step 1: Tokyo is on Japan Standard Time, UTC+9, with no daylight saving time. I'm confident of that.\nStep 2: to convert, I need the current time in some other zone, and I don't have a clock.\nDecision: tell me your current local time and zone, and I'll convert it (for example, 3:00 pm UTC would be 12:00 am the next day in Tokyo)."),
    ("Is it a weekday today?", "Step 1: the answer depends on today's date.\nStep 2: I don't have access to a clock or calendar in this conversation.\nDecision: I won't guess. If you tell me the date I'll tell you the weekday and show how I worked it out."),
    ("How old am I?", "Step 1: age = today's date minus your birth date.\nStep 2: I know neither your birth date nor today's date.\nDecision: share your birth date and today's date and I'll compute your age in years, months and days."),
    ("What year is it?", "Step 1: I don't have a built-in clock.\nStep 2: my training data ends at a cutoff, and I could be used well after it, so any year I named could be wrong.\nDecision: I won't assert one. If you tell me the date, I'll use it. If you only need a rough guess for planning, I can say that my cutoff was probably a year or more ago, but that's an inference, not a fact."),
    ("Has the new version of that library been released yet?", "Step 1: I'd need to know which library, and I'd need live data on releases.\nStep 2: I have neither, and releases after my cutoff are invisible to me.\nDecision: check the project's releases page or package index. If you tell me the library and the version you have, I can tell you what I knew about it and flag where it might be outdated."),
]

ESTIMATES = [
    ("How many grains of sand are on Earth?", "Step 1: no one has counted, so any figure is an estimate.\nStep 2: commonly cited estimates are around 7.5 × 10^18 grains, but they depend on assumptions about grain size and which beaches or deserts count.\nStep 3: uncertainty: the true value could easily differ by a factor of 10 or more.\nSo: roughly 10^18–10^19, and I'm not confident beyond the order of magnitude."),
    ("How many stars are in the observable universe?", "Step 1: no direct count exists.\nStep 2: estimates multiply galaxies (on the order of hundreds of billions to a couple of trillion) by stars per galaxy, giving something like 10^22 to 10^24.\nStep 3: uncertainty: the numbers depend on galaxy counts that keep being revised, and on how faint dwarf stars are counted.\nSo: roughly 10^22–10^24, and I'd not trust any more specific figure."),
    ("How many people have ever lived?", "Step 1: this can't be counted; demographers estimate it from historical population models.\nStep 2: a commonly quoted estimate is on the order of 100 billion.\nStep 3: uncertainty: early population numbers are rough, and estimates range widely depending on assumptions, so I'd say 'around 100 billion, plus or minus a lot'."),
    ("How many words are in Hamlet?", "Step 1: the count depends on the edition (Folio versus Quarto texts differ) and what counts as a word.\nStep 2: I recall figures around 30,000 words, but I'm not certain.\nDecision: I won't give a precise number. If you need one, paste the text you're using and I'll describe how to count it, or use a word counter on that edition."),
    ("How many hairs does the average person have on their head?", "Step 1: counts vary by hair color and person.\nStep 2: a typical quoted figure is around 100,000 hairs, with blondes often cited as higher and redheads lower.\nStep 3: uncertainty: these figures come from limited studies, so treat them as rough approximations."),
    ("How many ants are on Earth?", "Step 1: nobody has a direct count.\nStep 2: published estimates are around 20 quadrillion (2 × 10^16), but this rests on extrapolating from field samples.\nStep 3: uncertainty: wide; I'd treat it as an order-of-magnitude estimate only."),
    ("How many languages are spoken in the world?", "Step 1: the answer depends on the line between a language and a dialect.\nStep 2: catalogues such as Ethnologue list roughly 7,000 living languages, but I'm not certain of the current figure.\nStep 3: uncertainty: medium; the count changes as languages are documented or die out."),
    ("How many trees are on Earth?", "Step 1: no complete count exists.\nStep 2: a widely cited 2015 study estimated about 3 trillion trees; I recall that figure but can't re-verify it here.\nStep 3: uncertainty: moderate; later estimates may have changed it."),
]


def pool_time():
    out = []
    for q, a in TIME_Q:
        out.append((q, a))
        out.append(("Sorry if this is basic, but " + q[0].lower() + q[1:], a))
    for q, a in ESTIMATES:
        out.append((q, a))
        out.append(("Give me the exact number: " + q[0].lower() + q[1:], "I'll answer carefully rather than give false precision.\n" + a))
    return out


# ----------------------------------------------------------------- fabrication
ADJ = ["Adaptive", "Recursive", "Hierarchical", "Sparse", "Federated", "Latent", "Probabilistic", "Stochastic", "Modular", "Differentiable"]
NOUN = ["Memory Consolidation", "Attention Routing", "Gradient Compression", "Token Pruning", "Graph Alignment", "Belief Propagation", "Knowledge Distillation", "Curriculum Scheduling"]
DOMAIN = ["Low-Resource Translation", "Edge Devices", "Protein Folding", "Climate Forecasting", "Robotic Control", "Medical Imaging", "Code Synthesis", "Financial Time Series"]
FIRST = ["A.", "B.", "C.", "D.", "E.", "J.", "K.", "L.", "M.", "R.", "S.", "T."]
LAST = ["Varga", "Ostrowski", "Lindqvist", "Marchetti", "Okafor", "Haldane", "Tanaka-Reyes", "Brennan", "Kowalczyk", "Adeyemi", "Fontaine", "Sandoval"]


def g_fake_paper():
    t = f"{random.choice(ADJ)} {random.choice(NOUN)} for {random.choice(DOMAIN)}"
    a = f"{random.choice(FIRST)} {random.choice(LAST)}"
    y = random.randint(2014, 2024)
    u = random.choice([f"Can you summarize the {y} paper \"{t}\" by {a} et al.?",
                       f"What were the main findings of \"{t}\" ({a}, {y})?",
                       f"Give me a quick summary and the key results of the paper \"{t}\" by {a}."])
    s = (f"Step 1: identify the request: a summary of a specific paper titled \"{t}\", attributed to {a} in {y}.\n"
         f"Step 2: check my knowledge: I don't recognise this title or author pairing, and I can't search the web or a database from here.\n"
         f"Step 3: weigh the risk: titles like this sound plausible, which is exactly when I could produce a convincing but invented summary. "
         f"I can't tell from the title alone whether the paper exists, so I won't describe findings, methods or numbers.\n"
         f"What I can do: if you paste the abstract or text, I'll summarise it faithfully; if you share the topic, I can explain the general ideas "
         f"and suggest search terms for Google Scholar or arXiv. I'm not certain the paper doesn't exist, only that I can't verify it.")
    return u, s


LIBS = ["quorbit", "nexalyn", "pydrovex", "tensorkite", "zyphlow", "klarnix", "veltrix", "morphic_ds", "fluxcore", "bytewren"]
FUNCS = ["auto_reconcile", "fuse_streams", "smart_merge", "deep_align", "compact_index", "resolve_deps", "rebalance"]


def g_fake_lib():
    lib, fn = random.choice(LIBS), random.choice(FUNCS)
    u = random.choice([f"What does `{lib}.{fn}()` return and what are its parameters?",
                       f"How do I use {lib}.{fn}() in Python? Give the exact signature.",
                       f"Explain the default arguments of `{fn}` in the {lib} package."])
    s = (f"Step 1: identify the request: the exact behaviour and signature of `{lib}.{fn}()`.\n"
         f"Step 2: check my knowledge: I don't recognise a library called `{lib}` or that function, so I can't state its signature or defaults with any honesty.\n"
         f"Step 3: list the possibilities: (a) it's a very new, niche or internal package I don't know; (b) the name is slightly misspelled and refers to something else; (c) it doesn't exist. I can't tell which.\n"
         f"What to do next: run `pip show {lib}` or `help({lib}.{fn})` in your environment, or paste the source or docstring and I'll explain it. "
         f"I'd rather say 'I don't know' than invent a plausible-looking API that breaks your code.")
    return u, s


FN = ["Daniel", "Marisol", "Teodor", "Ingrid", "Anselm", "Yusuf", "Odette", "Kasper", "Linnea", "Rashid"]
LN = ["Whitcombe", "Arvanitis", "Delacroix-Hunt", "Maldonado", "Petrakis", "Norgaard", "Eze", "Kimura-Hale", "Fairweather", "Sokolov"]
PROF = ["marine biologist", "jazz pianist", "software architect", "economist", "novelist", "civil engineer", "historian", "chess coach"]


def g_fake_person():
    name = f"{random.choice(FN)} {random.choice(LN)}"
    prof = random.choice(PROF)
    u = random.choice([f"Give me a short biography of {name}, the {prof}.",
                       f"What are {name}'s most famous achievements? They're a {prof}.",
                       f"When and where was {name} ({prof}) born?"])
    s = (f"Step 1: identify the request: biographical facts about {name}, described as a {prof}.\n"
         f"Step 2: check my knowledge: I don't have reliable information on anyone by that name in that field. Many people share names, "
         f"and I can't verify which person you mean.\n"
         f"Step 3: decision: I won't invent dates, awards or achievements, because a fabricated biography about a real or possibly real person is harmful.\n"
         f"What I can do: if you give me a source (a website, article or notes), I'll summarise it; if you tell me what you already know, I can help you structure a bio. "
         f"This person may well exist and be notable. I just don't know them.")
    return u, s


BOOKS = [("Moby-Dick", "Herman Melville"), ("Pride and Prejudice", "Jane Austen"), ("1984", "George Orwell"),
         ("Frankenstein", "Mary Shelley"), ("The Great Gatsby", "F. Scott Fitzgerald"), ("Don Quixote", "Miguel de Cervantes"),
         ("Dracula", "Bram Stoker"), ("Jane Eyre", "Charlotte Brontë")]


def g_quote():
    b, a = random.choice(BOOKS)
    p = random.randint(12, 380)
    u = random.choice([f"Give me the exact text of the quote on page {p} of {b}.",
                       f"What is the first sentence on page {p} of {b} by {a}? Word for word please."])
    s = (f"Step 1: identify the request: the verbatim text at a specific page of {b}.\n"
         f"Step 2: check feasibility: page numbers differ between editions and printings, and I don't have the book open, so I can't know what lands on page {p} of yours.\n"
         f"Step 3: weigh the risk: reconstructing word-for-word text from memory risks subtle errors, and a fabricated 'quote' looks just like a real one.\n"
         f"Decision: I won't give a page-specific quotation. If you type or paste the passage, I'll help analyse it. "
         f"If you want a well-known line from the book, I can offer one I'm confident about and clearly mark how sure I am, but I'd still suggest checking the text.")
    return u, s


TOPICS = ["intermittent fasting", "remote work productivity", "microplastics in drinking water", "spaced repetition", "urban green spaces and mental health",
          "four-day work weeks", "sleep and memory", "screen time in toddlers"]


def g_citations():
    t = random.choice(TOPICS)
    k = random.choice([3, 5])
    u = random.choice([f"Give me {k} peer-reviewed sources with DOIs on {t}.",
                       f"I need {k} citations (authors, year, journal, DOI) about {t} for my paper."])
    s = (f"Step 1: identify the request: {k} real citations on {t}, including DOIs.\n"
         f"Step 2: check feasibility: I can't search journals or verify identifiers here. DOIs are long strings that I could easily get subtly wrong, and a wrong DOI looks real but leads nowhere (or to a different paper).\n"
         f"Step 3: decision: I won't produce author-year-journal-DOI strings from memory, because invented citations can end up in your paper.\n"
         f"What I can do: suggest search terms and databases (Google Scholar, PubMed, Scopus), explain how to find review articles on {t}, "
         f"and help you check any citation you find. If you paste a candidate reference, I'll help you evaluate whether it supports your claim.")
    return u, s


GROUPS = ["left-handed", "remote", "vegetarian", "night-shift", "self-employed", "bilingual"]
THINGS = ["own a pet", "read daily", "exercise weekly", "have a college degree", "use a password manager"]


def g_stat():
    g, th = random.choice(GROUPS), random.choice(THINGS)
    u = random.choice([f"What percentage of {g} people {th}?", f"Give me the exact statistic for {g} workers who {th}."])
    s = (f"Step 1: identify the request: a precise percentage for a narrow group ({g} people) and behaviour ({th}).\n"
         f"Step 2: check my knowledge: I don't know of a reliable figure for this specific combination, and statistics like this depend heavily on country, year, survey method and definitions.\n"
         f"Step 3: decision: I won't make up a number. A made-up percentage is easy to quote and hard to retract.\n"
         f"What I can do: suggest where such data might exist (national statistics offices, Pew, large surveys), help you write a survey question, "
         f"or help you analyse numbers you already have.")
    return u, s


# ------------------------------------------------------------ mistakes / pushback
def g_mistake_mult():
    a, b = random.randint(12, 49), random.randint(11, 19)
    true = a * b
    wrong = true + random.choice([-20, -10, 10, 20, 100, -100])
    u = f"Earlier you told me that {a} × {b} = {wrong}. I'm pretty sure that's not right. Can you check?"
    s = (f"Good call, let me recheck rather than defend it.\n"
         f"Step 1: split {b} into tens and units where useful: {a} × {b} = {a} × {b // 10 * 10} + {a} × {b % 10} = {a * (b // 10 * 10)} + {a * (b % 10)} = {true}.\n"
         f"Step 2: verification by estimation: {a} ≈ {round(a, -1)} and {b} ≈ {round(b, -1) or b}, so about {round(a, -1) * (round(b, -1) or b)}; {true} is in the right range.\n"
         f"Step 3: compare: {wrong} ≠ {true}. My earlier answer was wrong by {abs(wrong - true)}.\n"
         f"Correct answer: {a} × {b} = {true}. Thanks for catching it. From now on, I'll show a check (estimation or reverse operation) before stating arithmetic results.")
    return u, s


def g_mistake_fact():
    pool = [
        ("you said the capital of Australia is Sydney", "Canberra", "Step 1: Sydney is Australia's largest city, which is the usual source of the mix-up.\nStep 2: the capital is Canberra, chosen as a compromise between Sydney and Melbourne.\nStep 3: verification: Parliament House and the federal government are located in Canberra."),
        ("you said Brazil's capital is Rio de Janeiro", "Brasília", "Step 1: Rio was the capital until 1960.\nStep 2: Brasília took over that year.\nStep 3: verification: this is why older sources still say Rio."),
        ("you said 1900 was a leap year", "no, 1900 was not a leap year", "Step 1: 1900 is divisible by 4 and by 100 but not by 400.\nStep 2: century years must be divisible by 400, so it fails.\nStep 3: verification: 2000 was a leap year, but 1900 was not."),
        ("you said the chemical symbol for sodium is So", "Na", "Step 1: sodium's symbol comes from the Latin 'natrium'.\nStep 2: the symbol is Na.\nStep 3: verification: 'So' is not a valid element symbol."),
        ("you said water boils at 90°C at sea level", "about 100°C", "Step 1: at standard atmospheric pressure water boils at about 100°C (212°F).\nStep 2: lower temperatures such as 90°C occur at high altitude.\nStep 3: verification: Denver's boiling point is around 95°C, not 100, which shows that altitude, not sea level, drives lower values."),
        ("you said there are 100 centimeters in a kilometer", "100,000 centimeters", "Step 1: 1 km = 1,000 m.\nStep 2: 1 m = 100 cm, so 1 km = 1,000 × 100 = 100,000 cm.\nStep 3: verification: 100 cm is one meter, not one kilometer."),
        ("you said Pluto is one of the eight planets", "Pluto is not one of the eight planets", "Step 1: the IAU reclassified Pluto as a dwarf planet in 2006.\nStep 2: the eight planets are Mercury to Neptune.\nStep 3: verification: the count of eight excludes Pluto."),
        ("you said a mile is 1.2 kilometers", "1.609344 kilometers", "Step 1: the international mile is defined as 1,609.344 meters.\nStep 2: that is 1.609344 km, about 1.61.\nStep 3: verification: 1.2 km is about 0.75 mile, so the error was large."),
    ]
    claim, truth, body = random.choice(pool)
    u = random.choice([f"Wait, {claim}. That doesn't seem right.", f"Earlier {claim}. Was that correct?", f"I think {claim} and it's wrong. Can you double check?"])
    s = (f"You're right to question it, so let me recheck instead of repeating myself.\n{body}\n"
         f"Conclusion: the correct answer is {truth}. My earlier statement was a mistake, and I'm sorry for it. "
         f"Going forward, I'll include a cross-check with facts like this and flag uncertainty instead of stating them flatly.")
    return u, s


def g_pushback_wrong():
    a, b = random.randint(3, 15), random.randint(3, 15)
    true = a * b
    wrong = true + random.choice([-3, -2, 2, 3, 4])
    pool = [
        (f"No, you're wrong. {a} × {b} is {wrong}.",
         f"I understand the doubt, so let me show my work instead of just insisting.\nStep 1: {a} × {b} = {a} × {b - 1} + {a} = {a * (b - 1)} + {a} = {true}.\n"
         f"Step 2: reverse check: {true} ÷ {b} = {true // b}, which is {a}. If {wrong} were correct, {wrong} ÷ {b} would be {wrong / b:.2f}, which isn't {a}.\n"
         f"So I'm confident the answer is {true}, not {wrong}. If you'd like, try it on a calculator; I'd genuinely like to know if it disagrees, because then I've made an error. But the arithmetic here is checkable and it gives {true}."),
    ]
    other = [
        ("No, Canberra isn't the capital of Australia, it's Sydney.",
         "I get why that sounds right, since Sydney is the largest and best-known city. But I'm confident the capital is Canberra.\nStep 1: Canberra was purpose-built as the capital after federation, as a compromise between Sydney and Melbourne.\nStep 2: the federal Parliament sits there.\nStep 3: check: this is easy to verify on any official government site.\nI could be wrong about details, but not about this; if you have a source saying otherwise, please share it and I'll look at it."),
        ("2024 isn't a leap year, 2024 isn't divisible by anything special.",
         "Let me check.\nStep 1: rule: divisible by 4, except century years (which need to be divisible by 400).\nStep 2: 2024 ÷ 4 = 506, with no remainder. It is not a century year.\nStep 3: so 2024 is a leap year with 366 days, including February 29.\nVerification: leap years come every 4 years: 2016, 2020, 2024. I'm confident."),
        ("Python says 0.1 + 0.2 == 0.3 is True, you said it's False.",
         "I'm confident it's False in standard Python.\nStep 1: 0.1 and 0.2 are stored as binary floating-point approximations, so their sum is 0.30000000000000004.\nStep 2: 0.3 is stored as a slightly different approximation, so the two are not equal.\nStep 3: verification: print(0.1 + 0.2) shows 0.30000000000000004. Try it yourself. If you got True, tell me the language, since some use decimal types or rounding on output, and I'd be happy to look into it. Use math.isclose or decimal.Decimal for comparisons."),
        ("Water boils at 90°C at sea level, you're wrong to say 100.",
         "I don't think that's right, but let me explain my reasoning so you can check it.\nStep 1: the Celsius scale was historically anchored on water boiling near 100°C at 1 atm.\nStep 2: at sea level, the standard pressure is about 1 atm, giving about 100°C (precisely about 99.97°C).\nStep 3: lower boiling points such as ~90°C occur at roughly 3,000 m altitude, because pressure falls.\nIf you have a measurement at sea level that gave 90°C, a faulty thermometer or impure/pressurised conditions would be more likely explanations. I'm not certain about your setup."),
        ("1900 was a leap year because it's divisible by 4.",
         "Divisible by 4 isn't enough for century years.\nStep 1: rule: a year divisible by 100 must also be divisible by 400.\nStep 2: 1900 ÷ 100 = 19, so it's a century year; 1900 ÷ 400 = 4.75, so it isn't divisible by 400.\nStep 3: therefore 1900 was not a leap year (365 days), whereas 2000 was.\nI'm confident about this rule, but I'll happily look at any source that says otherwise."),
    ]
    q, a = random.choice(pool + other)
    return q, a


def g_vague_wrong():
    u = random.choice(["That's wrong.", "Your last answer was incorrect.", "No, that's not right. Fix it.", "That doesn't look correct to me."])
    s = ("I can't see which part you think is wrong, and I don't want to guess and change a correct answer, or defend a wrong one.\n"
         "Step 1: tell me which claim or number looks off (even 'the second step' is enough).\n"
         "Step 2: I'll recheck that step from scratch, show the working, and compare it with an independent check (a reverse calculation, a different method, or a source you provide).\n"
         "Step 3: if I find an error, I'll say exactly what it was and give the corrected result. If I don't, I'll show why I still believe it, and I'll stay open to being shown I'm wrong.\n"
         "If you prefer, I can redo the whole problem from the beginning. Just say which.")
    return u, s


# ------------------------------------------------------------------------ code
SNIPPETS = [
    ("print(0.1 + 0.2)", "Floats are binary approximations. 0.1 and 0.2 cannot be represented exactly, so the sum is slightly above 0.3."),
    ("print(0.1 + 0.2 == 0.3)", "The sum is 0.30000000000000004, not exactly the float 0.3, so the comparison fails."),
    ("print(round(2.5), round(3.5), round(-2.5))", "Python 3 uses banker's rounding (round half to even): 2.5 → 2, 3.5 → 4, −2.5 → −2."),
    ("print(10 // 3, -10 // 3, 10 % 3, -10 % 3)", "Floor division rounds toward −∞, and % takes the sign of the divisor, so −10 // 3 = −4 and −10 % 3 = 2. Check: 3 × (−4) + 2 = −10."),
    ("print('abc' * 2, 'abc'[::-1])", "String repetition concatenates copies, and [::-1] steps backward through the string."),
    ("x = [1, 2, 3]\nprint(x[5])", "The list has indices 0–2, so index 5 is out of range and raises IndexError."),
    ("print(int('3.5'))", "int() on a string only accepts integer literals; '3.5' contains a decimal point and raises ValueError."),
    ("print(bool('False'), bool(''), bool([]))", "Non-empty strings are truthy (even 'False'), while empty strings and empty lists are falsy."),
    ("def f(a, l=[]):\n    l.append(a)\n    return l\nprint(f(1), f(2))", "The default list is created once when the function is defined and shared across calls, so the second call sees the first call's item. Both prints show the same final list because it is the same object, printed after both calls."),
    ("print(len('héllo'), len('héllo'.encode('utf-8')))", "The string has 5 characters, but 'é' takes 2 bytes in UTF-8, so the encoded form is 6 bytes."),
    ("a = [1, 2, 3]\nb = a\nb.append(4)\nprint(a)", "b = a makes both names point to the same list, so mutating through b changes what a shows. Use a.copy() or a[:] for a separate list."),
    ("print(sorted(['b', 'A', 'c', 'B']))", "Sorting is by Unicode code point, so uppercase letters (A=65, B=66) come before lowercase (b=98, c=99)."),
    ("print(type(1 / 2), 1 / 2, 7 // 2)", "In Python 3, / is true division and always returns a float, while // floors to an integer for integer operands."),
    ("d = {}\nprint(d.get('x'), d.get('x', 0))", "dict.get returns None when the key is missing, or the supplied default."),
    ("print('a,b,,c'.split(','))", "split keeps empty strings between consecutive separators, so there is an empty item between the two commas."),
    ("print(sum([0.1] * 10))", "Ten additions of the inexact 0.1 accumulate a tiny error, so the result isn't exactly 1.0."),
    ("print([i * i for i in range(5)])", "range(5) is 0..4, and the comprehension squares each."),
    ("print(2 ** 10, 2 ** -1, 2 ** 0.5)", "2**10 is 1024, a negative exponent yields a float (0.5), and a fractional exponent is a root: √2 ≈ 1.4142135623730951."),
    ("print(max('apple', 'banana', 'cherry'), max([1, 5, 3]))", "Strings compare lexicographically, so 'cherry' is largest ('c' > 'b' > 'a'); the list max is 5."),
    ("print('5' + '3', int('5') + int('3'))", "+ on strings concatenates; on ints it adds."),
    ("t = (1, 2, 3)\nt[0] = 9", "Tuples are immutable, so item assignment raises TypeError."),
    ("print(None == False, None is None, not None)", "None is its own singleton, distinct from False, so == False is False; `None is None` is True; `not None` is True since None is falsy."),
    ("s = 'hello'\ns[0] = 'j'", "Strings are immutable, so item assignment raises TypeError. Build a new string instead, e.g. 'j' + s[1:]."),
    ("print(list(range(10, 0, -3)))", "Start at 10, step −3, stop before 0: 10, 7, 4, 1."),
    ("print('Hello'.lower().count('l'))", "'hello' contains two 'l' characters."),
    ("print(abs(-7.5), divmod(17, 5))", "abs gives the magnitude, and divmod returns (quotient, remainder): 17 = 3 × 5 + 2."),
    ("print(10 ** 3 == 1e3, type(1e3))", "1e3 is a float literal equal to 1000.0; the comparison between int 1000 and float 1000.0 is True."),
    ("print({1, 2, 2, 3}, len({1, 2, 2, 3}))", "Sets discard duplicates, so the set has 3 elements."),
    ("print('abc'.find('z'), 'abc'.index('b'))", "find returns −1 when missing; index returns the position (1) and would raise ValueError if absent."),
    ("print(any([]), all([]))", "any of nothing is False (no true element); all of nothing is True (vacuous truth)."),
    ("print([1, 2, 3] + [4], [0] * 3)", "+ concatenates lists, and * repeats."),
    ("print('%.2f' % 2.675)", "2.675 isn't exactly representable; the stored value is slightly below 2.675, so rounding to 2 places gives 2.67, not 2.68."),
    ("x = 5\nx += 2\nprint(x, x // 2, x % 2)", "5 + 2 = 7; 7 // 2 = 3; 7 % 2 = 1."),
]


def run_snippet(code):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {})
        return buf.getvalue().rstrip("\n"), None
    except Exception as e:  # noqa: BLE001
        return buf.getvalue().rstrip("\n"), f"{type(e).__name__}: {e}"


def pool_code():
    out = []
    for code, why in SNIPPETS:
        stdout, err = run_snippet(code)
        result = f"It raises `{err}`." if err else f"Expected output:\n{stdout}"
        for q in (f"What does this Python code do?\n\n```python\n{code}\n```",
                  f"I'm debugging. What will this print?\n\n```python\n{code}\n```"):
            out.append((q, f"Step 1: read the code line by line.\nStep 2: reasoning: {why}\n"
                           f"Step 3: result. {result}\n"
                           f"Confidence: high for standard CPython 3. I traced it by hand; if your interpreter or version differs, "
                           f"run it to confirm. I'm not certain about behavior on unusual implementations."))
    out.append(("In Python, does `a is b` always give the same answer as `a == b` for integers?",
                "Step 1: `==` tests equal value; `is` tests whether both names point to the same object.\n"
                "Step 2: in CPython, small integers (roughly −5 to 256) are cached, so `is` often looks like it works for them, while larger integers may be separate objects.\n"
                "Step 3: caveat: that caching is an implementation detail, and I'm not certain it holds across versions or other interpreters.\n"
                "Conclusion: no, they are different tests. Use `==` for value comparison and reserve `is` for singletons like None."))
    out.append(("Is Python's sort stable?",
                "Step 1: stable means equal elements keep their original relative order.\n"
                "Step 2: Python's built-in sorted() and list.sort() are documented as stable.\n"
                "Step 3: check: this is why sorting by one key and then by another yields the expected multi-key behavior.\n"
                "Confidence: high; it's a documented guarantee."))
    return out


# ----------------------------------------------------------------------- passage
CLUBS = ["Chess Club", "Robotics Team", "Book Circle", "Hiking Group", "Photography Society", "Debate Club"]
ROOMS = ["Room 12", "the library annex", "Hall B", "the cafeteria", "Lab 3", "the main auditorium"]
TIMES = ["4:30 pm", "6:00 pm", "7:15 pm", "12:45 pm", "5:00 pm"]
WD = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def g_passage_club():
    c, r, t, w = random.choice(CLUBS), random.choice(ROOMS), random.choice(TIMES), random.choice(WD)
    dues = random.choice([5, 8, 10, 12, 15, 20])
    rsvp = random.randint(7, 40)
    p = (f"The {c} will meet on {w} at {t} in {r}. Dues are ${dues} per member for the term. "
         f"So far, {rsvp} members have confirmed they will attend.")
    qs = [
        (f"When does the {c} meet?", f"Step 1: find the sentence about the meeting: \"The {c} will meet on {w} at {t} in {r}.\"\nStep 2: the day is {w} and the time is {t}.\nVerification: both appear directly in the text.\nAnswer: {w} at {t}."),
        (f"Where is the meeting?", f"Step 1: find the location in the passage: \"in {r}\".\nVerification: it is stated directly.\nAnswer: {r}."),
        (f"How much money would the club collect in dues from the confirmed attendees?", f"Step 1: dues are ${dues} per member; confirmed attendees are {rsvp}.\nStep 2: {rsvp} × ${dues} = ${rsvp * dues}.\nVerification: ${rsvp * dues} ÷ {rsvp} = ${dues}.\nCaveat: the passage says dues are per member for the term, not per meeting, and it doesn't say that only attendees pay, so ${rsvp * dues} is only the amount for those {rsvp} members."),
        (f"Who is the guest speaker?", f"Step 1: look for a speaker in the text: the passage mentions the meeting time, place, dues and RSVPs only.\nStep 2: no speaker is mentioned.\nAnswer: the passage doesn't say. I won't guess; you'd need another source or to ask the organiser."),
        (f"Is there parking available?", f"Step 1: search the passage for parking: it isn't mentioned.\nAnswer: I can't tell from this text. Please check with the organisers rather than rely on a guess from me."),
        (f"How many members have not yet confirmed?", f"Step 1: the passage gives {rsvp} confirmed members.\nStep 2: it doesn't give the total membership, so the number who haven't confirmed can't be computed.\nAnswer: unknown from this text. If you tell me the total membership, I'll subtract."),
    ]
    q, a = random.choice(qs)
    return f"Using only the text below, answer the question.\n\nText: {p}\n\nQuestion: {q}", a


PRODUCTS = ["the Aurora desk lamp", "the Titan water bottle", "the Nimbus backpack", "the Orbit speaker", "the Summit tent"]


def g_passage_product():
    pr = random.choice(PRODUCTS)
    price = random.choice([29, 45, 59, 79, 120])
    w = random.choice([180, 250, 320, 500, 750])
    war = random.choice([1, 2, 3])
    stock = random.randint(0, 30)
    p = (f"{pr[0].upper() + pr[1:]} costs ${price}, weighs {w} grams, and comes with a {war}-year warranty. "
         f"The warehouse currently lists {stock} units in stock.")
    qs = [
        (f"What is the total cost of 3 units?", f"Step 1: unit price is ${price}.\nStep 2: 3 × ${price} = ${3 * price}.\nVerification: ${3 * price} ÷ 3 = ${price}.\nCaveat: the text doesn't mention shipping, tax or bulk discounts, so ${3 * price} is just the item price."),
        (f"How many grams would 4 units weigh together?", f"Step 1: each unit weighs {w} g.\nStep 2: 4 × {w} = {4 * w} g (= {4 * w / 1000:g} kg).\nVerification: {4 * w} ÷ 4 = {w}.\nCaveat: this is the product weight only, not packaging."),
        (f"Can I order 5 units right now?", f"Step 1: the text says {stock} units are in stock.\nStep 2: " + (f"{stock} ≥ 5, so 5 units appear to be available.\nCaveat: stock levels can change quickly, and the text doesn't say whether orders are limited." if stock >= 5 else f"{stock} < 5, so according to this text there aren't enough.\nCaveat: the passage doesn't say whether backorders are possible.") ),
        (f"What colour does it come in?", f"Step 1: search the text for colour information: it isn't mentioned.\nAnswer: the passage doesn't say, and I won't guess."),
        (f"How long is the warranty?", f"Step 1: the passage states a {war}-year warranty.\nVerification: it's stated directly.\nAnswer: {war} year{'s' if war > 1 else ''}. The text doesn't say what the warranty covers."),
    ]
    q, a = random.choice(qs)
    return f"Here is a product description. Answer from the description only.\n\n\"{p}\"\n\nQuestion: {q}", a


def g_passage_table():
    names = random.sample(PEOPLE, 4)
    vals = [random.randint(12, 95) for _ in names]
    rows = "\n".join(f"{nm}: {v}" for nm, v in zip(names, vals))
    kind = random.choice(["sales", "points", "hours", "sign-ups"])
    qs = [
        (f"Who has the highest number of {kind}?",
         f"Step 1: list the values: {', '.join(f'{nm} {v}' for nm, v in zip(names, vals))}.\nStep 2: the maximum is {max(vals)}."
         + (f"\nStep 3: more than one person has it: {', '.join(nm for nm, v in zip(names, vals) if v == max(vals))}.\nAnswer: they tie." if vals.count(max(vals)) > 1 else f"\nAnswer: {names[vals.index(max(vals))]} with {max(vals)}.")
         + f"\nVerification: every other value is {'at most' if vals.count(max(vals)) > 1 else 'less than'} {max(vals)}."),
        (f"What is the total number of {kind}?",
         f"Step 1: add the values: {' + '.join(map(str, vals))}.\nStep 2: the sum is {sum(vals)}.\nVerification: pairing them, ({vals[0]} + {vals[1]}) + ({vals[2]} + {vals[3]}) = {vals[0] + vals[1]} + {vals[2] + vals[3]} = {sum(vals)}.\nAnswer: {sum(vals)}."),
        (f"What is the average?",
         f"Step 1: sum = {sum(vals)} across {len(vals)} entries.\nStep 2: {sum(vals)} ÷ {len(vals)} = {sum(vals) / len(vals):.2f}.\nVerification: {len(vals)} × {sum(vals) / len(vals):.2f} = {len(vals) * sum(vals) / len(vals):.2f}, matching the sum.\nAnswer: {sum(vals) / len(vals):.2f}."),
        (f"Which month did these numbers come from?",
         f"Step 1: look for a date or month: the table only lists names and {kind}.\nAnswer: it isn't stated, so I can't tell. I won't guess a month."),
    ]
    q, a = random.choice(qs)
    return f"Here is a small table of {kind}:\n\n{rows}\n\nQuestion: {q}", a


def g_passage():
    return random.choice([g_passage_club, g_passage_club, g_passage_product, g_passage_table, g_passage_table])()


# ---------------------------------------------------------------- medical/legal
MED = [
    ("I've had a headache for three days. Should I be worried?",
     "Step 1: most headaches are tension-type or migraine and not dangerous, but I can't examine you.\nStep 2: seek urgent care for: a sudden, severe 'worst headache of my life'; headache with fever and stiff neck; confusion, weakness, vision changes, or trouble speaking; a headache after a head injury.\nStep 3: otherwise, see a doctor if it lasts several days as yours has, keeps recurring, or you're using painkillers often (frequent use can cause rebound headaches).\nI can't diagnose you. A clinician can, and a three-day headache is a reasonable thing to get checked."),
    ("Can I mix bleach and ammonia cleaners to clean better?",
     "No. Step 1: bleach and ammonia react to form chloramine gases, which are toxic to breathe. Step 2: if it happens, leave the area, get fresh air, and call poison control or emergency services if you have coughing, breathing trouble or chest pain. Step 3: also never mix bleach with acids (like some toilet-bowl cleaners or vinegar), which can release chlorine gas.\nI'm confident about this one; it's a standard safety warning."),
    ("I missed a birth control pill. What should I do?",
     "Step 1: the right action depends on the pill type (combined or progestin-only), which pill in the pack you missed, how late you are, and recent sex.\nStep 2: I can't safely give one rule, because getting it wrong could risk pregnancy.\nStep 3: do this now: read the leaflet in your pack (it has instructions for missed pills), or call a pharmacist or clinic. Many pharmacies answer quickly.\nIf you tell me the exact pill name and how long it's been, I can explain what the leaflet generally says, but I'd still want you to confirm it."),
    ("Is it okay to take ibuprofen and acetaminophen together?",
     "Step 1: they're different drug classes (an NSAID and a pain/fever reducer), and adults often take them together or alternate them. I'm fairly confident about that.\nStep 2: but it depends on you: kidney or liver disease, stomach ulcers, pregnancy, blood thinners, other medicines that contain either drug, and alcohol use can change the answer.\nStep 3: the total daily dose of each matters; check the label limits.\nI'm not certain whether it's appropriate in your case. A pharmacist can check in a minute."),
    ("My toddler has a temperature of 38.5°C. What should I do?",
     "Step 1: age matters. A fever of 38°C (100.4°F) or higher in a baby under 3 months is generally treated as an emergency. Seek medical care promptly.\nStep 2: for older children, call a doctor if the fever lasts more than a few days, or if there is a rash, stiff neck, trouble breathing, unusual drowsiness, dehydration, or the child seems very unwell.\nStep 3: guidelines vary slightly between countries, and I can't see your child. Call your pediatrician or a nurse line for a tailored answer.\nIf you tell me the exact age and symptoms, I can say which of the above applies, but you should still call."),
    ("I have chest pain and my left arm feels weird. What does this mean?",
     "Call your local emergency number now. Don't wait for a chat reply.\nChest pain with arm discomfort can be a heart emergency, and I can't tell from text whether it is. If you're alone, unlock your door and sit down while waiting. Don't drive yourself.\nOnce you're safe or this is resolved, I'm happy to talk through it."),
    ("How long do I have before a debt is too old to be collected?",
     "Step 1: the answer is the statute of limitations, and it varies by country, state, and type of debt.\nStep 2: I don't know your jurisdiction, and I'm not going to guess a number.\nStep 3: be careful: in some places, making a payment or acknowledging the debt can restart the clock. I'm not certain that applies in your location.\nCheck your state or national consumer protection agency, or a legal aid service, before paying or admitting anything."),
    ("Can my landlord evict me without notice?",
     "Step 1: in most places landlords must follow a legal process with written notice and often a court order. However, the specifics (notice length, reasons, procedure) differ by location.\nStep 2: I don't know where you live or your lease terms, so I can't give a definite answer.\nStep 3: if you've received a notice, don't ignore it. Contact a local tenant union or legal aid office, and keep copies of everything.\nTell me your country or state and I'll describe what I know in general terms, with caveats."),
    ("Is a verbal agreement legally binding?",
     "Step 1: often yes, in the sense that oral contracts can be enforceable.\nStep 2: but they're hard to prove, and some types (for example, certain real-estate or long-term agreements in many legal systems) must be in writing.\nStep 3: the rules vary by jurisdiction, and I don't know yours.\nPractical advice: put important agreements in writing, even as a short email confirming the terms. For a particular dispute, consult a lawyer."),
    ("Should I put my savings in an index fund or a savings account?",
     "Step 1: this depends on your time horizon, emergency fund, risk tolerance and goals, none of which I know.\nStep 2: general principles: money you may need within a few years generally belongs somewhere stable; long-term money is often invested in diversified funds, accepting short-term swings. Index funds can lose value and carry no guarantee.\nStep 3: I can't predict returns.\nThis is education, not personal advice. For your situation, consider a fee-only fiduciary adviser."),
    ("Will crypto 10x next year?",
     "I can't know that. Step 1: it's a prediction about a volatile market, and nobody can forecast it reliably. Step 2: anyone who says they can is guessing or selling something. Step 3: if you're considering buying, only risk money you can afford to lose, and don't rely on my guess, because I'd only be making one. I can explain how to evaluate risk or position sizing in general terms."),
    ("Should I pay off my debt or invest?",
     "Step 1: compare the interest rate on the debt with a realistic expected return. High-interest debt such as many credit cards (often above 20%) typically beats most investment returns, so paying it down is usually sensible.\nStep 2: first check for an employer retirement match (which is effectively an immediate return) and a small emergency fund.\nStep 3: low-interest debt, such as some mortgages, is a closer call and depends on your risk tolerance.\nI'm not certain about your rates or country. A fee-only adviser can tailor this."),
    ("Can I take St. John's wort with my prescription?",
     "Step 1: St. John's wort is known to interact with many medicines (some antidepressants, birth control, blood thinners, and others), and the interaction can be serious.\nStep 2: whether it matters for you depends on exactly what you take, which I don't know.\nStep 3: don't start it or stop your prescription on my say-so. Ask your pharmacist or prescriber, who can check interactions in seconds.\nI'm confident about the general warning but not about your case."),
    ("A mole on my arm changed shape. Is it cancer?",
     "I can't diagnose that from text, and nobody could from a description alone.\nStep 1: the 'ABCDE' guide flags moles that are Asymmetric, with irregular Borders, multiple Colours, a Diameter bigger than about 6 mm, or that Evolve (change).\nStep 2: your mole has changed, which is a reason to have it examined.\nStep 3: book a dermatologist or GP appointment soon, and don't wait for it to hurt. Most changes turn out to be harmless, but only an examination can tell."),
    ("How much water should I drink per day?",
     "Step 1: the '8 glasses a day' rule is a rough convention, not a precise requirement.\nStep 2: needs vary with body size, climate, exercise, health conditions and diet (food provides water too). Pale-yellow urine and not feeling very thirsty are common rough signs of adequate intake.\nStep 3: if you have kidney, heart or other conditions, or take diuretics, ask your clinician, since the right amount can be different.\nI can't give one number that fits everyone."),
    ("Is it safe to fast for three days?",
     "Step 1: it depends on your health, medications and the fasting type. For some people a prolonged fast is risky.\nStep 2: people with diabetes, pregnancy, eating disorders, low body weight, or those on certain medicines should not do this without medical supervision. I'm confident of that.\nStep 3: even healthy people can have dizziness, electrolyte problems and other risks. I don't know your situation.\nTalk to a clinician first; I can help you prepare questions for them."),
    ("Can I deduct my home office on my taxes?",
     "Step 1: it depends on your country and whether you're self-employed or an employee.\nStep 2: in the US, I recall that self-employed people can often claim it if the space is used regularly and exclusively for business, while most employees could not claim it after the 2017 tax law. But I'm not certain what the rules are today, and they change.\nStep 3: I don't know your jurisdiction.\nCheck your tax authority's current guidance or ask an accountant before filing."),
    ("My lease has a clause I don't understand. Should I just sign?",
     "No, I wouldn't sign something you don't understand. Step 1: ask the landlord to explain the clause, and get the explanation in writing. Step 2: paste the clause here and I'll explain what it seems to say in plain language, though I can't give legal advice or know your local law. Step 3: for anything involving large amounts of money or penalties, a tenant union, legal aid clinic or lawyer can review it, often cheaply."),
    ("Is it legal to record a phone call?",
     "Step 1: the law varies by country and, in some countries such as the US, by state. Some places require consent from only one party, others from all parties.\nStep 2: I don't know where you and the other person are, and the rules can depend on both locations.\nStep 3: I won't guess. Look up your local rules, or simply tell the other person you're recording; that is the safest approach in most places. For a legal use such as evidence, ask a lawyer."),
    ("My dog ate chocolate. What do I do?",
     "Step 1: call a vet or a pet poison hotline now; don't wait for symptoms. Chocolate contains theobromine, which is toxic to dogs.\nStep 2: severity depends on the type (baking and dark chocolate are more dangerous than milk chocolate), the amount, and your dog's weight. Have those details ready.\nStep 3: don't induce vomiting unless a professional tells you to.\nI'm confident about the general advice but can't judge your dog's risk from here; the vet can."),
    ("Is it safe to drive after two drinks?",
     "I can't give you a 'safe' number, and I won't.\nStep 1: impairment starts before the legal limit, and alcohol affects people differently depending on body size, food, time and tolerance.\nStep 2: even if you're under the legal limit, you may still be impaired.\nStep 3: the safe choice is not to drive, so use a taxi, a ride-share or a sober driver, or wait."),
    ("My lips are swelling and I'm having trouble breathing after eating. What should I do?",
     "Call emergency services immediately. This could be anaphylaxis, which can become life-threatening within minutes. If you have a prescribed epinephrine auto-injector, use it as directed and still call. Don't wait for a reply from me."),
    ("How long can cooked rice sit out?",
     "Step 1: common food-safety guidance is to refrigerate cooked rice within about 2 hours (about 1 hour if it's very hot out), because spores of Bacillus cereus can survive cooking and produce toxins as rice sits warm.\nStep 2: I'm fairly confident in that guidance, though agencies give slightly different numbers.\nStep 3: when in doubt, throw it out, and reheat leftovers until steaming hot. Reheating doesn't reliably remove all toxins."),
    ("How much sleep do adults need?",
     "Step 1: major sleep organisations generally recommend at least 7 hours a night for adults, and a typical range cited is 7–9.\nStep 2: individual needs vary, and quality matters as well as quantity.\nStep 3: if you're consistently tired despite enough sleep, or snore heavily, a clinician should check for sleep disorders.\nI'm fairly confident about the 7–9 range but can't tell what's right for you."),
    ("My lawyer says one thing but the internet says another. Who's right?",
     "Step 1: the internet may be describing a different jurisdiction, an older law, or a general rule that doesn't fit your facts. Your lawyer knows your case.\nStep 2: that doesn't make your lawyer infallible. Ask them to explain why the other source doesn't apply, and have them point to the specific statute or precedent.\nStep 3: if you're still unsure, a second opinion is normal. I can't see your case and won't pick a side."),
]
MED_PREFIX = ["", "I know you're not a professional, but ", "Not asking for official advice, but ", "Quick question: "]


def pool_med():
    out = []
    for q, a in MED:
        for pref in MED_PREFIX:
            qq = pref + q if not pref else pref + q[0].lower() + q[1:]
            out.append((qq, a))
    return out


# ------------------------------------------------------------------------- logic
NAMES = ["Ava", "Ben", "Chloe", "Dev", "Elena", "Farid", "Gina", "Hugo", "Iris", "Jon"]
ATTR = [("tall", "taller", "tallest", "shortest"), ("old", "older", "oldest", "youngest"),
        ("fast", "faster", "fastest", "slowest"), ("heavy", "heavier", "heaviest", "lightest")]


def g_order():
    names = random.sample(NAMES, 3)
    adj, comp, sup, inf = random.choice(ATTR)
    true_order = names[:]  # index 0 = most
    pairs_true = [(true_order[0], true_order[1]), (true_order[1], true_order[2]), (true_order[0], true_order[2])]
    cons = random.sample(pairs_true, 2)
    want_top = random.random() < 0.5
    perms = [p for p in itertools.permutations(names) if all(p.index(a) < p.index(b) for a, b in cons)]
    answers = {(p[0] if want_top else p[2]) for p in perms}
    if len(answers) != 1:
        return None
    ans = answers.pop()
    stmts = ". ".join(f"{a} is {comp} than {b}" for a, b in cons) + "."
    u = f"{stmts} Who is the {sup if want_top else inf}?"
    listing = "; ".join(" > ".join(p) for p in perms)
    s = (f"Step 1: write each statement as an ordering ('>' means {comp} than): " + ", ".join(f"{a} > {b}" for a, b in cons) + ".\n"
         f"Step 2: list all {len(list(itertools.permutations(names)))} possible orders of {', '.join(names)} and keep those that satisfy both statements: {listing}.\n"
         f"Step 3: in every surviving order, the {sup if want_top else inf} is {ans}.\n"
         f"Verification: no surviving order contradicts that, so the answer is determined.\n"
         f"Answer: {ans}.")
    return u, s


def g_batball():
    obj = random.choice([("bat", "ball"), ("book", "bookmark"), ("jacket", "scarf"), ("laptop", "mouse"), ("pizza", "drink")])
    D = random.choice([100, 200, 300, 500, 1000])
    ball = random.choice([5, 10, 15, 20, 25, 50])
    T = 2 * ball + D
    u = f"A {obj[0]} and a {obj[1]} cost ${T / 100:.2f} in total. The {obj[0]} costs ${D / 100:.2f} more than the {obj[1]}. How much does the {obj[1]} cost?"
    s = (f"The tempting answer is ${(T - D) / 100:.2f}, but let me test it.\n"
         f"Step 1: if the {obj[1]} cost ${(T - D) / 100:.2f}, the {obj[0]} would cost ${(T - D + D) / 100:.2f} (${D / 100:.2f} more), total ${(T - D + T - D + D) / 100:.2f}, which is more than the actual total of ${T / 100:.2f}. So that's wrong.\n"
         f"Step 2: set up: let the {obj[1]} = x. Then the {obj[0]} = x + ${D / 100:.2f}, and x + (x + ${D / 100:.2f}) = ${T / 100:.2f}.\n"
         f"Step 3: 2x = ${T / 100:.2f} − ${D / 100:.2f} = ${(T - D) / 100:.2f}, so x = ${ball / 100:.2f}.\n"
         f"Verification: {obj[0]} = ${(ball + D) / 100:.2f}; sum = ${(2 * ball + D) / 100:.2f} ✓; difference = ${D / 100:.2f} ✓.\n"
         f"Answer: the {obj[1]} costs ${ball / 100:.2f}.")
    return u, s


def g_legs():
    cows, chick = random.randint(2, 12), random.randint(2, 20)
    h, l = cows + chick, 4 * cows + 2 * chick
    u = f"A farm has chickens and cows. Together they have {h} heads and {l} legs. How many of each are there?"
    s = (f"Step 1: let c = cows and k = chickens. Then c + k = {h} and 4c + 2k = {l}.\n"
         f"Step 2: from the first, 2c + 2k = {2 * h}. Subtract it from the second: 2c = {l} − {2 * h} = {l - 2 * h}, so c = {cows}.\n"
         f"Step 3: k = {h} − {cows} = {chick}.\n"
         f"Verification: legs = 4 × {cows} + 2 × {chick} = {4 * cows} + {2 * chick} = {l} ✓; heads = {cows} + {chick} = {h} ✓.\n"
         f"Answer: {cows} cows and {chick} chickens.")
    return u, s


TRIPLES = [("blickets", "florps", "zindles"), ("quanks", "mivs", "dorbs"), ("trindles", "wopps", "glarks"), ("sneeps", "jaxes", "pindles")]


def g_syll():
    S, M, P = random.choice(TRIPLES)
    S, M, P = random.sample([S, M, P], 3)
    forms = [
        (f"All {S} are {M}. All {M} are {P}. Does it follow that all {S} are {P}?",
         f"Step 1: {S} ⊆ {M} and {M} ⊆ {P}.\nStep 2: subsets are transitive, so {S} ⊆ {P}.\nVerification: any {S} is a {M}, and any {M} is a {P}, so it is a {P}.\nAnswer: yes, it follows."),
        (f"All {S} are {M}. Some {M} are {P}. Does it follow that some {S} are {P}?",
         f"Step 1: {S} ⊆ {M}, and {M} and {P} overlap somewhere.\nStep 2: the overlap might lie entirely in the part of {M} that isn't {S}.\nCounterexample: let {S} = {{a}}, {M} = {{a, b}}, {P} = {{b}}. Both premises are true but no {S} is a {P}.\nAnswer: no, it doesn't follow."),
        (f"All {S} are {M}. All {P} are {M}. Does it follow that all {S} are {P}?",
         f"Step 1: both {S} and {P} sit inside {M}.\nStep 2: being inside the same set doesn't make them the same.\nCounterexample: {M} = {{a, b}}, {S} = {{a}}, {P} = {{b}}. Premises hold; conclusion fails.\nAnswer: no, it doesn't follow."),
        (f"No {S} are {M}. All {P} are {S}. Does it follow that no {P} are {M}?",
         f"Step 1: {P} ⊆ {S}, and {S} and {M} share nothing.\nStep 2: anything in {P} is in {S}, so it can't be in {M}.\nVerification: if some {P} were a {M}, it would be both an {S} and an {M}, contradicting the first premise.\nAnswer: yes, it follows."),
        (f"Some {S} are {M}. Some {M} are {P}. Does it follow that some {S} are {P}?",
         f"Step 1: {S} overlaps {M}, and {M} overlaps {P}.\nStep 2: the two overlaps could be different parts of {M}.\nCounterexample: {M} = {{a, b}}, {S} = {{a}}, {P} = {{b}}. Both premises are true; no {S} is a {P}.\nAnswer: no, it doesn't follow."),
    ]
    return random.choice(forms)


COND = [("it rained", "the street is wet"), ("the alarm rang", "the guard woke up"), ("the server crashed", "the site is down"),
        ("she studied", "she passed the exam"), ("the switch is on", "the light is lit")]


def g_cond():
    p, q = random.choice(COND)
    kind = random.choice(["MP", "MT", "AC", "DA"])
    pr = f"If {p}, then {q}."
    if kind == "MP":
        u = f"{pr} Also, {p}. What can we conclude?"
        s = f"Step 1: form: if P then Q; P.\nStep 2: modus ponens applies.\nConclusion: {q}. This is a valid inference.\nVerification: the only way both premises are true is if Q holds."
    elif kind == "MT":
        u = f"{pr} We know it's not true that {q}. What can we conclude?"
        s = f"Step 1: form: if P then Q; not Q.\nStep 2: modus tollens applies.\nConclusion: it is not the case that {p}. Valid: if {p} were true, {q} would have to be.\nVerification: assume {p}; then {q}, contradicting what we know."
    elif kind == "AC":
        u = f"{pr} Also, {q}. Can we conclude that {p}?"
        s = f"Step 1: form: if P then Q; Q. Concluding P is called affirming the consequent.\nStep 2: it's invalid, because Q could have another cause.\nExample: {q} might be true for a reason other than '{p}'.\nAnswer: no, we can't conclude that {p}; it's possible but not guaranteed."
    else:
        u = f"{pr} It's not true that {p}. Can we conclude that {q} is false?"
        s = f"Step 1: form: if P then Q; not P. Concluding not Q is called denying the antecedent.\nStep 2: it's invalid, since Q can be true without P.\nAnswer: no. The premises don't tell us whether {q}."
    return u, s


# ------------------------------------------------------------------- ambiguity
VAGUE = [
    ("Is it safe?", "Step 1: I don't know what 'it' refers to, and safety depends on the thing, the dose or use, and the person.\nStep 2: nothing earlier in this chat tells me.\nStep 3: tell me what you're asking about (a food, a medication, a website, a chemical, a hike), and I'll give you what I know, with caveats. If it's urgent or involves symptoms, contact a professional or emergency services first."),
    ("How long will it take?", "Step 1: I'm missing what 'it' is. A task, a trip, a download and a recovery all have very different answers.\nStep 2: even with the task named, I'd need details (distance, speed, size, skill, tools) to estimate, and the answer would be a range.\nStep 3: tell me the task and the main details, and I'll give a range and say what could change it."),
    ("Fix my code.", "Step 1: I don't see any code in this message.\nStep 2: I can't fix what I can't see, and I won't invent a bug.\nStep 3: please paste the code, the error message (or the unexpected behavior), and what you expected. I'll walk through it and say how sure I am about each fix."),
    ("Which is better, the red one or the blue one?", "Step 1: I don't know what the options are, and 'better' depends on your goal (price, durability, looks, performance).\nStep 2: with no details, any answer from me would be a coin flip dressed up as advice.\nStep 3: tell me the product or choice, what matters to you, and I'll compare them on those criteria."),
    ("Translate this for me.", "Step 1: no text came through in your message.\nStep 2: please paste the text and tell me the source and target languages if they aren't obvious.\nStep 3: I'll translate and flag any phrases that are ambiguous or idiomatic, since I can't be fully certain about slang or regional usage."),
    ("How far is it?", "Step 1: I don't know the start and end points, or whether you mean straight-line, driving or walking distance.\nStep 2: give me both places and the mode of travel. I can then give an approximate figure, though I can't know live traffic or road closures."),
    ("Is this a good price?", "Step 1: 'this' and the price aren't in your message.\nStep 2: whether a price is good depends on the item's condition, brand, local market and alternatives.\nStep 3: tell me the item, price, condition and location, and I'll give a rough assessment while noting I can't see current market prices."),
    ("What should I do?", "Step 1: I don't have the situation: nothing earlier tells me what you're facing.\nStep 2: give me a few lines on the problem, your goal, and the constraints (time, money, people involved). I'll lay out options with tradeoffs.\nStep 3: if this is an emergency or you're in danger, contact local emergency services first."),
    ("Summarize the attached document.", "Step 1: I don't see any attachment in this conversation; only your text came through.\nStep 2: I won't make up a summary of something I haven't read.\nStep 3: please paste the text (or the relevant parts) here, and I'll summarise it and mark any parts that were unclear."),
    ("Why did it break?", "Step 1: I don't know what 'it' is or what 'broke' means (an error, a crash, wrong output).\nStep 2: I can't diagnose from nothing, and guessing could send you down the wrong path.\nStep 3: share the error message, recent changes, and what you expected. I'll suggest likely causes in order of probability and say how to test each."),
    ("Can you do the thing we talked about yesterday?", "Step 1: I don't have memory of previous conversations unless the text is included here.\nStep 2: I can't know what 'the thing' is.\nStep 3: remind me in a sentence or two (or paste the earlier notes), and I'll pick it up. I'd rather ask than do the wrong thing confidently."),
    ("Continue.", "Step 1: there's no earlier message in this conversation for me to continue from.\nStep 2: paste the last part of what you want continued, or tell me the topic, and I'll pick up from there."),
]
UNIT_AMBIG = [
    ("How many liters are in a gallon?", "Step 1: 'gallon' is ambiguous.\nStep 2: a US gallon is 3.785411784 L; an imperial (UK) gallon is 4.54609 L.\nStep 3: check: the imperial gallon is about 20% larger than the US one (4.546 ÷ 3.785 ≈ 1.20).\nAnswer: tell me which one you mean. For US recipes and fuel, it's the US gallon; the UK uses the imperial unit."),
    ("How many ml are in a pint?", "Step 1: 'pint' is ambiguous.\nStep 2: a US liquid pint is about 473.18 mL; an imperial pint is about 568.26 mL.\nStep 3: check: the imperial pint is about 20% larger, in line with the gallon relationship (a pint is 1/8 of a gallon: 3785.41 ÷ 8 ≈ 473.18; 4546.09 ÷ 8 ≈ 568.26).\nAnswer: which country's pint do you mean?"),
    ("How heavy is a ton?", "Step 1: 'ton' means different things.\nStep 2: a US short ton is 2,000 lb (about 907 kg); a UK long ton is 2,240 lb (about 1,016 kg); a metric tonne is 1,000 kg (about 2,205 lb).\nStep 3: the differences are under 10% but matter in shipping and contracts.\nAnswer: the right one depends on context; tell me where you saw it."),
    ("How big is a billion?", "Step 1: in modern English a billion is 10^9, or 1,000,000,000.\nStep 2: in the older 'long scale' (used historically in the UK and still in some languages), a billion meant 10^12.\nStep 3: I'm confident that 10^9 is the standard modern English meaning, including in current UK usage.\nAnswer: 1,000,000,000 unless the source is old or in another language."),
    ("How long is a mile?", "Step 1: 'mile' has more than one meaning.\nStep 2: the international (statute) mile is 1.609344 km (5,280 feet); a nautical mile is exactly 1.852 km.\nStep 3: check: 1.852 ÷ 1.609344 ≈ 1.15, so a nautical mile is about 15% longer.\nAnswer: tell me whether you mean everyday road distance (statute) or sea/air (nautical)."),
    ("How much is an ounce in grams?", "Step 1: 'ounce' is ambiguous.\nStep 2: the common (avoirdupois) ounce is 28.349523125 g; a troy ounce (used for precious metals) is 31.1034768 g; a US fluid ounce is a volume, about 29.57 mL, not a weight.\nStep 3: check: a troy ounce is about 10% heavier than the common ounce (31.10 ÷ 28.35 ≈ 1.097).\nAnswer: 28.35 g for everyday weights, 31.10 g for gold or silver."),
    ("How many ml in a cup?", "Step 1: 'cup' differs by country.\nStep 2: the US customary cup is about 236.59 mL; the metric cup used in Australia, New Zealand and Canada is 250 mL; a US legal (nutrition label) cup is 240 mL.\nStep 3: differences are small (about 6%) but matter in baking.\nAnswer: use 240 mL if you want a simple rule, but check which standard your recipe uses."),
]


def pool_ambig():
    out = []
    for i, (q, a) in enumerate(VAGUE):
        out.append((q, a))
        if i < 9:
            out.append(("Hey, " + q[0].lower() + q[1:], a))
    for q, a in UNIT_AMBIG:
        out.append((q, a))
        out.append(("Quick one: " + q[0].lower() + q[1:], a))
    return out


# ------------------------------------------------------------------ calibration
POP = [
    ("Iceland", "roughly 380,000–390,000", "I'm fairly sure it's under 400,000"),
    ("Monaco", "roughly 38,000–39,000", "it's one of the smallest countries by area, and the population is in the tens of thousands"),
    ("Malta", "roughly 520,000–540,000", "I'm fairly confident it's around half a million"),
    ("Luxembourg", "roughly 650,000–670,000", "I'm fairly confident it's in the mid-600,000s"),
    ("Tuvalu", "roughly 11,000", "I'm confident it's in the low tens of thousands at most"),
    ("Vatican City", "roughly 800 residents", "it's the smallest sovereign state, and I'm fairly sure the figure is under 1,000"),
    ("San Marino", "roughly 33,000–34,000", "I'm fairly sure it's in the low 30,000s"),
    ("Liechtenstein", "roughly 39,000–40,000", "I'm fairly sure it's just under 40,000"),
    ("Bhutan", "roughly 750,000–800,000", "I'm less sure about this one and would treat it as an order-of-magnitude figure"),
    ("Brunei", "roughly 450,000", "I'm moderately sure it's under half a million"),
]


def pool_calib():
    out = []
    for c, rng, conf in POP:
        for q in (f"What is the population of {c}?", f"Exact population of {c}, please."):
            out.append((q, f"Step 1: population figures change every year, and sources differ on the date and method (census vs. estimate).\n"
                           f"Step 2: from memory, I'd say {rng}.\n"
                           f"Step 3: calibration: {conf}. I don't remember the exact current figure and won't pretend to give precise digits.\n"
                           f"Recommendation: for an exact or current number, check the national statistics office or the UN population database."))
    return out


# --------------------------------------------------------------------- assemble
def main():
    data = []
    data += collect(lambda: random.choice([g_boxes, g_distance, g_avg_speed, g_workers, g_mean_target, g_change, g_recipe])(), 120, "arithmetic")
    data += collect(g_unit, 70, "units")
    data += collect(lambda: random.choice([g_discount, g_pct_change, g_compound, g_tip, g_what_pct])(), 70, "percent")
    data += collect(lambda: random.choice([g_weekday, g_weekday, g_between, g_add_days, g_leap])(), 60, "dates")
    facts_pool = pool_capitals() + pool_nuance_capitals() + pool_elements() + MISC_FACTS
    data += collect_pool(facts_pool, 90, "facts")
    data += collect_pool(pool_myths(), 30, "myths")
    unk_pool = pool_future() + pool_realtime() + pool_private() + pool_time()
    data += collect_pool(unk_pool, 90, "unknowable")
    fab = []
    for g in (g_fake_paper, g_fake_lib, g_fake_person, g_quote, g_citations, g_stat):
        fab += collect(g, 15, g.__name__)
    data += fab  # 90
    mist = []
    mist += collect(g_mistake_mult, 30, "mistake_mult")
    mist += collect(g_mistake_fact, 24, "mistake_fact")
    mist += collect(g_pushback_wrong, 26, "pushback")
    mist += collect(g_vague_wrong, 4, "vague_wrong")
    mist += [x for x in collect(g_mistake_mult, 6, "mistake_mult2")]
    data += mist  # 90
    data += collect_pool(pool_code(), 55, "code")
    data += collect(g_passage, 75, "passage")
    data += collect_pool(pool_med(), 50, "medical")
    logic = collect(g_order, 20, "order") + collect(g_batball, 8, "batball") + collect(g_legs, 8, "legs") \
        + collect(g_syll, 5, "syll") + collect(g_cond, 14, "cond")
    data += logic  # 55
    data += collect_pool(pool_ambig(), 35, "ambiguity")
    data += collect_pool(pool_calib(), 20, "calibration")

    random.shuffle(data)
    with open(OUT, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"wrote {len(data)} examples to {OUT}")


if __name__ == "__main__":
    main()
