"""sb_aicall scenarios B (voicemail), C (automated system) and D (nobody): plans, track assembly and labels.

Sub-scenarios: B pvm_out / cvm_out; C robo_in / auto_in / ivr_out; D ring_out / dead_in."""
from __future__ import annotations

import random

import numpy as np

import aic_scripts as S

SR = 16000
SUBS = {"B": ["pvm_out", "cvm_out"], "C": ["robo_in", "auto_in", "ivr_out"], "D": ["ring_out", "dead_in"]}
SUB_W = {"pvm_out": 0.6, "cvm_out": 0.4, "robo_in": 0.45, "auto_in": 0.2, "ivr_out": 0.35, "ring_out": 0.5,
         "dead_in": 0.5}


def plan_clip(key: str, scenario: str, split: str, rng: random.Random, voices: list[str], robo_keys: list[str]):
    subs = SUBS[scenario]
    sub = rng.choices(subs, weights=[SUB_W[s] for s in subs])[0]
    av, ov = rng.sample(voices, 2)
    ctx: dict = {}
    agent, other = [], []  # texts (TTS) in order of use
    p = {"key": key, "scenario": scenario, "sub": sub, "split": split, "direction": sub.split("_")[1],
         "agent_voice": av, "other_voice": ov, "agent_speed": round(rng.uniform(0.9, 1.1), 3),
         "other_speed": round(rng.uniform(0.9, 1.1), 3), "seed": rng.randrange(2 ** 31),
         "phone_fx": rng.random() < 0.7, "gated": rng.random() < 0.7, "robo": None}
    if sub in ("pvm_out", "cvm_out"):
        other = S.other_script("pvm" if sub == "pvm_out" else "cvm", rng)
        p["ring"] = rng.random() < 0.5
        p["talkover"] = rng.random() < 0.3
        p["beep"] = rng.random() < 0.65
        p["message"] = rng.random() < 0.6
        if p["talkover"]:
            agent.append(S.fill(rng.choice(S.TALKOVER_VM), rng, ctx))
        if p["message"]:
            agent.append(S.fill(rng.choice(S.VM_MESSAGE), rng, ctx))
    elif sub == "robo_in":
        p["robo"] = rng.choice(robo_keys)
        agent.append(S.fill(rng.choice(S.INBOUND_GREET), rng, ctx))
        p["n_reply"] = rng.choice([0, 1, 1, 2])
        for _ in range(p["n_reply"]):
            agent.append(S.fill(rng.choice(S.ROBO_REPLY), rng, ctx))
        p["robo_len"] = rng.uniform(8, 35)
        p["robo_during_greet"] = rng.random() < 0.4
    elif sub == "auto_in":
        agent.append(S.fill(rng.choice(S.INBOUND_GREET), rng, ctx))
        other.append(S.fill(rng.choice(S.AUTO_CALLER), rng, ctx))
        if rng.random() < 0.4:
            other.append(S.fill(rng.choice(S.AUTO_CALLER), rng, ctx))
        p["n_reply"] = rng.choice([0, 1, 1, 2])
        for _ in range(p["n_reply"]):
            agent.append(S.fill(rng.choice(S.ROBO_REPLY), rng, ctx))
        p["robo_during_greet"] = rng.random() < 0.4
    elif sub == "ivr_out":
        other = S.other_script("ivr", rng)
        p["ring"] = rng.random() < 0.5
        p["nav_during"] = rng.random() < 0.35
        agent.append(S.fill(rng.choice(S.IVR_NAV), rng, ctx))
        if rng.random() < 0.3:
            agent.append(S.fill(rng.choice(S.IVR_NAV), rng, ctx))
    elif sub == "ring_out":
        p["ring_s"] = rng.uniform(8, 28)
        p["connect"] = rng.random() < 0.7
        if p["connect"]:
            for _ in range(rng.choice([1, 2, 2])):
                agent.append(S.fill(rng.choice(S.ARE_YOU_THERE), rng, ctx))
            if rng.random() < 0.6:
                agent.append(S.fill(rng.choice(S.GOODBYE_NOBODY), rng, ctx))
    elif sub == "dead_in":
        agent.append(S.fill(rng.choice(S.INBOUND_GREET), rng, ctx))
        for _ in range(rng.choice([1, 2, 2])):
            agent.append(S.fill(rng.choice(S.ARE_YOU_THERE), rng, ctx))
        if rng.random() < 0.6:
            agent.append(S.fill(rng.choice(S.GOODBYE_NOBODY), rng, ctx))
    p["agent_lines"] = agent
    p["other_lines"] = other
    return p


# Assembly

def _norm(x, peak=0.5):
    m = float(np.abs(x).max()) if len(x) else 0.0
    return (x * (peak / m)).astype(np.float32) if m > 0 else x.astype(np.float32)


class Track:
    def __init__(self):
        self.parts = []  # (start_s, array)

    def put(self, t: float, x: np.ndarray):
        self.parts.append((t, np.asarray(x, np.float32)))
        return t + len(x) / SR

    def render(self, T: float) -> np.ndarray:
        n = int(round(T * SR))
        y = np.zeros(n, np.float32)
        for t, x in self.parts:
            i = int(round(t * SR))
            if i >= n:
                continue
            L = min(len(x), n - i)
            y[i:i + L] += x[:L]
        return y


def assemble(p: dict, agent_audio: list, other_audio: list, robo: np.ndarray | None):
    """agent_audio / other_audio: 16 kHz arrays aligned with p['agent_lines'] / p['other_lines'] (trimmed).
    Returns (agent_track, other_track_raw (before phone fx), T, facts)."""
    import audio_fx as FX  # data/sb_calls/audio_fx.py

    rng = random.Random(p["seed"])
    nrng = np.random.default_rng(p["seed"])
    A, O = Track(), Track()
    sp = {"agent": [], "other": []}
    facts = {"sub": p["sub"]}
    lvl_o = 0.5 * 10 ** (rng.uniform(-6, 4) / 20)
    t = rng.uniform(0.2, 1.0)
    sub = p["sub"]

    def say(track, role, t0, x, lvl=0.5):
        x = _norm(x, lvl)
        t1 = track.put(t0, x)
        sp[role].append((t0, t1))
        return t1

    if sub in ("pvm_out", "cvm_out", "ivr_out") and p.get("ring"):
        rb, _v = FX.ringback(nrng, float(rng.uniform(2.5, 9.0)))
        t = O.put(t, _norm(rb, 0.4 * lvl_o / 0.5)) + rng.uniform(0.2, 0.8)
        facts["ring"] = True
    if sub in ("pvm_out", "cvm_out"):
        ai = 0
        g0 = t
        for k, x in enumerate(other_audio):
            if k > 0:
                t += rng.uniform(0.15, 0.6)
            t = say(O, "other", t, x, lvl_o)
            if k == 0 and p["talkover"]:
                ta = t + rng.uniform(0.2, 0.8) - (rng.uniform(0.0, 0.6) if len(other_audio) > 1 else 0.0)
                say(A, "agent", ta, agent_audio[ai])
                ai += 1
        g_end = t
        facts["greet"] = [round(g0, 3), round(g_end, 3)]
        msg_from = g_end
        if p["beep"]:
            t += rng.uniform(0.15, 0.8)
            b = FX.beep(nrng, rng.uniform(0.3, 1.0))
            b0 = t
            t = O.put(t, _norm(b, lvl_o * rng.uniform(0.5, 1.0)))
            facts["beep"] = [round(b0, 3), round(t, 3)]
            msg_from = t
        # the agent's talk-over line may still be running: the message starts after both
        t = max(t, max((e for _, e in sp["agent"]), default=0.0))
        if p["message"]:
            t += rng.uniform(0.4, 1.5)
            t = say(A, "agent", t, agent_audio[ai])
        else:
            t += rng.uniform(0.5, 3.0)
        after = sum(max(0.0, e - max(s, msg_from)) for s, e in sp["agent"])
        facts["agent_after_greeting_s"] = round(after, 3)
        facts["beep_present"] = bool(p["beep"])
    elif sub in ("robo_in", "auto_in"):
        t_g = say(A, "agent", t, agent_audio[0])
        if p["robo_during_greet"]:
            t0 = t + rng.uniform(0.5, 2.0)
        else:
            t0 = t_g + rng.uniform(0.3, 2.0)
        if sub == "robo_in":
            L = int(min(len(robo), p["robo_len"] * SR))
            off = 0 if (rng.random() < 0.5 or len(robo) <= L) else rng.randrange(0, len(robo) - L)
            seg = robo[off:off + L].copy()
            f = min(len(seg) // 2, int(0.05 * SR))
            if f > 0:
                seg[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
            t_o = say(O, "other", t0, seg, lvl_o)
        else:
            t_o = t0
            for k, x in enumerate(other_audio):
                if k > 0:
                    t_o += rng.uniform(0.4, 1.2)
                t_o = say(O, "other", t_o, x, lvl_o)
        t = max(t_g, t_o)
        for k in range(1, len(agent_audio)):
            if k == 1 and rng.random() < 0.3 and t_o - t0 > 6:
                ta = t0 + rng.uniform(2.0, t_o - t0 - 2.0)
                say(A, "agent", ta, agent_audio[k])
            else:
                t += rng.uniform(0.5, 2.0)
                t = say(A, "agent", t, agent_audio[k])
            t = max(t, t_o)
        t = max([t] + [e for _, e in sp["agent"]])
    elif sub == "ivr_out":
        m0 = t
        nav_at = None
        if p["nav_during"] and len(other_audio) >= 2:
            nav_at = rng.randrange(1, len(other_audio))
        ai = 0
        for k, x in enumerate(other_audio):
            if k > 0:
                t += rng.uniform(0.2, 0.7)
            if nav_at == k and ai < len(agent_audio):
                say(A, "agent", t + rng.uniform(-0.3, 0.5), agent_audio[ai])
                ai += 1
            t = say(O, "other", t, x, lvl_o)
        t = max([t] + [e for _, e in sp["agent"]])
        while ai < len(agent_audio):
            t += rng.uniform(0.5, 2.0)
            t = say(A, "agent", t, agent_audio[ai])
            ai += 1
        facts["menu"] = [round(m0, 3), round(t, 3)]
    elif sub == "ring_out":
        rb, _v = FX.ringback(nrng, p["ring_s"])
        t = O.put(t, _norm(rb, 0.4 * lvl_o / 0.5))
        for x in agent_audio:
            t += rng.uniform(1.0, 4.0)
            t = say(A, "agent", t, x)
    elif sub == "dead_in":
        t = say(A, "agent", t, agent_audio[0])
        for x in agent_audio[1:]:
            t += rng.uniform(2.5, 6.0)
            t = say(A, "agent", t, x)
    T = min(60.0, t + rng.uniform(0.4, 1.5))
    T = max(T, 8.0)
    a = A.render(T)
    o = O.render(T)
    if not p["gated"]:
        o = o + FX.line_noise(nrng, rng.choice(["white", "pink", "hum"]), len(o), rng.uniform(-62, -45))
    facts["T"] = round(T, 3)
    facts["speech"] = {r: [(round(s, 3), round(min(e, T), 3)) for s, e in v if s < T] for r, v in sp.items()}
    return a, o, T, facts


def labels(p: dict, facts: dict) -> dict:
    """-> {family: [(answer, None)]} for synthetic clips."""
    sc, sub = p["scenario"], p["sub"]
    live = {"B": "voicemail", "C": "automated", "D": "nobody"}[sc]
    fam = "live_in" if p["direction"] == "in" else "live_out"
    out = {fam: [("no", None)], "live_choice": [(live, None)]}
    if sc == "B":
        out["tone"] = [("yes" if facts["beep_present"] else "no", None)]
        after = facts["agent_after_greeting_s"]
        if after >= 3.0:
            out["vm_left"] = [("yes", None)]
        elif after <= 0.0:
            out["vm_left"] = [("no", None)]
    if sc == "C":
        out["auto_in" if p["direction"] == "in" else "auto_out"] = [("yes", None)]
    return out
