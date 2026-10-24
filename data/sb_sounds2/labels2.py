"""Label logic for sb_sounds2 (no audio I/O): FSD50K hard negatives, FMA / OpenMIC-2018 genre, instrument and tempo
rules, and question paraphrases.
"""

from __future__ import annotations

import math
import random
import re

import numpy as np

import fsd_labels as FL  # copy of data/sb_sounds/labels.py (class table, Taxonomy, co-occurrence, levels)

# Part 1. FSD50K hard negatives
# Classes in one group are never asked as absent for each other: they overlap in meaning or occur together (a running
# tap is pouring water; a car idling is a car; a drum kit contains a snare). Confusable but distinct classes (bark vs
# growl, car vs truck, male vs female speech) are not here: they are the hard negatives.
SEMANTIC_BLOCK: list[set[str]] = [
    {"Water", "Liquid", "Pour", "Fill_(with_liquid)", "Trickle_and_dribble", "Drip", "Splash_and_splatter",
     "Gurgling", "Stream", "Water_tap_and_faucet", "Sink_(filling_or_washing)", "Bathtub_(filling_or_washing)",
     "Toilet_flush", "Boiling"},
    {"Water", "Liquid", "Rain", "Raindrop", "Thunderstorm", "Ocean", "Waves_and_surf", "Stream", "Boat_and_Water_vehicle"},
    {"Rain", "Raindrop", "Thunderstorm", "Thunder"},
    {"Ocean", "Waves_and_surf"},
    {"Human_voice", "Speech", "Conversation", "Chatter", "Crowd", "Speech_synthesizer", "Whispering"},
    {"Conversation", "Chatter", "Crowd", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking",
     "Child_speech_and_kid_speaking", "Human_group_actions", "Cheering", "Applause"},
    {"Shout", "Yell", "Screaming"},
    {"Laughter", "Giggle", "Chuckle_and_chortle"},
    {"Singing", "Male_singing", "Female_singing", "Human_voice"},
    {"Clapping", "Applause", "Hands", "Cheering"},
    {"Breathing", "Sigh", "Gasp", "Respiratory_sounds"},
    {"Chime", "Wind_chime"},
    {"Door", "Slam", "Knock", "Cupboard_open_or_close", "Drawer_open_or_close", "Sliding_door", "Squeak"},
    {"Knock", "Tap", "Thump_and_thud"},
    {"Boom", "Explosion", "Thunder", "Fireworks"},
    {"Engine", "Idling", "Accelerating_and_revving_and_vroom", "Engine_starting", "Car", "Truck", "Bus",
     "Motorcycle", "Race_car_and_auto_racing", "Car_passing_by", "Motor_vehicle_(road)", "Vehicle",
     "Traffic_noise_and_roadway_noise", "Aircraft", "Fixed-wing_aircraft_and_airplane", "Boat_and_Water_vehicle",
     "Vehicle_horn_and_car_horn_and_honking"},
    {"Car", "Car_passing_by", "Race_car_and_auto_racing", "Traffic_noise_and_roadway_noise",
     "Vehicle_horn_and_car_horn_and_honking"},
    {"Truck", "Bus", "Motorcycle", "Traffic_noise_and_roadway_noise", "Vehicle_horn_and_car_horn_and_honking"},
    {"Train", "Rail_transport", "Subway_and_metro_and_underground"},
    {"Drum_kit", "Drum", "Snare_drum", "Bass_drum", "Cymbal", "Crash_cymbal", "Hi-hat", "Percussion"},
    {"Strum", "Guitar", "Acoustic_guitar", "Electric_guitar", "Bass_guitar", "Plucked_string_instrument"},
    {"Music", "Musical_instrument"},
    {"Bird", "Bird_vocalization_and_bird_call_and_bird_song", "Chirp_and_tweet", "Crow", "Gull_and_seagull",
     "Fowl", "Chicken_and_rooster"},
    {"Insect", "Buzz", "Cricket"},
    {"Typing", "Computer_keyboard", "Typewriter"},
    {"Clock", "Tick", "Tick-tock", "Mechanisms", "Ratchet_and_pawl"},
    {"Walk_and_footsteps", "Run"},
    {"Fire", "Crackle"},
    {"Glass", "Chink_and_clink", "Shatter", "Dishes_and_pots_and_pans", "Cutlery_and_silverware", "Coin_(dropping)"},
    {"Bicycle", "Bicycle_bell"},
    {"Tools", "Power_tool", "Drill", "Hammer", "Sawing"},
    {"Domestic_sounds_and_home_sounds", "Door", "Cupboard_open_or_close", "Drawer_open_or_close", "Sliding_door",
     "Dishes_and_pots_and_pans", "Cutlery_and_silverware", "Microwave_oven", "Toilet_flush", "Sink_(filling_or_washing)",
     "Bathtub_(filling_or_washing)", "Water_tap_and_faucet"},
    {"Wind", "Whoosh_and_swoosh_and_swish"},
    {"Crumpling_and_crinkling", "Crushing"},
    {"Alarm", "Siren"},
    {"Telephone", "Ringtone"},
    {"Rattle", "Rattle_(instrument)"},
]


def semantic_blocked(labels) -> set[str]:
    s = set(labels)
    out: set[str] = set()
    for g in SEMANTIC_BLOCK:
        if g & s:
            out |= g
    return out - s


class FullOntology:
    """Direct parents / grandparents of the FSD50K vocabulary classes in the full AudioSet ontology (also through
    nodes that are not in the vocabulary). `vocab`: {mid: name}; `nodes`: AudioSet ontology json."""

    def __init__(self, vocab: dict[str, str], nodes: list[dict]):
        self.mid2name = dict(vocab)
        self.name2mid = {n: m for m, n in vocab.items()}
        self.official = {}
        node_by = {n["id"]: n for n in nodes}
        for m, n in vocab.items():
            self.official[n] = node_by[m]["name"] if m in node_by else n.replace("_", " ")
        self.parents: dict[str, set[str]] = {}
        for n in nodes:
            for c in n.get("child_ids", []):
                self.parents.setdefault(c, set()).add(n["id"])
        self.children: dict[str, set[str]] = {}
        for c, ps in self.parents.items():
            for p in ps:
                self.children.setdefault(p, set()).add(c)

    def _desc_vocab(self, mid: str, depth: int) -> set[str]:
        """Vocabulary classes reachable downward from mid within `depth` steps (through any nodes)."""
        out, frontier = set(), {mid}
        for _ in range(depth):
            nxt = set()
            for f in frontier:
                nxt |= self.children.get(f, set())
            out |= {self.mid2name[x] for x in nxt if x in self.mid2name}
            frontier = nxt
        return out

    def direct_siblings(self, name: str, broad: bool | None = None) -> set[str]:
        """Vocab classes sharing a direct ontology parent with `name`.
        broad=False: only through specific parents; broad=True: only through top-level parents (e.g. "Sounds of
        things", whose children are barely related); None: both."""
        m = self.name2mid[name]
        out = set()
        for p in self.parents.get(m, set()):
            is_broad = not self.parents.get(p)  # a top-level AudioSet category
            if broad is None or broad == is_broad:
                out |= {self.mid2name[c] for c in self.children.get(p, set()) if c in self.mid2name}
        return out - {name}

    def cousins(self, name: str) -> set[str]:
        """Vocab classes within 2 steps below a grandparent of `name`."""
        m = self.name2mid[name]
        out = set()
        for p in self.parents.get(m, set()):
            for g in self.parents.get(p, set()):
                out |= self._desc_vocab(g, 2)
        return out - {name}


def valid_negative(cand: str, labels, leaves, tax: FL.Taxonomy, cooc: dict, max_cooc: float = 0.02) -> bool:
    """`cand` may be asked as absent: presence-usable, not a label, not an ancestor of a label, not a descendant of
    a leaf label (siblings under a smeared parent label are allowed), not semantically blocked, and co-occurring with
    each leaf label in < max_cooc of FSD50K dev clips."""
    s = set(labels)
    if cand not in FL.CLASSES or not FL.CLASSES[cand][3] or cand in s:
        return False
    if any(cand in tax.ancestors[l] for l in s):
        return False
    if any(cand in tax.descendants[l] for l in leaves):
        return False
    if cand in semantic_blocked(s):
        return False
    for l in leaves:
        if cooc.get(l, {}).get(cand, 0.0) >= max_cooc:
            return False
    return True


def confusable(name: str) -> set[str]:
    """Acoustically confusable classes: members of sb_sounds' CONFUSION_GROUPS."""
    out = set()
    for g in FL.CONFUSION_GROUPS:
        if name in g:
            out |= g
    return out - {name}


def near_negatives(labels, leaves, tax, onto: FullOntology, cooc, max_cooc: float = 0.02) -> dict[str, str]:
    """{class: kind} of valid near negatives of the clip's leaves, by tier:
    'sibling' (shares a specific direct ontology parent) and 'confusable' (same sb_sounds confusion group) first;
    'broad_sibling' (shares only a top-level parent, or a vocab-level parent) only if tier 1 is empty;
    'cousin' only if nothing closer exists."""
    def collect(pairs):
        out = {}
        for c, kind in pairs:
            if c not in out and valid_negative(c, labels, leaves, tax, cooc, max_cooc):
                out[c] = kind
        return out

    tier1 = collect([(c, "sibling") for l in leaves for c in sorted(onto.direct_siblings(l, broad=False))]
                    + [(c, "confusable") for l in leaves for c in sorted(confusable(l))])
    if tier1:
        return tier1
    tier2 = collect([(c, "broad_sibling") for l in leaves
                     for c in sorted(onto.direct_siblings(l, broad=True) | tax.siblings(l))])
    if tier2:
        return tier2
    return collect([(c, "cousin") for l in leaves for c in sorted(onto.cousins(l))])


def pick_negative(labels, leaves, tax, onto, cooc, rng: random.Random, want_near: bool,
                  max_cooc: float = 0.02) -> tuple[str | None, str]:
    near = near_negatives(labels, leaves, tax, onto, cooc, max_cooc)
    if want_near and near:
        c = rng.choice(sorted(near))
        return c, near[c]
    pool = [x for x in tax.names if x not in near and valid_negative(x, labels, leaves, tax, cooc, max_cooc)]
    if not pool:
        return None, "none"
    return rng.choice(pool), "random"


def near_distractors(labels, leaves, gold: str, k: int, tax, onto, cooc, rng: random.Random,
                     display) -> list[tuple[str, str]] | None:
    """k distractors for 'which of these sounds is present', all near negatives (siblings / confusable / cousins of
    the gold class or of the clip's leaves), pairwise unrelated, distinct display strings. None if impossible."""
    near: dict[str, str] = {}
    for c in sorted(onto.direct_siblings(gold, broad=False)):
        near.setdefault(c, "sibling")
    for c in sorted(confusable(gold)):
        near.setdefault(c, "confusable")
    for c in sorted(onto.direct_siblings(gold, broad=True) | tax.siblings(gold)):
        near.setdefault(c, "broad_sibling")
    for c in sorted(onto.cousins(gold)):
        near.setdefault(c, "cousin")
    near = {c: k for c, k in near.items() if valid_negative(c, labels, leaves, tax, cooc)}

    def rank(c):  # siblings of the gold class first, then confusables, broad siblings, cousins
        return {"sibling": 0, "confusable": 1, "broad_sibling": 2, "cousin": 3}[near[c]]
    cands = sorted(near)
    rng.shuffle(cands)
    cands.sort(key=rank)
    chosen: list[str] = []
    names = {display(gold)}
    for c in cands:
        if len(chosen) >= k:
            break
        if any(tax.related(c, x) for x in chosen) or c in semantic_blocked(chosen) or display(c) in names:
            continue
        chosen.append(c)
        names.add(display(c))
    if len(chosen) < k:
        return None
    return [(c, near[c]) for c in chosen]


# Part 2. Music
# FMA genres (genres.csv: genre_id, parent, title, top_level)
NON_MUSIC_GENRES = {"Spoken", "Banter", "Comedy", "Musical Theater", "Poetry", "Radio", "Interview", "Radio Art",
                    "Talk Radio", "Radio Theater", "Spoken Weird", "Spoken Word", "Audio Collage", "Field Recordings",
                    "Novelty", "Kid-Friendly", "Sound Effects", "Holiday", "Christmas", "Sound Art", "Sound Collage",
                    "Sound Poetry", "Unclassifiable", "Musique Concrete", "Compilation"}

# genres that may be asked / shown as options, with their display name (others are still used for blocking)
GENRE_DISPLAY = {
    "Blues": "blues", "Gospel": "gospel", "Classical": "classical", "20th Century Classical": "20th-century classical",
    "Chamber Music": "chamber music", "Choral Music": "choral music", "Contemporary Classical": "contemporary classical",
    "Opera": "opera", "Symphony": "symphonic / orchestral", "Country": "country", "Americana": "americana",
    "Bluegrass": "bluegrass", "Country & Western": "country & western", "Rockabilly": "rockabilly",
    "Easy Listening": "easy listening", "Lounge": "lounge", "Nu-Jazz": "nu jazz", "Electronic": "electronic",
    "Ambient Electronic": "ambient electronic", "Breakcore - Hard": "breakcore", "Chip Music": "chiptune / chip music",
    "Dance": "dance", "Downtempo": "downtempo", "Drum & Bass": "drum and bass", "Dubstep": "dubstep",
    "Glitch": "glitch", "House": "house", "Chill-out": "chill-out", "IDM": "IDM (intelligent dance music)",
    "Jungle": "jungle", "Minimal Electronic": "minimal electronic", "Techno": "techno", "Bigbeat": "big beat",
    "Trip-Hop": "trip-hop", "Experimental": "experimental", "Avant-Garde": "avant-garde", "Drone": "drone",
    "Electroacoustic": "electroacoustic", "Improv": "free improvisation", "Minimalism": "minimalism",
    "Noise": "noise music", "Folk": "folk", "British Folk": "British folk", "Freak-Folk": "freak folk",
    "Free-Folk": "free folk", "Psych-Folk": "psych folk", "Singer-Songwriter": "singer-songwriter",
    "Hip-Hop": "hip-hop", "Abstract Hip-Hop": "abstract hip-hop", "Alternative Hip-Hop": "alternative hip-hop",
    "Breakbeat": "breakbeat", "Hip-Hop Beats": "hip-hop beats", "Nerdcore": "nerdcore", "Rap": "rap",
    "Ambient": "ambient", "New Age": "new age", "Soundtrack": "soundtrack / film score",
    "International": "world / international", "African": "African", "Afrobeat": "afrobeat",
    "North African": "North African", "Balkan": "Balkan", "Brazilian": "Brazilian", "Celtic": "Celtic",
    "Fado": "fado", "Klezmer": "klezmer", "Romany (Gypsy)": "Romani (gypsy) music", "Spanish": "Spanish",
    "Flamenco": "flamenco", "French": "French chanson / French", "Indian": "Indian", "Latin": "Latin",
    "Latin America": "Latin American", "Cumbia": "cumbia", "Salsa": "salsa", "Tango": "tango",
    "Middle East": "Middle Eastern", "Turkish": "Turkish", "Polka": "polka", "Reggae - Dub": "reggae / dub",
    "Reggae - Dancehall": "dancehall", "Jazz": "jazz", "Big Band/Swing": "big band / swing", "Free-Jazz": "free jazz",
    "Jazz: Vocal": "vocal jazz", "Modern Jazz": "modern jazz", "Old-Time / Historic": "old-time / historic recording",
    "Pop": "pop", "Experimental Pop": "experimental pop", "Synth Pop": "synth-pop", "Rock": "rock",
    "Garage": "garage rock", "Surf": "surf rock", "Goth": "goth", "Indie-Rock": "indie rock",
    "Industrial": "industrial", "Krautrock": "krautrock", "Lo-Fi": "lo-fi", "Noise-Rock": "noise rock",
    "Sludge": "sludge metal", "Metal": "metal", "Black-Metal": "black metal", "Death-Metal": "death metal",
    "Grindcore": "grindcore", "New Wave": "new wave", "Post-Rock": "post-rock", "Space-Rock": "space rock",
    "Progressive": "progressive rock", "Psych-Rock": "psychedelic rock", "Punk": "punk",
    "Electro-Punk": "electro-punk", "Hardcore": "hardcore punk", "Thrash": "thrash", "No Wave": "no wave",
    "Post-Punk": "post-punk", "Power-Pop": "power pop", "Rock Opera": "rock opera", "Shoegaze": "shoegaze",
    "Soul-RnB": "soul / R&B", "Disco": "disco", "Funk": "funk",
}
TOP_GENRES = ["Blues", "Classical", "Country", "Easy Listening", "Electronic", "Experimental", "Folk", "Hip-Hop",
              "International", "Jazz", "Old-Time / Historic", "Pop", "Rock", "Soul-RnB"]
# allowed as the correct answer, never as a distractor / "no" genre (too broad or orthogonal to style)
NEVER_NEGATIVE = {"Experimental", "Avant-Garde", "Experimental Pop", "Soundtrack", "Lo-Fi", "Singer-Songwriter",
                  "Easy Listening", "International", "Electroacoustic", "Minimalism", "Improv"}
# genres that overlap too much to be negatives / distractors for each other
GENRE_BLOCK: list[set[str]] = [
    {"Free-Folk", "Freak-Folk", "Psych-Folk", "Folk", "British Folk", "Celtic", "Americana", "Singer-Songwriter"},
    {"Americana", "Country", "Bluegrass", "Country & Western", "Rockabilly"},
    {"Ambient", "Ambient Electronic", "Chill-out", "Downtempo", "New Age", "Drone", "Minimal Electronic"},
    {"IDM", "Glitch", "Minimal Electronic", "Electronic", "Breakcore - Hard"},
    {"Dance", "House", "Techno", "Disco", "Synth Pop", "Bigbeat"},
    {"Drum & Bass", "Jungle", "Breakcore - Hard", "Breakbeat"},
    {"Hip-Hop Beats", "Breakbeat", "Abstract Hip-Hop", "Trip-Hop", "Downtempo", "Alternative Hip-Hop", "Electronic"},
    {"Rap", "Hip-Hop", "Alternative Hip-Hop", "Nerdcore", "Abstract Hip-Hop"},
    {"Indie-Rock", "Lo-Fi", "Garage", "Power-Pop", "Pop", "Rock", "Surf"},
    {"Post-Punk", "New Wave", "No Wave", "Goth", "Synth Pop"},
    {"Psych-Rock", "Space-Rock", "Krautrock", "Progressive", "Post-Rock", "Shoegaze", "Psych-Folk"},
    {"Punk", "Hardcore", "Thrash", "Electro-Punk", "Power-Pop", "Garage"},
    {"Noise-Rock", "Sludge", "Metal", "Industrial", "Noise", "Loud-Rock"},
    {"Metal", "Black-Metal", "Death-Metal", "Grindcore", "Thrash", "Hardcore", "Sludge"},
    {"Experimental", "Avant-Garde", "Experimental Pop", "Noise", "Electroacoustic", "Improv", "Minimalism", "Drone"},
    {"Free-Jazz", "Jazz: Out", "Improv", "Avant-Garde", "Jazz"},
    {"Modern Jazz", "Nu-Jazz", "Lounge", "Jazz: Vocal", "Big Band/Swing", "Easy Listening"},
    {"Contemporary Classical", "20th Century Classical", "Composed Music", "Minimalism", "Classical", "Soundtrack"},
    {"Electronic", "Synth Pop", "Electro-Punk", "Industrial", "Chip Music", "Chiptune", "Ambient", "Experimental",
     "Hip-Hop Beats", "Trip-Hop", "Downtempo"},
    {"Pop", "Synth Pop", "Electronic", "Soul-RnB", "Folk", "Experimental Pop", "Power-Pop"},
    {"Reggae - Dub", "Dubstep", "Reggae - Dancehall"},
    {"Latin America", "Latin", "Cumbia", "Salsa", "Tango", "Brazilian", "Spanish", "Flamenco"},
    {"Klezmer", "Balkan", "Romany (Gypsy)", "Europe", "Polka"},
    {"African", "Afrobeat", "North African", "Middle East", "Turkish"},
    {"Blues", "Gospel", "Soul-RnB", "Funk", "Rock"},
    {"Soul-RnB", "Funk", "Disco", "Hip-Hop"},
    {"Old-Time / Historic", "Blues", "Jazz", "Big Band/Swing", "Country", "Folk"},
]


class GenreTree:
    def __init__(self, rows: list[dict]):
        self.title = {int(r["genre_id"]): r["title"] for r in rows}
        self.id_of = {r["title"]: int(r["genre_id"]) for r in rows}
        self.parent = {int(r["genre_id"]): int(r["parent"]) for r in rows}
        self.children: dict[int, set[int]] = {}
        for g, p in self.parent.items():
            self.children.setdefault(p, set()).add(g)

    def ancestors(self, g: int) -> set[int]:
        out = set()
        while self.parent.get(g, 0):
            g = self.parent[g]
            out.add(g)
        return out

    def descendants(self, g: int) -> set[int]:
        out, st = set(), [g]
        while st:
            for c in self.children.get(st.pop(), set()):
                if c not in out:
                    out.add(c)
                    st.append(c)
        return out

    def root(self, g: int) -> int:
        while self.parent.get(g, 0):
            g = self.parent[g]
        return g

    def siblings(self, g: int) -> set[int]:
        return self.children.get(self.parent.get(g, 0), set()) - {g}


def genre_blocked(titles) -> set[str]:
    s = set(titles)
    out: set[str] = set()
    for grp in GENRE_BLOCK:
        if grp & s:
            out |= grp
    return out


def track_genres_ok(titles) -> bool:
    """A track is used for music questions only if it has at least one genre and none of them is non-music."""
    return bool(titles) and not (set(titles) & NON_MUSIC_GENRES)


def genre_valid_negative(cand: int, track: set[int], tree: GenreTree) -> bool:
    """`track`: the track's genre ids including ancestors (FMA genres_all)."""
    t = tree.title[cand]
    if t not in GENRE_DISPLAY or t in NEVER_NEGATIVE or cand in track:
        return False
    leaves = [g for g in track if not (tree.descendants(g) & track)]
    if any(cand in tree.descendants(g) for g in leaves):  # siblings under an ancestor genre stay allowed
        return False
    if t in genre_blocked(tree.title[g] for g in track):
        return False
    return True


def genre_near(track: set[int], tree: GenreTree) -> list[int]:
    """Valid negatives that are siblings of one of the track's genres (other roots for root genres)."""
    out = set()
    for g in track:
        out |= tree.siblings(g)
    return sorted(c for c in out if genre_valid_negative(c, track, tree))


def genre_random(track: set[int], tree: GenreTree) -> list[int]:
    return sorted(c for c in tree.title if genre_valid_negative(c, track, tree))


# OpenMIC-2018 instruments
INSTRUMENTS = {  # openmic name -> (option label, noun phrase)
    "accordion": ("accordion", "an accordion"), "banjo": ("banjo", "a banjo"),
    "bass": ("bass (bass guitar or double bass)", "a bass guitar or double bass"), "cello": ("cello", "a cello"),
    "clarinet": ("clarinet", "a clarinet"), "cymbals": ("cymbals", "cymbals"), "drums": ("drums", "drums"),
    "flute": ("flute", "a flute"), "guitar": ("guitar", "a guitar"),
    "mallet_percussion": ("mallet percussion (xylophone, marimba, vibraphone)", "mallet percussion like a xylophone or vibraphone"),
    "mandolin": ("mandolin", "a mandolin"), "organ": ("organ", "an organ"), "piano": ("piano", "a piano"),
    "saxophone": ("saxophone", "a saxophone"), "synthesizer": ("synthesizer", "a synthesizer"),
    "trombone": ("trombone", "a trombone"), "trumpet": ("trumpet", "a trumpet"), "ukulele": ("ukulele", "a ukulele"),
    "violin": ("violin", "a violin"), "voice": ("singing voice", "a singing voice"),
}
INSTR_NEAR = [  # acoustically confusable instruments (hard negatives for each other)
    {"violin", "cello"}, {"guitar", "banjo", "mandolin", "ukulele", "bass"}, {"piano", "organ", "synthesizer",
    "accordion", "mallet_percussion"}, {"trumpet", "trombone", "saxophone"}, {"flute", "clarinet", "saxophone"},
    {"drums", "cymbals", "mallet_percussion"}, {"bass", "cello", "trombone"}, {"accordion", "organ", "clarinet"},
]
POS_REL, NEG_REL = 0.75, 0.25  # OpenMIC aggregated relevance thresholds (present / verified absent)


def instr_near(name: str) -> set[str]:
    out = set()
    for g in INSTR_NEAR:
        if name in g:
            out |= g
    return out - {name}


def instrument_labels(rows: list[tuple[str, float]]) -> tuple[set[str], set[str]]:
    """[(instrument, relevance)] for one clip -> (present, verified absent); the uncertain middle is dropped."""
    pos = {i for i, r in rows if r >= POS_REL and i in INSTRUMENTS}
    neg = {i for i, r in rows if r <= NEG_REL and i in INSTRUMENTS} - pos
    return pos, neg


# licences
def licence_class(url: str) -> str | None:
    """Official FMA licence URL -> 'CC-BY' | 'CC-BY-SA' | 'CC0' | 'PD' (commercially usable) or None."""
    u = (url or "").lower()
    m = re.search(r"creativecommons\.org/licenses/([a-z\-+]+)/", u)
    if m:
        return {"by": "CC-BY", "by-sa": "CC-BY-SA", "publicdomain": "PD"}.get(m.group(1))
    if "publicdomain/zero" in u:
        return "CC0"
    if "publicdomain/mark" in u:
        return "PD"
    return None


def mirror_licence_class(name: str) -> str | None:
    """Licence class name of the benjamin-paine FMA repack ('CC-BY 4.0', 'CC0 1.0', ...)."""
    n = (name or "").strip()
    if re.fullmatch(r"CC-BY \d\.\d", n):
        return "CC-BY"
    if re.fullmatch(r"CC-BY-SA \d\.\d", n):
        return "CC-BY-SA"
    if n == "CC0 1.0":
        return "CC0"
    if n == "Public Domain Mark 1.0":
        return "PD"
    return None


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def parse_sdd_artists(text: str) -> tuple[set[str], set[str]]:
    """Song Describer audio_licenses.txt -> (normalised artist names, track ids)."""
    arts, ids = set(), set()
    for block in text.split("--\n"):
        lines = [l for l in block.strip().split("\n") if l.strip()]
        if len(lines) < 2:
            continue
        tid = lines[0].strip().split("/")[-1].split(".")[0]
        if tid.isdigit():
            ids.add(tid)
        m = re.match(r"(.*) by (.*?) from Jamendo", lines[1])
        if m:
            arts.add(norm_name(m.group(2)))
    return arts - {""}, ids


# tempo
TEMPO_LEVELS = ["slow", "medium", "fast"]
TEMPO_BANDS = {"slow": (40.0, 82.0), "medium": (95.0, 125.0), "fast": (140.0, 200.0)}
FPS = 100  # onset envelope frames per second (hop 160 at 16 kHz)


def onset_envelope(x: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Spectral-flux onset strength on 40 log-spaced bands, 100 frames/s, mean-removed and >= 0."""
    n_fft, hop = 1024, sr // FPS
    if len(x) < n_fft + hop:
        return np.zeros(0)
    win = np.hanning(n_fft).astype(np.float32)
    n = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * win, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    edges = np.geomspace(40, sr / 2 * 0.95, 41)
    bands = np.stack([spec[:, (freqs >= lo) & (freqs < hi)].sum(1) for lo, hi in zip(edges[:-1], edges[1:])], 1)
    lb = np.log1p(1000 * bands / (bands.max() + 1e-12))
    flux = np.maximum(np.diff(lb, axis=0), 0).sum(1)
    k = 2 * FPS // 10 + 1  # 0.2 s moving average removed
    flux = flux - np.convolve(flux, np.ones(k) / k, mode="same")
    return np.maximum(flux, 0)


def tempo_curve(env: np.ndarray, lo: float = 30.0, hi: float = 240.0):
    """Autocorrelation of the onset envelope as a function of BPM, plus a log-normal prior (120 BPM, 1 octave)."""
    if len(env) < 4 * FPS or env.std() == 0:
        return None
    e = (env - env.mean()) / (env.std() + 1e-9)
    ac = np.correlate(e, e, mode="full")[len(e) - 1:] / np.arange(len(e), 0, -1)
    lags = np.arange(int(FPS * 60 / hi), int(FPS * 60 / lo) + 1)
    lags = lags[lags < len(ac)]
    bpm = 60.0 * FPS / lags
    strength = np.maximum(ac[lags], 0)
    prior = np.exp(-0.5 * (np.log2(bpm / 120.0) / 1.0) ** 2)
    return bpm, strength, prior


def estimate_tempo(x: np.ndarray, sr: int = 16000) -> dict:
    """Tempo estimate with confidence checks. Returns {'bpm', 'ambiguity', 'halves': [bpm1, bpm2], 'peak'}."""
    env = onset_envelope(x, sr)
    res = {"bpm": None, "ambiguity": None, "halves": [None, None], "peak": None,
           "onset_rate": float((env > env.mean() + 2 * env.std()).mean() * FPS) if len(env) else None}

    def best(e):
        tc = tempo_curve(e)
        if tc is None:
            return None, None, None
        bpm, s, pr = tc
        w = s * pr
        i = int(np.argmax(w))
        return float(bpm[i]), (bpm, s, pr, w), i

    b, curve, i = best(env)
    if b is None:
        return res
    bpm, s, pr, w = curve
    # octave ambiguity: weighted strength of the double / half tempo relative to the chosen one
    alt = []
    for f in (2.0, 0.5, 1.5, 2 / 3):
        j = int(np.argmin(np.abs(bpm - b * f)))
        if abs(bpm[j] - b * f) / (b * f) < 0.04:
            alt.append(w[j] / (w[i] + 1e-12))
    res["bpm"] = b
    res["ambiguity"] = float(max(alt)) if alt else 0.0
    res["peak"] = float(s[i] / (np.mean(s) + 1e-9))
    h = len(env) // 2
    res["halves"] = [best(env[:h])[0], best(env[h:])[0]]
    return res


def tempo_class(t: dict, max_ambiguity: float = 0.7, agree: float = 0.05, min_peak: float = 1.5) -> str | None:
    """slow / medium / fast from a confident tempo estimate, None when unsure or between the bands."""
    b = t.get("bpm")
    if b is None or t.get("ambiguity") is None or t["ambiguity"] > max_ambiguity or (t.get("peak") or 0) < min_peak:
        return None
    for h in t.get("halves", []):
        if h is None or abs(h - b) / b > agree:
            return None
    for name, (lo, hi) in TEMPO_BANDS.items():
        if lo <= b <= hi:
            return name
    return None


# Part 3. Questions (>= 5 paraphrases each; the evaluation set's FSD50K templates are not used)
Q_PRES_NAME = ['Is the sound "{name}" present in this clip?', 'Can you hear "{name}" in this recording?',
               'Does the audio include the sound class "{name}"?', 'Would you tag this recording with "{name}"?',
               'Sound check: is "{name}" audible in this audio?', 'Is there any "{name}" in this clip?',
               'Listen carefully. Is "{name}" one of the sounds here?']
Q_PRES_NP = FL.Q_PRESENCE
Q_WHICH = ["Which of these sounds is present in the recording?", "Which one of these sounds can you hear?",
           "Which sound occurs in this clip?", "One of these sounds is in the audio. Which one?",
           "Pick the sound that is actually present in this recording.",
           "Which of the following can be heard in this clip?", "Several similar sounds are listed. Which one is here?"]

Q_GENRE_TOP = ["What genre is this music?", "Which genre best describes this track?", "What style of music is this?",
               "Which of these genres fits this excerpt best?", "How would you classify this music by genre?",
               "Pick the genre of this song.", "Which genre does this piece of music belong to?"]
Q_GENRE_SUB = ["Which sub-genre best describes this music?", "Which of these closely related styles fits this track?",
               "This is {parent} music. Which more specific style is it?",
               "Within {parent}, which style is this excerpt?", "Which of these similar genres is the best fit?",
               "Choose the most fitting style for this track."]
Q_GENRE_NOUL = ["Is this {g} music?", "Would you call this track {g}?", "Does this excerpt sound like {g}?",
                "Is the genre of this music {g}?", "Could this song be filed under {g}?",
                "Is this piece an example of {g}?"]
Q_INSTR = ["Can you hear {np} in this music?", "Is {np} playing in this excerpt?", "Does this track feature {np}?",
           "Is {np} part of the instrumentation here?", "Do you hear {np} anywhere in this clip?",
           "Listen to the arrangement: is there {np}?"]
Q_INSTR_WHICH = ["Which of these instruments can you hear?", "Which instrument is playing in this excerpt?",
                 "One of these instruments is in the music. Which one?", "Which of these is part of the arrangement?",
                 "Pick the instrument that is actually audible here."]
Q_VOCALS = ["Does this music have vocals?", "Is someone singing in this track?", "Can you hear a singing voice here?",
            "Are there sung vocals in this excerpt?", "Does this song include a vocalist?"]
Q_INSTRUMENTAL = ["Is this track instrumental, with no singing?", "Is this an instrumental piece (no vocals)?",
                  "Is this music purely instrumental?", "Does this excerpt have no vocals at all?",
                  "Is there no singing in this music?"]
Q_USE_BED = ["Could this excerpt work as an instrumental bed under a voice-over, i.e. it has no singing?",
             "Would this be suitable as background music for a podcast intro where nobody should be singing?",
             "Is this usable as vocal-free background music under narration?",
             "We need music without any singing to play under speech. Is this suitable?",
             "Is this excerpt suitable as on-hold music that must not contain vocals?"]
Q_TEMPO = ["Is the tempo of this music slow, medium or fast?", "How fast is this music?",
           "What is the pace of this track?", "Rate the tempo of this excerpt.", "How quick is the beat in this music?",
           "Is this a slow, mid-tempo or fast piece?"]
Q_LEVEL = ["How loud is this music?", "What is the playback level of this track?",
           "Is this music quiet, moderate or loud in volume?", "Rate the loudness of this excerpt.",
           "How strong is the audio level of this song?"]
Q_LIVE = ["Is this a live recording?", "Was this performance recorded live?", "Does this sound like a live performance?",
          "Is this a live performance rather than a studio recording?", "Was this captured at a live show or session?"]
