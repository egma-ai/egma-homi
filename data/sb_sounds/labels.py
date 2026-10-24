"""Label logic for sb_sounds (no audio I/O): FSD50K class table, ontology relations, confusion groups, negative
selection, NSynth and signal-level rules, question paraphrases.
"""

from __future__ import annotations

import math
import random

import numpy as np

# Broad categories (family 3: "what kind of sound is this?")
SP, MU, AN, VE, TM, WA, HH, HU, AL, NA, EX = (
    "speech", "music", "animal sounds", "vehicles or engines", "tools or machinery", "water or liquid",
    "household sounds", "human sounds other than speech", "alarm or ringing", "wind, thunder or fire",
    "explosion or gunshot")
CATEGORIES = [SP, MU, AN, VE, TM, WA, HH, HU, AL, NA, EX]

# name: (option label, noun phrase, broad category or None, usable as presence target, usable as identification answer)
# category None = ambiguous between categories -> clip skipped for the category question.
CLASSES: dict[str, tuple[str, str, str | None, bool, bool]] = {
    "Accelerating_and_revving_and_vroom": ("engine revving", "an engine revving", VE, True, True),
    "Accordion": ("accordion", "an accordion", MU, True, True),
    "Acoustic_guitar": ("acoustic guitar", "an acoustic guitar", MU, True, True),
    "Aircraft": ("aircraft", "an aircraft", VE, True, True),
    "Alarm": ("alarm", "an alarm", AL, True, True),
    "Animal": ("animal", "an animal", AN, True, False),
    "Applause": ("applause", "applause", HU, True, True),
    "Bark": ("dog barking", "a dog barking", AN, True, True),
    "Bass_drum": ("bass drum", "a bass drum", MU, True, True),
    "Bass_guitar": ("bass guitar", "a bass guitar", MU, True, True),
    "Bathtub_(filling_or_washing)": ("bathtub filling", "a bathtub being filled or washed", None, True, True),
    "Bell": ("bell", "a bell", None, True, True),
    "Bicycle": ("bicycle", "a bicycle", VE, True, True),
    "Bicycle_bell": ("bicycle bell", "a bicycle bell", None, True, True),
    "Bird": ("bird", "a bird", AN, True, True),
    "Bird_vocalization_and_bird_call_and_bird_song": ("bird song", "a bird singing or calling", AN, True, True),
    "Boat_and_Water_vehicle": ("boat", "a boat", VE, True, True),
    "Boiling": ("boiling liquid", "something boiling", WA, True, True),
    "Boom": ("boom", "a deep boom", EX, True, True),
    "Bowed_string_instrument": ("violin or cello", "a bowed string instrument like a violin or cello", MU, True, True),
    "Brass_instrument": ("brass instrument", "a brass instrument", MU, True, True),
    "Breathing": ("breathing", "someone breathing", HU, True, True),
    "Burping_and_eructation": ("burping", "someone burping", HU, True, True),
    "Bus": ("bus", "a bus", VE, True, True),
    "Buzz": ("insect buzzing", "an insect buzzing", AN, True, True),
    "Camera": ("camera shutter", "a camera shutter clicking", TM, True, True),
    "Car": ("car", "a car", VE, True, True),
    "Car_passing_by": ("car passing by", "a car passing by", VE, True, True),
    "Cat": ("cat", "a cat", AN, True, True),
    "Chatter": ("people chattering", "people chattering", None, True, True),
    "Cheering": ("cheering", "people cheering", HU, True, True),
    "Chewing_and_mastication": ("chewing", "someone chewing", HU, True, True),
    "Chicken_and_rooster": ("chicken or rooster", "a chicken or rooster", AN, True, True),
    "Child_speech_and_kid_speaking": ("child speaking", "a child speaking", SP, True, True),
    "Chime": ("chimes", "chimes", None, True, True),
    "Chink_and_clink": ("clinking", "clinking, like glasses touching", None, True, True),
    "Chirp_and_tweet": ("birds chirping", "birds chirping", AN, True, True),
    "Chuckle_and_chortle": ("chuckling", "someone chuckling", HU, True, True),
    "Church_bell": ("church bell", "a church bell", None, True, True),
    "Clapping": ("clapping", "clapping", HU, True, True),
    "Clock": ("clock", "a clock", None, True, True),
    "Coin_(dropping)": ("coins dropping", "coins dropping", HH, True, True),
    "Computer_keyboard": ("computer keyboard", "typing on a computer keyboard", HH, True, True),
    "Conversation": ("conversation", "people having a conversation", SP, True, True),
    "Cough": ("coughing", "someone coughing", HU, True, True),
    "Cowbell": ("cowbell", "a cowbell", None, True, True),
    "Crack": ("cracking", "a cracking sound", None, True, True),
    "Crackle": ("crackling", "crackling", None, True, True),
    "Crash_cymbal": ("crash cymbal", "a crash cymbal", MU, True, True),
    "Cricket": ("crickets", "crickets chirping", AN, True, True),
    "Crow": ("crow", "a crow cawing", AN, True, True),
    "Crowd": ("crowd", "a crowd", None, True, True),
    "Crumpling_and_crinkling": ("crumpling", "something being crumpled", None, True, True),
    "Crushing": ("crushing", "something being crushed", None, True, True),
    "Crying_and_sobbing": ("crying", "someone crying", HU, True, True),
    "Cupboard_open_or_close": ("cupboard door", "a cupboard opening or closing", HH, True, True),
    "Cutlery_and_silverware": ("cutlery", "cutlery clattering", HH, True, True),
    "Cymbal": ("cymbal", "a cymbal", MU, True, True),
    "Dishes_and_pots_and_pans": ("dishes and pans", "dishes or pots and pans clattering", HH, True, True),
    "Dog": ("dog", "a dog", AN, True, True),
    "Domestic_animals_and_pets": ("pet animal", "a pet animal", AN, False, False),
    "Domestic_sounds_and_home_sounds": ("household sounds", "household sounds", HH, False, False),
    "Door": ("door", "a door", HH, True, True),
    "Doorbell": ("doorbell", "a doorbell", AL, True, True),
    "Drawer_open_or_close": ("drawer", "a drawer opening or closing", HH, True, True),
    "Drill": ("drill", "a power drill", TM, True, True),
    "Drip": ("dripping", "dripping water", WA, True, True),
    "Drum": ("drum", "a drum", MU, True, True),
    "Drum_kit": ("drum kit", "a drum kit", MU, True, True),
    "Electric_guitar": ("electric guitar", "an electric guitar", MU, True, True),
    "Engine": ("engine", "an engine", VE, True, True),
    "Engine_starting": ("engine starting", "an engine starting", VE, True, True),
    "Explosion": ("explosion", "an explosion", EX, True, True),
    "Fart": ("fart", "someone farting", HU, True, True),
    "Female_singing": ("woman singing", "a woman singing", None, True, True),
    "Female_speech_and_woman_speaking": ("woman speaking", "a woman speaking", SP, True, True),
    "Fill_(with_liquid)": ("filling a container", "a container being filled with liquid", WA, True, True),
    "Finger_snapping": ("finger snapping", "fingers snapping", HU, True, True),
    "Fire": ("fire", "a fire burning", NA, True, True),
    "Fireworks": ("fireworks", "fireworks", EX, True, True),
    "Fixed-wing_aircraft_and_airplane": ("airplane", "an airplane", VE, True, True),
    "Fowl": ("poultry", "poultry such as chickens or ducks", AN, True, True),
    "Frog": ("frog", "frogs croaking", AN, True, True),
    "Frying_(food)": ("frying food", "food frying in a pan", HH, True, True),
    "Gasp": ("gasp", "someone gasping", HU, True, True),
    "Giggle": ("giggling", "someone giggling", HU, True, True),
    "Glass": ("glass", "glass clinking or breaking", None, True, False),
    "Glockenspiel": ("glockenspiel", "a glockenspiel", MU, True, True),
    "Gong": ("gong", "a gong", MU, True, True),
    "Growling": ("growling", "an animal growling", AN, True, True),
    "Guitar": ("guitar", "a guitar", MU, True, True),
    "Gull_and_seagull": ("seagulls", "seagulls calling", AN, True, True),
    "Gunshot_and_gunfire": ("gunshots", "gunshots", EX, True, True),
    "Gurgling": ("gurgling", "water gurgling", WA, True, True),
    "Hammer": ("hammering", "a hammer", TM, True, True),
    "Hands": ("hand sounds", "hand sounds", HU, False, False),
    "Harmonica": ("harmonica", "a harmonica", MU, True, True),
    "Harp": ("harp", "a harp", MU, True, True),
    "Hi-hat": ("hi-hat", "a hi-hat cymbal", MU, True, True),
    "Hiss": ("hiss", "a hiss", None, False, False),  # ambiguous (cat hiss / steam / noise)
    "Human_group_actions": ("group of people", "a group of people", None, False, False),
    "Human_voice": ("human voice", "a human voice", None, True, False),
    "Idling": ("engine idling", "an engine idling", VE, True, True),
    "Insect": ("insect", "an insect", AN, True, True),
    "Keyboard_(musical)": ("keyboard instrument", "a keyboard instrument such as a piano or organ", MU, True, True),
    "Keys_jangling": ("keys jangling", "keys jangling", HH, True, True),
    "Knock": ("knocking", "knocking", HH, True, True),
    "Laughter": ("laughter", "someone laughing", HU, True, True),
    "Liquid": ("liquid", "liquid sounds like pouring or splashing", WA, True, False),
    "Livestock_and_farm_animals_and_working_animals": ("farm animals", "farm animals", AN, True, False),
    "Male_singing": ("man singing", "a man singing", None, True, True),
    "Male_speech_and_man_speaking": ("man speaking", "a man speaking", SP, True, True),
    "Mallet_percussion": ("xylophone-like percussion", "a mallet percussion instrument like a xylophone", MU, True, True),
    "Marimba_and_xylophone": ("marimba or xylophone", "a marimba or xylophone", MU, True, True),
    "Mechanical_fan": ("fan", "a mechanical fan", TM, True, True),
    "Mechanisms": ("mechanism", "a mechanism", TM, False, False),
    "Meow": ("cat meowing", "a cat meowing", AN, True, True),
    "Microwave_oven": ("microwave oven", "a microwave oven", HH, True, True),
    "Motor_vehicle_(road)": ("road vehicle", "a road vehicle", VE, True, False),
    "Motorcycle": ("motorcycle", "a motorcycle", VE, True, True),
    "Music": ("music", "music", MU, True, False),
    "Musical_instrument": ("musical instrument", "a musical instrument", MU, True, False),
    "Ocean": ("ocean", "the ocean", WA, True, True),
    "Organ": ("organ", "an organ", MU, True, True),
    "Packing_tape_and_duct_tape": ("packing tape", "tape being pulled off a roll", HH, True, True),
    "Percussion": ("percussion", "percussion", MU, True, False),
    "Piano": ("piano", "a piano", MU, True, True),
    "Plucked_string_instrument": ("plucked strings", "a plucked string instrument", MU, True, False),
    "Pour": ("pouring", "liquid being poured", WA, True, True),
    "Power_tool": ("power tool", "a power tool", TM, True, True),
    "Printer": ("printer", "a printer", TM, True, True),
    "Purr": ("cat purring", "a cat purring", AN, True, True),
    "Race_car_and_auto_racing": ("race car", "a race car", VE, True, True),
    "Rail_transport": ("train or tram", "a train or other rail vehicle", VE, True, True),
    "Rain": ("rain", "rain", WA, True, True),
    "Raindrop": ("raindrops", "raindrops", WA, True, True),
    "Ratchet_and_pawl": ("ratchet", "a ratchet clicking", TM, True, True),
    "Rattle": ("rattle", "a rattle", None, False, False),  # ambiguous (animal rattle vs object)
    "Rattle_(instrument)": ("shaker", "a rattle or shaker instrument", MU, True, True),
    "Respiratory_sounds": ("respiratory sounds", "respiratory sounds", HU, False, False),
    "Ringtone": ("ringtone", "a phone ringtone", AL, True, True),
    "Run": ("running", "someone running", HU, True, True),
    "Sawing": ("sawing", "sawing", TM, True, True),
    "Scissors": ("scissors", "scissors cutting", HH, True, True),
    "Scratching_(performance_technique)": ("DJ scratching", "DJ record scratching", MU, True, True),
    "Screaming": ("screaming", "someone screaming", HU, True, True),
    "Screech": ("screech", "a screech", None, True, True),
    "Shatter": ("shattering", "something shattering", None, True, True),
    "Shout": ("shouting", "someone shouting", None, True, True),
    "Sigh": ("sigh", "someone sighing", HU, True, True),
    "Singing": ("singing", "someone singing", None, True, True),
    "Sink_(filling_or_washing)": ("sink", "a sink being filled or used for washing", None, True, True),
    "Siren": ("siren", "a siren", AL, True, True),
    "Skateboard": ("skateboard", "a skateboard", VE, True, True),
    "Slam": ("slam", "something slamming, like a door", HH, True, True),
    "Sliding_door": ("sliding door", "a sliding door", HH, True, True),
    "Snare_drum": ("snare drum", "a snare drum", MU, True, True),
    "Sneeze": ("sneezing", "someone sneezing", HU, True, True),
    "Speech": ("speech", "someone speaking", SP, True, True),
    "Speech_synthesizer": ("synthesized speech", "a computer-generated voice speaking", SP, True, True),
    "Splash_and_splatter": ("splashing", "splashing", WA, True, True),
    "Squeak": ("squeak", "a squeak", None, True, True),
    "Stream": ("stream", "a flowing stream", WA, True, True),
    "Strum": ("strumming", "strings being strummed", MU, True, True),
    "Subway_and_metro_and_underground": ("subway train", "a subway or metro train", VE, True, True),
    "Tabla": ("tabla", "a tabla", MU, True, True),
    "Tambourine": ("tambourine", "a tambourine", MU, True, True),
    "Tap": ("tapping", "tapping", None, True, True),
    "Tearing": ("tearing", "something being torn", None, True, True),
    "Telephone": ("telephone", "a telephone", AL, True, True),
    "Thump_and_thud": ("thump", "a thump or thud", None, True, True),
    "Thunder": ("thunder", "thunder", NA, True, True),
    "Thunderstorm": ("thunderstorm", "a thunderstorm", NA, True, True),
    "Tick": ("ticking", "ticking", None, True, True),
    "Tick-tock": ("clock ticking", "a clock ticking", None, True, True),
    "Toilet_flush": ("toilet flushing", "a toilet flushing", None, True, True),
    "Tools": ("tools", "tools being used", TM, True, False),
    "Traffic_noise_and_roadway_noise": ("traffic", "road traffic", VE, True, True),
    "Train": ("train", "a train", VE, True, True),
    "Trickle_and_dribble": ("trickling", "liquid trickling", WA, True, True),
    "Truck": ("truck", "a truck", VE, True, True),
    "Trumpet": ("trumpet", "a trumpet", MU, True, True),
    "Typewriter": ("typewriter", "a typewriter", HH, True, True),
    "Typing": ("typing", "typing", HH, True, True),
    "Vehicle": ("vehicle", "a vehicle", VE, True, False),
    "Vehicle_horn_and_car_horn_and_honking": ("car horn", "a car horn honking", None, True, True),
    "Walk_and_footsteps": ("footsteps", "footsteps", HU, True, True),
    "Water": ("water", "water", WA, True, False),
    "Water_tap_and_faucet": ("running tap", "a running tap", None, True, True),
    "Waves_and_surf": ("waves", "waves crashing", WA, True, True),
    "Whispering": ("whispering", "someone whispering", SP, True, True),
    "Whoosh_and_swoosh_and_swish": ("whoosh", "a whoosh or swish", None, True, True),
    "Wild_animals": ("wild animal", "a wild animal", AN, True, False),
    "Wind": ("wind", "wind", NA, True, True),
    "Wind_chime": ("wind chimes", "wind chimes", None, True, True),
    "Wind_instrument_and_woodwind_instrument": ("woodwind instrument", "a woodwind instrument such as a flute or clarinet", MU, True, True),
    "Wood": ("wood", "wood being knocked or creaking", None, False, False),
    "Writing": ("writing", "someone writing with a pen or pencil", HH, True, True),
    "Yell": ("yelling", "someone yelling", None, True, True),
    "Zipper_(clothing)": ("zipper", "a zipper", HH, True, True),
}

# Classes whose names overlap in meaning across ontology branches (e.g. "water" vs "pouring", "bird" vs
# "chicken", "drum kit" vs "cymbal"). If a clip has any class of a group, no class of that group may be
# used as an absent ("no") class or as a distractor for that clip. Conservative on purpose: a wrong "no"
# label is much worse than a missing hard negative.
CONFUSION_GROUPS: list[set[str]] = [
    {"Water", "Liquid", "Pour", "Fill_(with_liquid)", "Trickle_and_dribble", "Drip", "Splash_and_splatter",
     "Gurgling", "Stream", "Water_tap_and_faucet", "Sink_(filling_or_washing)", "Bathtub_(filling_or_washing)",
     "Toilet_flush", "Rain", "Raindrop", "Ocean", "Waves_and_surf", "Boiling", "Boat_and_Water_vehicle"},
    {"Human_voice", "Speech", "Conversation", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking",
     "Child_speech_and_kid_speaking", "Speech_synthesizer", "Chatter", "Crowd", "Whispering", "Singing",
     "Male_singing", "Female_singing", "Shout", "Yell", "Screaming", "Laughter", "Giggle", "Chuckle_and_chortle",
     "Crying_and_sobbing", "Sigh", "Cheering", "Applause", "Human_group_actions", "Gasp", "Breathing"},
    {"Bell", "Chime", "Wind_chime", "Church_bell", "Cowbell", "Bicycle_bell", "Doorbell", "Glockenspiel", "Gong",
     "Telephone", "Ringtone", "Alarm", "Mallet_percussion", "Marimba_and_xylophone"},
    {"Thump_and_thud", "Knock", "Slam", "Tap", "Door", "Hammer", "Crack", "Wood", "Boom", "Explosion", "Thunder",
     "Gunshot_and_gunfire", "Fireworks", "Bass_drum", "Drum", "Cupboard_open_or_close", "Drawer_open_or_close",
     "Sliding_door", "Walk_and_footsteps", "Run"},
    {"Vehicle", "Motor_vehicle_(road)", "Car", "Car_passing_by", "Race_car_and_auto_racing",
     "Traffic_noise_and_roadway_noise", "Truck", "Bus", "Motorcycle", "Engine", "Engine_starting", "Idling",
     "Accelerating_and_revving_and_vroom", "Vehicle_horn_and_car_horn_and_honking", "Skateboard", "Bicycle"},
    {"Vehicle", "Rail_transport", "Train", "Subway_and_metro_and_underground", "Engine", "Idling"},
    {"Vehicle", "Aircraft", "Fixed-wing_aircraft_and_airplane", "Engine", "Wind", "Whoosh_and_swoosh_and_swish"},
    {"Vehicle", "Boat_and_Water_vehicle", "Engine", "Idling"},
    {"Engine", "Power_tool", "Drill", "Mechanical_fan", "Tools", "Mechanisms", "Sawing", "Hammer", "Ratchet_and_pawl",
     "Printer", "Microwave_oven", "Idling"},
    {"Music", "Musical_instrument", "Percussion", "Drum_kit", "Drum", "Snare_drum", "Bass_drum", "Cymbal",
     "Crash_cymbal", "Hi-hat", "Tambourine", "Tabla", "Rattle_(instrument)"},
    {"Guitar", "Acoustic_guitar", "Electric_guitar", "Bass_guitar", "Strum", "Plucked_string_instrument", "Harp"},
    {"Keyboard_(musical)", "Piano", "Organ", "Accordion", "Harmonica"},
    {"Wind_instrument_and_woodwind_instrument", "Brass_instrument", "Trumpet", "Harmonica", "Accordion"},
    {"Wind", "Whoosh_and_swoosh_and_swish", "Thunderstorm", "Thunder", "Rain", "Raindrop", "Ocean",
     "Waves_and_surf", "Mechanical_fan"},
    {"Fire", "Crackle", "Frying_(food)", "Boiling", "Fireworks"},
    {"Respiratory_sounds", "Breathing", "Gasp", "Sigh", "Cough", "Sneeze", "Burping_and_eructation"},
    {"Walk_and_footsteps", "Run", "Skateboard"},
    {"Animal", "Bird", "Bird_vocalization_and_bird_call_and_bird_song", "Chirp_and_tweet", "Crow",
     "Gull_and_seagull", "Fowl", "Chicken_and_rooster", "Wild_animals", "Livestock_and_farm_animals_and_working_animals",
     "Insect", "Cricket", "Frog"},
    {"Insect", "Buzz", "Cricket", "Mechanical_fan"},
    {"Cat", "Dog", "Growling", "Hiss", "Purr", "Meow", "Bark", "Domestic_animals_and_pets"},
    {"Glass", "Chink_and_clink", "Shatter", "Dishes_and_pots_and_pans", "Cutlery_and_silverware", "Coin_(dropping)",
     "Keys_jangling", "Crushing", "Bell", "Wind_chime"},
    {"Door", "Cupboard_open_or_close", "Drawer_open_or_close", "Sliding_door", "Slam", "Knock", "Squeak", "Tap",
     "Doorbell", "Domestic_sounds_and_home_sounds"},
    {"Crumpling_and_crinkling", "Tearing", "Packing_tape_and_duct_tape", "Crushing", "Writing", "Scissors"},
    {"Screech", "Squeak", "Scratching_(performance_technique)", "Car", "Race_car_and_auto_racing"},
    {"Typing", "Computer_keyboard", "Typewriter", "Writing", "Printer", "Tick", "Clock", "Tick-tock", "Camera",
     "Mechanisms", "Ratchet_and_pawl"},
    {"Hands", "Clapping", "Finger_snapping", "Applause"},
    {"Thunderstorm", "Thunder", "Boom", "Explosion", "Fireworks", "Gunshot_and_gunfire"},
]

# Background/ambience classes: a single clip of these may hold several kinds of sound, so they are not used
# for the "how many kinds of sound" question.
AMBIENCE = {"Traffic_noise_and_roadway_noise", "Crowd", "Chatter", "Conversation", "Rain", "Wind", "Ocean",
            "Waves_and_surf", "Stream", "Thunderstorm", "Applause", "Cheering", "Human_group_actions",
            "Domestic_sounds_and_home_sounds", "Water", "Vehicle", "Motor_vehicle_(road)", "Music"}

# label sets used for the scene questions
SPEECH_YES = {"Speech", "Conversation", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking",
              "Child_speech_and_kid_speaking"}
BELL_LIKE = {"Bell", "Chime", "Wind_chime", "Church_bell", "Cowbell", "Bicycle_bell"}


def validate_class_table(vocab: list[str]) -> None:
    """Every FSD50K class must have exactly one row, and every group member must be a real class."""
    missing = sorted(set(vocab) - set(CLASSES))
    extra = sorted(set(CLASSES) - set(vocab))
    bad = sorted({c for g in CONFUSION_GROUPS for c in g} - set(vocab)) + sorted(AMBIENCE - set(vocab))
    if missing or extra or bad:
        raise ValueError(f"class table mismatch: missing={missing} extra={extra} bad_group_members={bad}")


# Taxonomy (FSD50K vocabulary inside the AudioSet ontology)
class Taxonomy:
    """`vocab`: {mid: name} for the 200 FSD50K classes; `ontology`: list of AudioSet ontology nodes
    ({"id", "name", "child_ids"}). Relations are computed between vocabulary classes, walking through
    ontology nodes that are not in the vocabulary."""

    def __init__(self, vocab: dict[str, str], ontology: list[dict]):
        self.mid2name = dict(vocab)
        self.names = sorted(vocab.values())
        parents: dict[str, list[str]] = {}
        for node in ontology:
            for c in node.get("child_ids", []):
                parents.setdefault(c, []).append(node["id"])

        def vocab_parents(mid):  # nearest ancestors that are in the vocabulary
            res, stack, seen = set(), list(parents.get(mid, [])), set()
            while stack:
                p = stack.pop()
                if p in seen:
                    continue
                seen.add(p)
                if p in vocab:
                    res.add(p)
                else:
                    stack += parents.get(p, [])
            return res

        self.parents = {vocab[m]: {vocab[p] for p in vocab_parents(m)} for m in vocab}
        self.ancestors: dict[str, set[str]] = {}
        for n in self.names:
            out, stack = set(), list(self.parents[n])
            while stack:
                p = stack.pop()
                if p not in out:
                    out.add(p)
                    stack += list(self.parents[p])
            self.ancestors[n] = out
        self.descendants: dict[str, set[str]] = {n: set() for n in self.names}
        for n, anc in self.ancestors.items():
            for a in anc:
                self.descendants[a].add(n)
        self.children: dict[str, set[str]] = {n: set() for n in self.names}
        for n, ps in self.parents.items():
            for p in ps:
                self.children[p].add(n)

    def related(self, a: str, b: str) -> bool:
        return a == b or b in self.ancestors[a] or b in self.descendants[a]

    def siblings(self, name: str) -> set[str]:
        out = set()
        for p in self.parents[name]:
            out |= self.children[p]
        return out - {name} - self.ancestors[name] - self.descendants[name]

    def leaves(self, labels) -> list[str]:
        """Most specific labels of a clip: labels with no descendant also among the labels."""
        s = set(labels)
        return sorted(l for l in s if not (self.descendants[l] & s))


def group_blocked(labels) -> set[str]:
    """All classes sharing a confusion group with any of `labels`."""
    s = set(labels)
    out = set()
    for g in CONFUSION_GROUPS:
        if g & s:
            out |= g
    return out


def cooccurrence(label_lists) -> dict[str, dict[str, float]]:
    """P(b in labels | a in labels) over a corpus of clip label lists."""
    count: dict[str, int] = {}
    pair: dict[str, dict[str, int]] = {}
    for labels in label_lists:
        s = set(labels)
        for a in s:
            count[a] = count.get(a, 0) + 1
            row = pair.setdefault(a, {})
            for b in s:
                if b != a:
                    row[b] = row.get(b, 0) + 1
    return {a: {b: n / count[a] for b, n in row.items()} for a, row in pair.items()}


def is_valid_absent(cand: str, labels, tax: Taxonomy, cooc: dict, max_cooc: float = 0.02) -> bool:
    """True if `cand` can be asked as an absent class for a clip with `labels`."""
    s = set(labels)
    if cand not in CLASSES or not CLASSES[cand][3]:
        return False
    if cand in s or cand in group_blocked(s):
        return False
    for l in s:
        if tax.related(cand, l):
            return False
        if cooc.get(l, {}).get(cand, 0.0) >= max_cooc:
            return False
    return True


def pick_absent(labels, tax: Taxonomy, cooc: dict, rng: random.Random, hard: bool) -> str | None:
    """Pick an absent class. hard=True: a sibling of one of the clip's labels (same parent in the ontology);
    falls back to a random class if no valid sibling exists. Returns None if nothing is valid."""
    s = sorted(set(labels))
    if hard:
        sibs = sorted({x for l in s for x in tax.siblings(l)})
        sibs = [x for x in sibs if is_valid_absent(x, s, tax, cooc)]
        if sibs:
            return rng.choice(sibs)
    pool = [x for x in tax.names if is_valid_absent(x, s, tax, cooc)]
    return rng.choice(pool) if pool else None


def pick_distractors(labels, gold: str, k: int, tax: Taxonomy, cooc: dict, rng: random.Random,
                     n_hard: int = 2, need_ident: bool = True) -> list[str] | None:
    """k distractor classes for an identification question: up to n_hard siblings of the gold class, the rest
    random; all valid absent classes, pairwise unrelated, with distinct option labels."""
    chosen: list[str] = []

    def ok(x):
        if need_ident and not CLASSES[x][4]:
            return False
        if not is_valid_absent(x, labels, tax, cooc):
            return False
        if any(tax.related(x, c) for c in chosen) or x in group_blocked(chosen):
            return False
        return CLASSES[x][0] not in {CLASSES[c][0] for c in chosen} | {CLASSES[gold][0]}

    sibs = sorted(tax.siblings(gold))
    rng.shuffle(sibs)
    for x in sibs:
        if len(chosen) >= min(n_hard, k):
            break
        if ok(x):
            chosen.append(x)
    pool = list(tax.names)
    rng.shuffle(pool)
    for x in pool:
        if len(chosen) >= k:
            break
        if ok(x):
            chosen.append(x)
    return chosen if len(chosen) == k else None


def clip_category(labels, tax: Taxonomy) -> str | None:
    """Broad category of a clip from its most specific labels; None if they disagree or are ambiguous."""
    cats = {CLASSES[l][2] for l in tax.leaves(labels)}
    if len(cats) != 1 or None in cats:
        return None
    return cats.pop()


def music_answer(labels) -> str | None:
    """'yes' / 'no' for "is there music?", None if ambiguous (e.g. only bells or singing)."""
    s = set(labels)
    musical = {l for l in s if l == "Music" or l in MUSIC_SUBTREE}
    if not musical and not (s & {"Singing", "Male_singing", "Female_singing", "Scratching_(performance_technique)"}):
        if s & (BELL_LIKE | {"Doorbell", "Ringtone", "Alarm", "Telephone", "Whistle"}):
            return None
        return "no"
    real = musical - BELL_LIKE - {"Music", "Musical_instrument"}
    if real:
        return "yes"
    return None


MUSIC_SUBTREE: set[str] = set()  # filled by init_music_subtree(tax)


def init_music_subtree(tax: Taxonomy) -> None:
    MUSIC_SUBTREE.clear()
    MUSIC_SUBTREE.update(tax.descendants["Music"] | {"Music"})


def speech_answer(labels) -> str | None:
    """'yes' if someone is speaking (speech labels), 'no' if no human-voice-like label at all, else None."""
    s = set(labels)
    if s & SPEECH_YES:
        return "yes"
    voice = CONFUSION_GROUPS[1]
    if s & voice:
        return None
    return "no"


# NSynth
NSYNTH_FAMILY = {"bass": "bass", "brass": "brass", "flute": "flute", "guitar": "guitar", "keyboard": "keyboard",
                 "mallet": "mallet percussion", "organ": "organ", "reed": "reed instrument",
                 "string": "bowed strings", "synth_lead": "synth lead", "vocal": "voice"}
NSYNTH_SOURCE = {"acoustic": "acoustic instrument", "electronic": "electronic instrument",
                 "synthetic": "synthesizer"}
REGISTER_LEVELS = ["low", "middle", "high"]


def pitch_register(midi_pitch: int) -> str | None:
    """MIDI pitch -> low (<= A2, 45) / middle (E3..G4, 52..67) / high (>= E5, 76); None in the gaps so that
    borderline notes are never labelled."""
    if midi_pitch <= 45:
        return "low"
    if 52 <= midi_pitch <= 67:
        return "middle"
    if midi_pitch >= 76:
        return "high"
    return None


# Signal measurements
FRAME = 320  # 20 ms at 16 kHz


def frame_rms_db(x: np.ndarray, frame: int = FRAME) -> np.ndarray:
    n = len(x) // frame
    if n == 0:
        return np.array([20 * math.log10(max(float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0, 1e-10))])
    f = np.asarray(x[: n * frame], dtype=np.float64).reshape(n, frame)
    return 20 * np.log10(np.maximum(np.sqrt(np.mean(f * f, axis=1)), 1e-10))


def active_rms_db(x: np.ndarray, rel_db: float = 30.0) -> float:
    """Level of the 'active' part: RMS over frames within rel_db of the loudest frame."""
    db = frame_rms_db(x)
    act = db[db >= db.max() - rel_db]
    return float(10 * np.log10(np.mean(10 ** (act / 10))))


def active_until_end(x: np.ndarray, tail_s: float = 0.4, rel_db: float = 20.0, sr: int = 16000) -> bool:
    """True if the sound is still clearly going on during the last `tail_s` seconds: every tail frame is within
    rel_db of the clip's active level."""
    db = frame_rms_db(x)
    n_tail = max(1, int(tail_s * sr / FRAME))
    if len(db) < 2 * n_tail:
        return False
    return bool(np.all(db[-n_tail:] >= active_rms_db(x) - rel_db))


LEVELS = ["quiet", "moderate", "loud"]
LEVEL_RANGES_DB = {"quiet": (-46.0, -38.0), "moderate": (-30.0, -25.0), "loud": (-16.0, -11.0)}


def feasible_levels(x: np.ndarray, peak_max: float = 0.99) -> list[str]:
    """Loudness levels this clip can be scaled to (top of the range) without its peak exceeding peak_max."""
    lvl = active_rms_db(x)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    out = []
    for name, (lo, hi) in LEVEL_RANGES_DB.items():
        gain = 10 ** ((lo - lvl) / 20)
        if peak * gain <= peak_max:
            out.append(name)
    return out


# Question paraphrases (>= 5 per question type)
Q_PRESENCE = ["Is there {np} in this recording?", "Can you hear {np}?", "Does this clip contain {np}?",
              "Do you hear {np} anywhere in the audio?", "Is the sound of {np} present in this clip?",
              "Listen carefully: is there {np} in this audio?", "Does the recording include {np}?"]
# classes whose noun phrase is plural ("Is there" -> "Are there")
PLURAL = {"Chatter", "Cheering", "Conversation", "Chirp_and_tweet", "Cricket", "Gull_and_seagull", "Frog",
          "Coin_(dropping)", "Fireworks", "Gunshot_and_gunfire", "Raindrop", "Wind_chime", "Chime",
          "Walk_and_footsteps", "Dishes_and_pots_and_pans", "Finger_snapping", "Keys_jangling", "Waves_and_surf",
          "Livestock_and_farm_animals_and_working_animals", "Tools", "Liquid", "Strum", "Scissors",
          "Domestic_sounds_and_home_sounds", "Hands", "Respiratory_sounds"}


def presence_question(template: str, cls: str) -> str:
    """Fill a presence template with the class noun phrase, fixing number agreement for plural phrases."""
    q = template.format(np=CLASSES[cls][1])
    if cls in PLURAL:
        q = q.replace("Is there ", "Are there ").replace("is there ", "are there ")
        q = q.replace("Is the sound of ", "Is the sound of ")
    return q
Q_IDENT = ["Which sound is most prominent in this recording?", "What is making this sound?",
           "What do you hear in this clip?", "Which of these best describes the sound?",
           "What is the main sound in this audio?", "Identify the sound in this recording.",
           "What sound is this?"]
Q_CATEGORY = ["What kind of sound is this, broadly?", "Which category best describes this sound?",
              "What type of sound is in this recording?", "Broadly speaking, what are you hearing?",
              "Which group does this sound belong to?", "What general kind of audio is this?"]
Q_MUSIC = ["Is there music playing?", "Can you hear any music in this clip?", "Does this recording contain music?",
           "Is music audible in this audio?", "Is any music or musical instrument playing here?",
           "Do you hear music in the background or foreground?"]
Q_SPEECH = ["Is someone talking?", "Is someone speaking in this recording?", "Can you hear a person talking?",
            "Does this clip contain speech?", "Is there a human voice speaking words here?",
            "Do you hear anyone talk in this audio?"]
Q_COUNT = ["How many different kinds of sound can you hear?", "How many distinct sound sources are in this clip?",
           "How many different types of sound does this recording contain?",
           "Count the different kinds of sound you hear: how many are there?",
           "How many separate sounds of different kinds are mixed in this audio?"]
Q_LEVEL = ["How loud is this recording?", "What is the volume level of this audio?",
           "Is this clip quiet, moderate or loud in volume?", "How loud does this audio play back?",
           "Rate the loudness of this recording.", "How strong is the audio level in this clip?"]
Q_TRAIL = ["Does the sound stop before the end of the clip?", "Is the end of this recording silent?",
           "Does the audio fall silent before the clip ends?",
           "Is there a quiet stretch at the end of the clip after the sound stops?",
           "Does the sound end before the recording does, leaving silence at the end?"]
Q_NS_FAMILY = ["Which instrument family is playing this note?", "What kind of instrument produced this note?",
               "Which type of instrument is this?", "What instrument family does this sound come from?",
               "Identify the instrument family of this note."]
Q_NS_SOURCE = ["How was this note produced?", "Is this an acoustic instrument, an electronic instrument or a synthesizer?",
               "What kind of sound source is this note from?",
               "Was this note played on an acoustic instrument, an electronic one, or a synthesizer?",
               "Which best describes where this sound comes from?"]
Q_NS_REGISTER = ["Is this note low, middle or high in pitch?", "How high is the pitch of this note?",
                 "What pitch register is this note in?", "Rate the pitch of this note.",
                 "Is the note's pitch low, medium or high?"]
Q_NS_QUALITY = {
    "bright": ["Does this note sound bright?", "Is the tone of this note bright, with lots of high frequencies?",
               "Would you describe this note's timbre as bright?", "Is this a bright-sounding note?",
               "Does the note have a bright, crisp tone?"],
    "dark": ["Does this note sound dark?", "Is the tone of this note dark and mellow?",
             "Would you describe this note's timbre as dark?", "Is this a dark-sounding note?",
             "Does the note lack high frequencies, sounding dark?"],
    "distortion": ["Does this note sound distorted?", "Is there distortion on this note?",
                   "Does the sound have a gritty, distorted quality?", "Is this note distorted?",
                   "Can you hear distortion in this note?"],
    "percussive": ["Does this note have a sharp, percussive attack?", "Is this note percussive?",
                   "Does the note start with a hard, percussive hit?", "Is the onset of this note percussive?",
                   "Does this sound like a struck or percussive note?"],
    "reverb": ["Is there reverb on this note?", "Does this note sound reverberant, like in a large room?",
               "Can you hear reverb or echo on this note?", "Is this note recorded with reverb?",
               "Does the note ring on with room reverberation?"],
    "fast_decay": ["Does this note die away quickly?", "Does the note decay fast after it starts?",
                   "Is this a short note that fades quickly?", "Does the sound fade out quickly?",
                   "Does this note have a fast decay?"],
}
