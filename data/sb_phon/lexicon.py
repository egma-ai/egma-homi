"""sb_phon text resources and phonology helpers (CMUdict ARPAbet and misaki phoneme strings)."""
from __future__ import annotations

import re

# Voices
EN_VOICES = ["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", "af_nicole", "af_nova",
             "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
             "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
             "bm_daniel", "bm_fable", "bm_george", "bm_lewis"]
CAL_VOICES = ["af_nova", "am_eric", "bf_lily"]
TRAIN_VOICES = [v for v in EN_VOICES if v not in CAL_VOICES]

# CMUdict
ARPA_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
STOPS = {"P", "B", "T", "D", "K", "G"}
VOICED_STOPS = {"B", "D", "G"}
FRICATIVE_PLACE = {"F", "TH", "S", "SH", "V", "DH", "Z", "ZH", "HH"}  # place contrasts that a phone band blurs
IPA = {"AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "ER": "ɝ", "EY": "eɪ",
       "IH": "ɪ", "IY": "i", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "u", "B": "b", "CH": "tʃ", "D": "d",
       "DH": "ð", "F": "f", "G": "ɡ", "HH": "h", "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
       "P": "p", "R": "r", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z",
       "ZH": "ʒ"}


def parse_cmudict(text: str) -> dict[str, list[tuple[str, ...]]]:
    out: dict[str, list[tuple[str, ...]]] = {}
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        w, *ph = line.split()
        w = re.sub(r"\(\d+\)$", "", w)
        out.setdefault(w, []).append(tuple(ph))
    return out


def base(p: str) -> str:
    return re.sub(r"\d", "", p)


def syllables(pron) -> int:
    return sum(1 for p in pron if p[-1].isdigit())


def primary_index(pron) -> int | None:
    """0-based syllable index of the primary stress (None unless exactly one '1')."""
    st = [p[-1] for p in pron if p[-1].isdigit()]
    return st.index("1") if st.count("1") == 1 else None


def unstressed(pron) -> tuple[str, ...]:
    return tuple(base(p) for p in pron)


def onset(pron) -> tuple[str, ...]:
    out = []
    for p in pron:
        if base(p) in ARPA_VOWELS:
            break
        out.append(base(p))
    return tuple(out)


def coda(pron) -> tuple[str, ...]:
    out = []
    for p in reversed(pron):
        if base(p) in ARPA_VOWELS:
            break
        out.append(base(p))
    return tuple(reversed(out))


def nucleus(pron) -> str | None:
    v = [base(p) for p in pron if base(p) in ARPA_VOWELS]
    return v[0] if len(v) == 1 else None


def ipa(phones) -> str:
    return "".join(IPA[base(p)] for p in phones)


def consistent(prons, f):
    """f(pron) if all CMUdict pronunciations agree, else None."""
    vals = {f(p) for p in prons}
    return vals.pop() if len(vals) == 1 else None


# misaki
MISAKI_VOWELS = set("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻᵊ")
GB_DIGRAPHS = ("ɪə", "ʊə", "ɛə")


def misaki_nuclei(ps: str, gb: bool = False) -> list[int]:
    """Character positions of the syllable nuclei in a misaki phoneme string."""
    pos, i = [], 0
    while i < len(ps):
        c = ps[i]
        if c in MISAKI_VOWELS:
            pos.append(i)
            if gb and ps[i:i + 2] in GB_DIGRAPHS:
                i += 2
                continue
        i += 1
    return pos


def misaki_syllables(ps: str, gb: bool = False) -> int:
    return len(misaki_nuclei(ps, gb))


def misaki_primary(ps: str, gb: bool = False) -> int | None:
    if ps.count("ˈ") != 1:
        return None
    k = ps.index("ˈ")
    nuc = misaki_nuclei(ps, gb)
    for j, p in enumerate(nuc):
        if p > k:
            return j
    return None


def strip_stress(ps: str) -> str:
    return ps.replace("ˈ", "").replace("ˌ", "")


# Word lists
# Common short words (minimal pairs, sound identification, final-stop release). Labels always come from CMUdict.
COMMON = """
bat pat cat hat mat rat sat fat vat bad pad dad mad sad had lad bag rag tag wag back pack sack tack rack lack
bed red fed led wed bet pet set get net wet jet met let vet bit pit sit hit kit fit lit big dig fig pig wig
bin pin tin win sin fin kin thin shin chin bun fun gun run sun but cut hut nut shut bug dug hug jug mug rug tug
cup pup cub rub tub dot hot lot not pot rot got cot cod nod rod pod mop top hop pop cop shop chop log dog fog jog
beat seat heat meat neat feet sheet bead read lead need seed feed weed bean mean lean seen keen teen green
ship sheep chip cheap lip leap hip heap dip deep sip fill feel pill peel mill meal hill heal still steal
full fool pull pool look luck foot boot good food could hood book buck cook took tuck
cap cart heart ban cab fan van fine vine few view safe save leaf leave half have
think sink thick sick mouth mouse three free thought fought sue zoo bus buzz rice rise peace peas
shoe chew share chair wash watch light right long wrong lock rock late rate fly fry glass grass
tap tab rope robe lap lab mob coat goat cold gold come gum card guard curl girl
tie die time dime town down two do ten den tear dear pie buy pea bee pear bear
wine west vest yes jam map nap mail nail sing ring thing king wing
make take cake bake lake rake wake fake shake day bay say way pay may lay ray
boat note vote nice mice dice price bone cone tone phone zone lime chime
bite kite white site lake leak like lick lock luck bake beak book brick bread break
cart cord bark park dark mark fork pork cork corn horn torn born
bird word heard herd fur her sir stir
plate plane plan clap clip slip slap snap snack stack stuck stick trick truck track
road rode load toad code mode
sock sack suck sick seek soak
bell ball bill bull boil
coin join noise toys boys
house loud cloud crowd proud
price prize rice ride wide side tide hide
""".split()

# Base sentences: plain declaratives without contractions, digits or hyphens. One word in [a|b] is swappable
# (the editing-term repair uses the other alternative as the "wrong" word).
SENTENCES = [
    "My sister painted the [old|new] fence yesterday",
    "The [new|young] teacher gave us a long test",
    "We should take the [early|late] train to the city",
    "Our neighbor found a [small|black] cat in the garden",
    "The children played outside until [dinner|lunch] was ready",
    "He left his [blue|green] jacket on the bus",
    "I will call the doctor on [Monday|Tuesday] morning",
    "The bakery sells fresh bread every [morning|evening]",
    "She wrote a long letter to her [brother|cousin]",
    "They moved into a [bigger|smaller] house last year",
    "The [manager|owner] wants the report before noon",
    "We watched a funny movie at the [theater|library]",
    "My father fixed the [kitchen|bathroom] sink this weekend",
    "The river was too cold for [swimming|fishing] today",
    "Please bring two [chairs|tables] to the meeting room",
    "The store closes at [nine|ten] on weekdays",
    "Our team won the game in the [final|second] minute",
    "The museum is next to the [old|main] train station",
    "I need to buy [milk|rice] and eggs after work",
    "The garden looks beautiful in the [spring|summer]",
    "Her phone was under the [sofa|bed] all morning",
    "The pilot said the flight would be [smooth|short]",
    "We planted [tomatoes|flowers] behind the house",
    "My uncle drives a [red|white] truck to work",
    "The library has a large collection of [maps|books]",
    "The baby finally fell asleep after [lunch|dinner]",
    "The mechanic checked the [brakes|tires] on my car",
    "Everyone enjoyed the [soup|cake] at the party",
    "The students cleaned the [classroom|hallway] after school",
    "A strong wind knocked over the [fence|tree] last night",
    "The waiter brought us [water|coffee] and a menu",
    "My friend is learning to play the [piano|guitar]",
    "The hotel room had a view of the [ocean|mountains]",
    "We walked along the [beach|river] for an hour",
    "The doctor told him to drink more [water|tea]",
    "The [office|school] will be closed on the holiday",
    "Grandma baked an apple pie for [dessert|breakfast]",
    "The farmer sold [potatoes|apples] at the market",
    "The concert started later than we [expected|planned]",
    "Somebody left the [front|back] door open again",
]

STOPWORDS = set("""a an the to of in on at for and or but is was were be been are am i we you he she they it my our
your his her their this that these those with from by as into than then there so do did not no will would should can
could us me him them its""".split())

# Stress-based reasoning: sentence with emphasisable words; meaning of emphasis per word index.
# {N} = name, other slots listed per template.
NAMES = ["Anna", "David", "Maria", "Peter", "Laura", "Simon", "Rachel", "Daniel", "Emma", "Oliver", "Sophie",
         "Henry", "Julia", "Martin", "Clara", "Lucas"]
REASON = [
    {"text": "{N} bought the {C} car on {D}", "slots": {"C": ["red", "blue", "green", "white", "black"],
                                                        "D": ["Monday", "Friday", "Sunday", "Tuesday"]},
     "emph": {0: "It was {N}, not someone else, who bought it", 1: "{N} paid for it rather than borrowing it",
              3: "The car was {C}, not another colour", 4: "It was a car, not some other vehicle",
              6: "It happened on {D}, not on another day"}},
    {"text": "I never said {P} took your {O}", "slots": {"P": ["she", "he"], "O": ["money", "keys", "phone", "bike"]},
     "emph": {0: "Someone else said it, not me", 1: "I deny ever having said it",
              2: "I may have hinted at it, but I did not say it", 3: "Someone other than {P} took it",
              4: "{P} did something else with it, such as borrowing it", 5: "{P} took somebody else's {O}",
              6: "{P} took something of yours, but not your {O}"}},
    {"text": "We are meeting at {T} on {D}", "slots": {"T": ["noon", "five", "eight", "ten"],
                                                      "D": ["Monday", "Thursday", "Saturday"]},
     "emph": {0: "We, not other people, are the ones meeting", 4: "The time is {T}, not some other time",
              6: "The day is {D}, not another day"}},
    {"text": "{N} wants the {S} box", "slots": {"S": ["small", "large", "red", "wooden"]},
     "emph": {0: "{N}, not someone else, wants it", 1: "{N} really does want it",
              3: "The {S} box, not one of the other boxes", 4: "The box, not some other item"}},
    {"text": "Please put the {O} on the {F}", "slots": {"O": ["plates", "books", "flowers", "glasses"],
                                                       "F": ["table", "shelf", "desk", "counter"]},
     "emph": {1: "Place it there instead of throwing or dropping it", 3: "The {O}, not some other things, go there",
              6: "They go on the {F}, not somewhere else"}},
    {"text": "{N} can drive us to the {L} today", "slots": {"L": ["airport", "station", "hospital", "stadium"]},
     "emph": {0: "{N}, not anyone else, can drive us", 2: "{N} will drive us rather than walk with us",
              6: "We are going to the {L}, not another place", 7: "It happens today, not another day"}},
    {"text": "The {A} dog chased the {B} cat", "slots": {"A": ["big", "brown", "old", "small"],
                                                        "B": ["white", "grey", "little", "black"]},
     "emph": {1: "It was the {A} dog, not a different dog", 2: "A dog did it, not some other animal",
              3: "The dog chased it rather than ignoring it", 5: "It was the {B} cat, not a different cat",
              6: "The dog chased a cat, not another animal"}},
    {"text": "{N} finished the report on {D}", "slots": {"D": ["Monday", "Friday", "Wednesday"]},
     "emph": {0: "{N}, not someone else, finished it", 1: "{N} completed it, not just started it",
              3: "It was the report, not some other task", 5: "It was done on {D}, not another day"}},
    {"text": "You told me the party was on {D}", "slots": {"D": ["Saturday", "Friday", "Sunday"]},
     "emph": {0: "You, not someone else, told me", 2: "You told me, not somebody else",
              4: "It was the party, not some other event", 7: "The party was said to be on {D}, not another day"}},
    {"text": "{P} only eats {F} for lunch", "slots": {"P": ["She", "He"], "F": ["salad", "soup", "rice", "fruit"]},
     "emph": {0: "{P}, unlike other people, eats that", 1: "Nothing else but {F}",
              3: "{F}, not some other food", 5: "At lunch, not at other meals"}},
    {"text": "I asked {N} to call the {R}", "slots": {"R": ["doctor", "plumber", "teacher", "bank"]},
     "emph": {0: "I, not someone else, made the request", 2: "{N}, not someone else, was asked",
              4: "{N} was asked to phone them, not to visit or email", 6: "The {R}, not someone else, should be called"}},
    {"text": "My brother lost his {O} at the {L}", "slots": {"O": ["wallet", "watch", "keys", "hat"],
                                                            "L": ["park", "beach", "mall", "gym"]},
     "emph": {1: "My brother, not another relative, lost it", 2: "He lost it rather than having it stolen",
              4: "It was his {O}, not something else", 7: "It happened at the {L}, not somewhere else"}},
]

LIST_NOUNS = ["lake", "river", "forest", "mountain", "island", "valley", "desert", "garden", "castle", "bridge",
              "market", "village", "harbor", "meadow", "canyon", "temple", "tower", "station"]
LIST_FRAMES = ["We visited a {0}, a {1}, a {2} and a {3}", "On the map I saw a {0}, a {1}, a {2} and a {3}",
               "The painting shows a {0}, a {1}, a {2} and a {3}", "Our tour included a {0}, a {1}, a {2} and a {3}"]

# Statements that also work as echo questions when spoken with a rise (no question words, no inversion).
INTON_SENTENCES = [
    "You finished the whole project already", "She is moving to another city", "The meeting starts at noon",
    "They sold the old house", "He forgot his keys again", "We are leaving tomorrow morning",
    "The store is closed on Sunday", "You talked to the manager", "Your brother is coming with us",
    "The train was late again", "She painted the kitchen yellow", "You want the window open",
    "They live near the river", "He called the doctor this morning", "The tickets were sold out",
    "You already ate dinner", "The baby is sleeping upstairs", "We need more paper for the printer",
    "Her flight lands in the evening", "You read the whole book in one day", "The restaurant opens at seven",
    "They won the game last night", "You lost your phone again", "The weather will be warm tomorrow",
    "He is studying medicine now", "The package arrived this morning", "You have been to London before",
    "She bought a new car", "The children are playing outside", "You remember the address",
]
