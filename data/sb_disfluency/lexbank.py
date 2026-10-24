"""Text resources for sb_disfluency, all written for this builder: base sentences and disfluency vocabularies, and
the word banks and chunk grammar used to generate metrical lines."""
from __future__ import annotations

# Disfluency base sentences: plain declaratives with one swappable word [right|wrong], no digits / contractions /
# hyphens and no disfluency vocabulary (checked in build.check_base_sentence).
DISFL_SENTENCES = [
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
    "Please bring two [chairs|tables] to the meeting room",
    "The closes at [nine|ten] on weekdays",
    "The museum is next to the [old|main] train station",
    "I need to buy [milk|rice] and eggs after work",
    "The garden looks beautiful in the [spring|summer]",
    "Her phone was under the [sofa|bed] all morning",
    "We planted [tomatoes|flowers] behind the house",
    "My uncle drives a [red|white] truck to work",
    "The mechanic checked the [brakes|tires] on my car",
    "Everyone enjoyed the [soup|cake] at the party",
    "A strong wind knocked over the [fence|tree] last night",
    "The waiter brought us [water|coffee] and a menu",
    "My friend is learning to play the [piano|guitar]",
    "The hotel room had a view of the [ocean|mountains]",
    "We walked along the [beach|river] for an hour",
    "Grandma baked an apple for [dessert|breakfast]",
    "The farmer sold [potatoes|apples] at the market",
    "Somebody left the [front|back] door open again",
    "The bus to the airport leaves every [hour|evening]",
    "Our class visited the [science|history] museum on Friday",
    "I borrowed three books from the [library|school] last week",
    "The dentist moved my appointment to [Thursday|Wednesday]",
    "He parked the car behind the [bank|bakery]",
    "My cousin bought a [yellow|purple] bicycle for the summer",
    "We had pancakes and [juice|tea] for breakfast",
    "The plumber will come on [Friday|Saturday] afternoon",
    "She keeps her keys in the [kitchen|hallway] drawer",
    "The new restaurant serves [pizza|pasta] and salads",
    "They painted the bedroom a light [green|gray] color",
    "Our flight to the coast was delayed by [two|three] hours",
    "The kids built a castle out of [sand|snow]",
    "I sent the package to my [aunt|uncle] in the city",
    "The meeting moved to the [second|third] floor",
    "He always drinks [black|green] tea in the afternoon",
    "We need a bigger [table|couch] for the living room",
    "My brother plays [soccer|tennis] every weekend",
    "The library opens at [eight|nine] in the morning",
    "She found her old [camera|diary] in the attic",
    "The train stopped at the [north|south] station",
    "I left the umbrella in the [car|office] again",
    "Our teacher read a story about a [brave|clever] fox",
    "The shop on the corner sells [shoes|hats] and bags",
    "We celebrated his birthday at the [lake|park]",
    "The cat sleeps on the [chair|rug] by the window",
    "They are building a new [bridge|school] near the river",
    "My grandmother grows [roses|herbs] in her garden",
    "The concert in the park starts at [seven|six]",
    "He carried the [boxes|bags] up to the attic",
    "Our neighbors adopted a [puppy|kitten] last month",
    "The soup needs a little more [salt|pepper]",
    "I met my friend at the [station|airport] after lunch",
    "The office printer ran out of [paper|ink] this morning",
    "She rides her bike to the [market|gym] on Sundays",
    "The doctor said the results look [good|fine]",
    "We stayed at a small hotel near the [harbor|castle]",
    "The students cleaned the [classroom|hallway] after school",
    "My mother called me on [Sunday|Saturday] evening",
    "The baby finally fell asleep after [lunch|dinner]",
    "He forgot his [wallet|passport] at the hotel",
    "The pharmacy closes early on [Sundays|holidays]",
    "We ordered a [large|medium] pizza for the movie night",
    "Our team practices in the [gym|field] after class",
    "The river was too cold for [swimming|fishing] today",
    "I keep my notes in a [blue|black] folder",
    "The festival will be held in [June|July] this year",
    "She is planning a trip to the [mountains|coast] in the fall",
    "The bakery downtown makes the best [bread|cookies]",
    "My father reads the newspaper every [morning|night]",
    "The new park has a pond and a [playground|garden]",
    "We painted the fence [white|brown] last spring",
    "They watched the game at the [stadium|cinema] downtown",
    "The cook added fresh [basil|garlic] to the sauce",
    "Our neighbor plays the [violin|drums] every evening",
    "The road to the village was covered in [snow|mud]",
    "I bought a [warm|new] coat for the winter",
    "The children fed the [ducks|geese] at the pond",
    "My sister works at a [hospital|bank] in the city",
]

FILLED = ["um", "uh", "er"]
FILLED_W = [0.42, 0.40, 0.18]
MARKER_START = ["well", "so", "you know"]
MARKER_MID = ["you know", "like"]
EDIT_TERMS = ["I mean", "sorry", "or rather", "rather", "no wait", "no, sorry", "sorry, I mean"]
ABANDONED = ["I was going to", "We could", "Did you", "There was a", "Maybe we", "It was the", "They said",
             "When we", "If you", "The first", "Can you", "We have to", "She was", "Do you", "He told"]
# words that must never occur in a base sentence (they are disfluency vocabulary)
DISFL_VOCAB = {"well", "so", "like", "know", "mean", "sorry", "rather", "wait", "no", "um", "uh", "er", "actually",
               "okay", "umm", "uhh", "hmm"}

# Verse word banks: category -> (shape, words). U = unstressed monosyllable, S = stressed monosyllable, digits =
# CMUdict stress of a polysyllable ("10" trochee, "01" iamb, "100" dactyl, "010" amphibrach). Polysyllables are
# verified at plan time; words that fail are dropped and reported.
BANK = {
    "DET": ("U", "the my your his her our their this that".split()),
    "PREP1": ("U", "in on by through from with to at near past".split()),
    "CONJ": ("U", "and but as while when where".split()),
    "HAD": ("U", ["had"]),
    "PREP2i": ("01", "beneath above across along beyond below upon among around within behind between".split()),
    "PREP2t": ("10", "under over after".split()),
    "N1": ("S", """moon sea night wind rain star light stone road hill bird tree sky song heart field door bell ship snow
        leaf rose wave shore flame cloud dawn frost sand lake mist sun storm gate wall bridge grass smoke dream ghost
        king bride child horse crow fox deer wolf owl stream pond bay day way spray clay hay ray sight flight
        kite height knight glee bee key knee eye pie sigh tune dune noon spoon bone throne tone zone
        lane plain chain train grain crane cane car scar mill quill well shell spell floor
        oar roar store gong throng art dart part start tide side ride guide pride hand land
        band strand gold fold mold glow show lawn fawn sleep sheep heap leap stream beam gleam team cream seam
        cave grave fate mist list dust rust crust trust word herd wood hood hall ball call fall
        cake flake snake wake dove glove love time rhyme chime nest chest west rest quest crest spring king ring
        wing thing string sting street sweet wheat fleet boat coat goat moat note""".split()),
    "N1pl": ("S", "stars waves birds trees leaves clouds winds ships bells stones hills roads fields doors flames dreams "
                  "ghosts kings crows owls streams".split()),
    "N2t": ("10", """river morning garden window shadow meadow mountain water candle harbor forest summer winter valley
        ocean city lantern thunder willow feather maiden sparrow ember mirror castle island tower flower pillow autumn
        cabin chapel kingdom orchard pebble ribbon robin saddle sailor shepherd spider stranger temple timber tunnel
        velvet village wagon widow wonder raven falcon dragon lion poet giant sister mother father brother
        daughter""".split()),
    "N2t_pl": ("10", """rivers mountains shadows meadows candles lanterns willows waters sparrows embers ravens falcons
        dragons castles islands flowers gardens windows forests valleys oceans sailors strangers maidens poets giants
        tunnels wagons robins""".split()),
    "N2i": ("01", """delight lagoon balloon cocoon monsoon domain terrain campaign guitar cigar hotel gazelle parade
        cascade bouquet machine ravine marine cartoon platoon harpoon typhoon decay display array bazaar canal giraffe
        retreat defeat surprise command demand belief relief design decline""".split()),
    "N3d": ("100", "memory melody mystery harmony wilderness galaxy fantasy lavender animal monument".split()),
    "ADJ1": ("S", """cold dark bright old soft still slow deep pale gray warm wild high long sweet lost green blue red
        white black proud sad kind brave small vast faint lone bare thin grim calm clear""".split()),
    "ADJ2t": ("10", """golden silent gentle quiet distant hollow empty little yellow silver lonely ancient frozen broken
        sleepy misty stormy sunny rosy dusty crimson purple scarlet tender humble restless endless weary heavy narrow
        shallow simple secret sacred wooden woven faded""".split()),
    "ADJ3d": ("100", """beautiful wonderful shimmering wandering flickering glittering delicate desolate radiant
        slumbering quivering lingering""".split()),
    "V1": ("S", """fell sang burned slept turned glowed called shone flowed rose fled wept dreamed sighed stood ran came
        stayed swayed played rang grew blew flew""".split()),
    "V2t": ("10", """wandered whispered lingered glittered trembled shivered flickered glimmered slumbered murmured faded
        waited rested vanished wondered listened hurried""".split()),
    "V1t": ("S", "held kept lost found touched crossed kissed knew saw heard sought warmed filled left passed reached "
                 "watched".split()),
    "V2tt": ("10", "followed carried covered gathered answered entered greeted haunted guarded measured treasured".split()),
    "V2it": ("01", "forgot embraced released ignored adored admired obeyed betrayed believed received pursued "
                   "defied".split()),
    "V2i": ("01", "returned appeared remained awoke arose replied escaped rejoiced revived survived dissolved".split()),
    "V3m": ("010", "departed descended ascended retreated resounded awakened".split()),
    "ADV2i": ("01", "tonight today away alone again below above ahead around afar aloud abroad apart aglow ashore "
                    "asleep awake anew".split()),
    "ADV1": ("S", "far high low near deep home slow".split()),
}
# line-final categories (the rhyme word): all end on a stressed syllable
END_CATS = {"N1", "N2i", "ADV2i", "ADV1"}
REGULAR_ED = lambda w: w.endswith("ed")  # noqa: E731  "had" + participle needs a regular past form

# chunk types: role -> list of category sequences ("had" is the literal auxiliary)
CHUNKS = {
    "NP": [["DET", "N1"], ["DET", "N2t"], ["DET", "ADJ2t", "N1"], ["DET", "ADJ2t", "N2t"], ["DET", "N2i"],
           ["DET", "N3d"], ["DET", "ADJ3d", "N1"], ["ADJ2t", "N2t_pl"], ["N1pl"], ["ADJ2t", "N1pl"], ["N2t_pl"],
           ["DET", "ADJ1", "N2i"]],
    "VP": [["V1"], ["V2t"], ["V2i"], ["V3m"], ["HAD", "V2i_ed"], ["HAD", "V1_ed"], ["V1t", "DET", "N1"],
           ["V1t", "DET", "N2t"], ["V2tt", "DET", "N1"], ["V2it", "DET", "N1"], ["V1t", "DET", "ADJ2t", "N1"],
           ["V2tt", "DET", "N2t"], ["V2it", "DET", "N2t"], ["V1t", "DET", "N2i"], ["V2it", "DET", "ADJ2t", "N1"]],
    "PP": [["PREP1", "DET", "N1"], ["PREP1", "DET", "N2t"], ["PREP1", "DET", "ADJ2t", "N1"], ["PREP2i", "DET", "N1"],
           ["PREP2i", "DET", "N2t"], ["PREP2t", "DET", "N1"], ["PREP2t", "DET", "N2t"],
           ["PREP2i", "DET", "ADJ2t", "N1"], ["PREP2t", "DET", "ADJ2t", "N1"], ["PREP1", "DET", "ADJ3d", "N1"]],
    "ADV": [["ADV2i"], ["ADV1"]],
    "CONJ": [["CONJ"]],
}
FRAMES = [("NP", "VP", "PP"), ("NP", "VP"), ("NP", "VP", "ADV"), ("NP", "PP"), ("CONJ", "NP", "VP", "PP"),
          ("CONJ", "NP", "VP"), ("PP", "NP", "VP"), ("VP", "PP"), ("NP", "VP", "PP", "PP"), ("VP", "PP", "PP"),
          ("CONJ", "NP", "PP"), ("PP", "NP", "VP", "ADV"), ("NP", "VP", "PP", "ADV")]

# meters (masculine endings only): name -> (family, stress pattern; 1 = stressed syllable)
METERS = {
    "iamb3": ("iambic", "010101"), "iamb4": ("iambic", "01010101"), "iamb5": ("iambic", "0101010101"),
    "troch4": ("trochaic", "1010101"), "troch5": ("trochaic", "101010101"),
    "anap2": ("anapestic", "001001"), "anap3": ("anapestic", "001001001"), "anap4": ("anapestic", "001001001001"),
    "dact3": ("dactylic", "1001001"), "dact4": ("dactylic", "1001001001"),
}
FAMILIES = ["iambic", "trochaic", "anapestic", "dactylic"]
