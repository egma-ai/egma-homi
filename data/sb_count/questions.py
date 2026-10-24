"""sb_count question paraphrases (our own wording; >= 8 per question type). Validation asserts that no item question
equals any MMSU question string."""
from __future__ import annotations

# turns
TURN_COUNT = [
    "How many speaking turns does this conversation contain? Every change of speaker starts a new turn.",
    "Count the turns in this exchange (a turn ends whenever a different person starts talking). How many are there?",
    "How many turns are taken in total in this recording? A pause by the same person does not start a new turn.",
    "In this dialog, how many separate turns are there, counting a new turn each time the speaker switches?",
    "How many times does the floor pass to a speaker, counting the opening turn? In other words, how many turns are there?",
    "Listen to the conversation and count its turns, where one turn is a stretch of talk by one person. What is the total?",
    "What is the number of conversational turns in this audio?",
    "How many turns make up this conversation? (Consecutive sentences by the same person count as one turn.)",
    "Counting each uninterrupted stretch by one talker as a single turn, how many turns do you hear?",
    "How many back-and-forth turns are in this clip, counting the first one?",
]
TURN_CHANGES = [
    "How many times does the speaker change in this recording?",
    "How many speaker switches happen in this conversation?",
    "Count how often a different person takes over talking. How many switches are there?",
    "How many times does the talking pass from one person to another?",
    "How many hand-overs between speakers occur in this clip?",
    "How many changes of speaker can you hear?",
    "How often does someone new start talking after another person, in total?",
    "What is the number of speaker changes in this dialog?",
]
TURN_NOUL = {
    "more": ["Are there more than {n} turns in this conversation?", "Does this dialog have more than {n} speaking turns?",
             "Is the number of turns in this exchange greater than {n}?", "Do the speakers take more than {n} turns in total?",
             "Counting each change of speaker as a new turn, are there more than {n} turns?",
             "Does the conversation contain over {n} turns?", "Is the turn count here above {n}?",
             "Would you count more than {n} turns in this recording?"],
    "atleast": ["Are there at least {n} turns in this conversation?", "Does this dialog contain {n} or more turns?",
                "Is the number of speaking turns {n} or higher?", "Do the speakers take at least {n} turns in total?",
                "Counting a new turn at every change of speaker, are there {n} turns or more?",
                "Does this exchange reach at least {n} turns?", "Is the turn count at least {n}?",
                "Would you count {n} or more turns here?"],
    "exactly": ["Are there exactly {n} turns in this conversation?", "Does this dialog have exactly {n} speaking turns?",
                "Is the number of turns in this exchange exactly {n}?", "Do the speakers take exactly {n} turns in total?",
                "Counting each change of speaker as a new turn, are there precisely {n} turns?",
                "Is the turn count here equal to {n}?", "Would you count exactly {n} turns in this recording?",
                "Does the conversation consist of {n} turns, no more and no fewer?"],
    "fewer": ["Are there fewer than {n} turns in this conversation?", "Does this dialog have less than {n} speaking turns?",
              "Is the number of turns in this exchange below {n}?", "Do the speakers take fewer than {n} turns in total?",
              "Counting each change of speaker as a new turn, are there fewer than {n} turns?",
              "Does the conversation contain under {n} turns?", "Is the turn count here smaller than {n}?",
              "Would you count fewer than {n} turns in this recording?"],
}
TURN_SCORE = [
    "Roughly how many turns does this conversation have?",
    "Which range contains the number of speaking turns in this clip?",
    "How long is this exchange in terms of turns (a new turn at every change of speaker)?",
    "Pick the range that matches the number of turns in this dialog.",
    "About how many turns are taken in this recording?",
    "Into which band does the turn count of this conversation fall?",
    "How many turns, approximately, are there in this exchange?",
    "Which of these turn-count ranges fits this audio?",
]
TURN_BINS = [
    [("2 or 3", 2, 3), ("4 or 5", 4, 5), ("6 or 7", 6, 7), ("8 or more", 8, 99)],
    [("2 to 4 turns", 2, 4), ("5 to 7 turns", 5, 7), ("8 to 12 turns", 8, 12)],
    [("just 2 or 3 turns", 2, 3), ("4 to 6 turns", 4, 6), ("7 to 9 turns", 7, 9), ("10 or more turns", 10, 99)],
]

# speaker count
SPK_COUNT = [
    "How many different people are speaking in this recording?",
    "How many distinct voices can you hear in this clip?",
    "Count the different speakers in this audio. How many are there?",
    "How many separate individuals talk in this recording?",
    "What is the number of unique speakers heard here?",
    "How many different talkers appear in this clip, counting each person once?",
    "Listen carefully: how many people in total speak in this audio?",
    "How many distinct persons can be heard talking here?",
    "If you count each voice only once, how many speakers are there?",
    "How many unique voices are present in this audio clip?",
]
SPK_NOUL = {
    "more": ["Are there more than {n} different speakers in this audio?", "Do more than {n} distinct people talk in this clip?",
             "Is the number of different voices greater than {n}?", "Can you hear more than {n} separate speakers?",
             "Counting each person once, are there over {n} speakers?", "Does this recording feature more than {n} talkers?",
             "Are more than {n} unique voices present?", "Is the speaker count above {n}?"],
    "atleast": ["Are there at least {n} different speakers in this audio?", "Do {n} or more distinct people talk in this clip?",
                "Is the number of different voices {n} or higher?", "Can you hear at least {n} separate speakers?",
                "Counting each person once, are there {n} speakers or more?", "Does this recording feature at least {n} talkers?",
                "Are {n} or more unique voices present?", "Is the speaker count at least {n}?"],
    "exactly": ["Are there exactly {n} different speakers in this audio?", "Do exactly {n} distinct people talk in this clip?",
                "Is the number of different voices exactly {n}?", "Can you hear precisely {n} separate speakers?",
                "Counting each person once, are there exactly {n} speakers?", "Does this recording feature exactly {n} talkers?",
                "Are exactly {n} unique voices present?", "Is the speaker count equal to {n}?"],
    "fewer": ["Are there fewer than {n} different speakers in this audio?", "Do fewer than {n} distinct people talk in this clip?",
              "Is the number of different voices below {n}?", "Can you hear fewer than {n} separate speakers?",
              "Counting each person once, are there under {n} speakers?", "Does this recording feature fewer than {n} talkers?",
              "Are fewer than {n} unique voices present?", "Is the speaker count smaller than {n}?"],
}
SPK_SINGLE = [
    "Is only one person speaking in this recording?", "Is this the voice of a single speaker throughout?",
    "Does just one person talk in this clip?", "Is there only one talker in this audio?",
    "Is everything in this clip said by the same single person?", "Is this a single-speaker recording?",
    "Do all the words here come from one speaker?", "Is a lone speaker the only voice heard here?",
]
SPK_SCORE = [
    "Roughly how many people speak in this clip?", "Which range matches the number of different speakers here?",
    "About how many distinct voices are in this recording?", "Pick the band that contains the number of speakers.",
    "How many talkers, approximately, can be heard?", "Into which range does the number of different speakers fall?",
    "How crowded is this recording in terms of distinct speakers?", "Which of these speaker-count ranges fits the audio?",
]
SPK_BINS = [
    [("one", 1, 1), ("two or three", 2, 3), ("four or more", 4, 99)],
    [("1 or 2", 1, 2), ("3 or 4", 3, 4), ("5 or more", 5, 99)],
    [("a single speaker", 1, 1), ("two speakers", 2, 2), ("three to five speakers", 3, 5), ("six or more speakers", 6, 99)],
]
COUNT_FMT_TURN = ["{n}", "{n} turns", "{w}", "{w} turns"]
COUNT_FMT_CHANGE = ["{n}", "{n} times", "{w}", "{w} changes"]
COUNT_FMT_SPK = ["{n}", "{n} speakers", "{w}", "{w} voices", "{n} different people", "{w} speakers"]

# who spoke most
MOST_TIME = [
    "Which speaker talks for the longest total time? (Speakers are numbered by when they first speak.)",
    "Who does most of the talking, counting speakers in the order they first appear?",
    "Numbering the speakers by their first appearance, which one speaks for the most time overall?",
    "Which person holds the floor the longest in total? Speakers are ordered by who talks first.",
    "Which of the speakers (ordered by first appearance) has the greatest total speaking time?",
    "Who talks the most in this conversation, where 'first speaker' means the one who speaks first?",
    "Across the whole clip, which speaker's voice is heard for the longest time? (Order = first appearance.)",
    "Which speaker dominates the talking time? Speakers are labelled by the order in which they start talking.",
]
MOST_TURNS = [
    "Which speaker takes the most turns? (Speakers are numbered by when they first speak.)",
    "Who speaks up the most often, counting turns, with speakers ordered by first appearance?",
    "Numbering the speakers by their first appearance, which one has the largest number of turns?",
    "Which person takes the floor most often? Speakers are ordered by who talks first.",
    "Which of the speakers (ordered by first appearance) gets the most separate turns?",
    "Who has the most turns in this conversation, where 'first speaker' means the one who speaks first?",
    "Which speaker comes in the most times? (Order = first appearance.)",
    "Counting turns, which speaker is the most active? Speakers are labelled by the order in which they start talking.",
]
SPK_REF = ["the {o} speaker", "the {o} person to talk", "the {o} voice", "speaker {n}"]

# speaker identity (numbered clips)
SEP_INTRO = {
    "beep": ["The audio plays several short {noun}s, separated by beeps. ", "Several {noun}s follow one another, with a beep between them. ",
             "You will hear a series of {noun}s split by beep tones. ", ""],
    "spoken": ["Each {noun} is announced by its number. ", "A voice announces the number of every {noun}. ",
               "The {noun}s are introduced by number. ", ""],
    "pause": ["Several {noun}s are played one after another with short pauses in between. ",
              "The audio is a sequence of {noun}s separated by silence. ", "You will hear {noun}s back to back, separated by pauses. ", ""],
}
SPKID_MATCH = [
    "Which {noun} is spoken by the same person as {noun} {n}?",
    "Which other {noun} has the same speaker as {noun} {n}?",
    "{Noun} {n} was recorded by one person. Which other {noun} is that same person?",
    "Find the {noun} whose voice matches {noun} {n}. Which one is it?",
    "The speaker of {noun} {n} also appears in one other {noun}. Which one?",
    "Which {noun} comes from the same talker as {noun} {n}?",
    "Whose voice in the other {noun}s matches the voice in {noun} {n}? Pick the {noun}.",
    "Which {noun} features the same voice as {noun} {n}?",
    "Same voice as {noun} {n}: which {noun} is it?",
    "Listen to {noun} {n}. Which of the other {noun}s is spoken by that person too?",
]
SPKID_NONE = ["none of the other {noun}s", "no other {noun} matches", "none of them"]
SPKID_PAIR = [
    "Two of the {noun}s are spoken by the same person. Which two?",
    "Which pair of {noun}s shares the same speaker?",
    "Exactly one speaker appears twice among these {noun}s. In which {noun}s?",
    "Which two {noun}s have the same voice?",
    "One person speaks in two different {noun}s. Which ones?",
    "Which {noun}s were recorded by the same talker?",
    "Pick the two {noun}s that come from one and the same speaker.",
    "Which pair of {noun}s is the same person talking?",
]
SPKID_NOUL = [
    "Are {noun} {a} and {noun} {b} spoken by the same person?",
    "Is the speaker of {noun} {a} the same as the speaker of {noun} {b}?",
    "Do {noun} {a} and {noun} {b} have the same voice?",
    "Is {noun} {b} the same talker as {noun} {a}?",
    "Does the person in {noun} {a} also speak in {noun} {b}?",
    "Are {noun} {a} and {noun} {b} from one and the same speaker?",
    "Would you say {noun} {a} and {noun} {b} are the same person?",
    "Is the voice in {noun} {a} identical to the voice in {noun} {b}?",
]
SPKID_COUNT = [
    "How many different speakers are heard across all the {noun}s?",
    "Counting each person once, how many distinct voices appear in these {noun}s?",
    "How many separate people recorded these {noun}s?",
    "Across the {noun}s, how many unique talkers are there?",
    "How many different individuals speak in this series of {noun}s?",
    "If you group the {noun}s by speaker, how many different speakers do you get?",
    "How many distinct speakers do the {noun}s come from?",
    "How many different voices are in this sequence of {noun}s?",
]
CLIP_OPT = ["{noun} {n}", "the {o} {noun}", "{Noun} {n}", "number {n}"]

# same speaker (two segments)
SAME_NOUL = [
    "Are the two {noun}s spoken by the same person?",
    "Is the same speaker heard in both {noun}s?",
    "Do both {noun}s come from one speaker?",
    "Is the second {noun} the same voice as the first?",
    "Do the first and second {noun}s have the same talker?",
    "Is it one and the same person speaking in both {noun}s?",
    "Would you say both {noun}s are the same speaker?",
    "Does the voice in the second {noun} belong to the person from the first {noun}?",
    "Are these two {noun}s recorded by the same individual?",
]
SAME_CHOICE = [
    "Are the two {noun}s from the same speaker or from two different speakers?",
    "Same speaker or different speakers: what are these two {noun}s?",
    "How many different people speak across the two {noun}s?",
    "Do the two {noun}s share a speaker?",
    "Which describes the two {noun}s?",
    "Is the voice in both {noun}s the same?",
    "Compare the speakers of the two {noun}s. What do you find?",
    "What is the relationship between the speakers of the two {noun}s?",
]
SAME_OPTS = [("the same speaker", "two different speakers"), ("same person", "different people"),
             ("one speaker", "two speakers"), ("same voice", "different voices")]
NOUNS = ["clip", "recording", "sample", "voice"]
