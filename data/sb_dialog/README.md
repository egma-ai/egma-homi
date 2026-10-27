# sb_dialog

Conversation-flow questions about phone calls and meetings (20-180 s clips): talk-over, who talked more, silences, slow
responses, who spoke first or last, how many people speak.

## Questions
- Two-party calls (HVB): `a_over`, `c_over`, `long_sil`, `slow_resp` (yes/no); `a_over_when`, `c_over_when`, `over_who`,
  `more_talk`, `first_spk`, `last_spk`, `sil_where` (choice); `agent_share`, `ov_amount` (score).
- Multi-party (AMI, Earnings-21): `mp_overlap`, `long_sil`, `same_first_last`, `dominant` (yes/no); `mp_overlap_when`,
  `n_spk`, `sil_where` (choice); `ov_amount` (score).
- Labels come from per-speaker timings (word alignments, reference diarization, or the timings of HVB calls we re-laid
  with planted talk-overs, silences and slow responses). Every threshold has a margin; ambiguous cases are not asked,
  and HVB labels that disagree with a per-channel energy VAD are dropped. Yes/no families are 50/50; at most 4 per clip.

## Sources
| Source | Use | Licence | Revision |
|---|---|---|---|
| HarperValleyBank | two-party bank calls, separate agent / caller channels, word timings | CC BY 4.0 (repo LICENSE) | `cricketclub/gridspace-stanford-harper-valley@0bd721e877c4a85d8c13ff837e68661ea6200a98` |
| AMI Meeting Corpus | Mix-Headset audio + manual word alignments (ami_public_manual_1.6.2), official mirror | CC BY 4.0 | HF mirror `edinburghcstr/ami@46f28f2503e2ec48f8867a84eef356c70476beab` |
| Earnings-21 | earnings calls + reference RTTM diarization | CC BY-SA 4.0 (share-alike; excluded from training, see DATA.md) | `revdotcom/speech-datasets@c05ab6fd8b4b627d123c922a22a39e993dd37635` |

Held out: every SLUE-HVB test conversation (`asapp/slue-phase-2@e2989c55a53593a8e39b8f8ebdb47ccaccbe484a`), which covers
all HVB calls in our evaluation set. CAL uses disjoint HVB speakers, AMI meeting groups and Earnings-21 calls.

## Run
    modal run --detach data/sb_dialog/modal_build.py::full    # ::smoke for a small run, ::prefetch to download only

Output on the `homi` volume: `/vol/stageb/sb_dialog/` (`items_train.jsonl`, `items_cal.jsonl`, `audio/`,
`summary.json`).

## Tests
    uv run --python 3.12 --with pytest --with numpy --with scipy --with soundfile pytest data/sb_dialog/tests
