# sb_calltext

Whole-call questions about what was said and how the caller felt, labelled from the transcript.

## Questions
- HarperValleyBank calls: caller sentiment and emotion, frustration, issue resolved, call reason, personal details
  asked, price asked, next step agreed. Earnings-22 windows: Q&A or prepared remarks, tone, news, topic, figures.
- Labels come from `Qwen/Qwen3.6-35B-A3B-FP8` @ `95a723d08a9490559dae23d0cff1d9466213d989` (Apache-2.0, vLLM 0.31.0,
  greedy), asked twice with the options in canonical and reversed order. A label is kept only if both runs pick the
  same option with probability >= 0.7; the target is the mean of the two distributions.

## Sources
| Source | Use | Licence | Revision |
|---|---|---|---|
| HarperValleyBank | whole calls (mono mix + codec copy), human transcripts | CC BY 4.0 | `cricketclub/gridspace-stanford-harper-valley@0bd721e877c4a85d8c13ff837e68661ea6200a98` |
| Earnings-22 | 20-180 s windows, force-aligned transcripts | CC BY-SA 4.0 (share-alike, flagged; left out of egma-homi training, see DATA.md) | `revdotcom/speech-datasets@c05ab6fd8b4b627d123c922a22a39e993dd37635` |
| AMI Meeting Corpus | code path only, not in the published data | CC BY 4.0 | `ami_public_manual_1.6.2` |

Held out: every SLUE-HVB test conversation (`asapp/slue-phase-2@e2989c55a53593a8e39b8f8ebdb47ccaccbe484a`) and every
HVB call in our evaluation set. CAL (~3%) is conversation-disjoint (HVB) and call-disjoint (Earnings-22).

## Run and test
    modal run --detach data/sb_calltext/modal_build.py::full     # prep -> label (1 x H100) -> assemble
    uv run --python 3.12 --with pytest --with numpy --with scipy --with soundfile pytest data/sb_calltext/tests
The build writes `/vol/stageb/sb_calltext/` on the `egma-homi` volume: `items_train.jsonl`, `items_cal.jsonl`,
`audio/`, `summary.json`.
