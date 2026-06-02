# fake-video-corpus
This is the first, to our knowledge, annotated dataset of debunked and verified user-generated videos (UGVs), along with multiple near-duplicate reposted versions of them. For details refer to [Fake Video Corpus](https://mklab.iti.gr/results/fake-video-corpus/).

The dataset comprises videos from a variety of event categories, such as politics, sports, natural disasters, accidents, wars, etc. Currently, it consists of 200 unique debunked videos (for simplicity also referred to as fake) and 180 unique verified videos (also referred to as real). In particular, different types of fake video are included:

- Staged videos where actors perform scripted actions under direction.
- Videos where contextual information is false (e.g. the claimed video location is wrong).
- Past videos presented as UGV from breaking events.
- Videos of which the visual or audio content has been altered through editing.
- Computer-generated Imagery (CGI) posing as real.

The dataset was extended following a largely automatic systematic process that combines text search and near-duplicate video retrieval, followed by manual annotation using a set of guidelines. More specifically:

1. For each video in the original set, the video title was used as input.
2. The title was reformulated to a more general form (called the “event title”). For example, a video with title “Video Tornado IRMA en Florida EEUU Video impactante” was assigned to event “Tornado IRMA at Florida”.
3. The event title was translated from English into four major languages: Russian, Arabic, French, and German using Google Translate. These languages were selected after preliminary tests indicated that near-duplicate videos appear with increased frequency in these languages.
4. The video title, event title, and the four translations were used as separate queries to the three target platforms: YouTube, Facebook, Twitter. All returned videos were aggregated in a common pool.
5. A near-duplicate retrieval algorithm was used to search within this pool for near-duplicates of the video.
6. After manual inspection, erroneous results were removed and only actual near-duplicates were retained.

**The overall dataset consist of 3957 videos annotated as fake and 2458 annotated as real.**

| Categories for near-duplicates of fake videos include | Categories for near-duplicates of real videos include |
| ------------- |:-------------:| 
|Fake: those that reproduce the same false claims | Real: those that reproduce the same factual claims|
|Uncertain: those that express doubts on the veracity of the claim | Uncertain: those that express doubts on the veracity of the claim|
|Debunk: those that attempt to debunk the original claim | Debunk: those that attempt to debunk their claims as false|
|Parody: those that use the content for fun/entertainment | Parody: those that use the content for fun/entertainment|
|Real: those that contain the earlier, original source from which the fake was made| |

Facebook videos that were relevant to the dataset but were published by individual users (and thus could not be accessed through the API) were excluded from this dataset.

**Dataset**

The initial 200 fake and 180 real videos are contained in FVC.csv.

The near duplicates are contained in FVC_dup.csv.

The text queries for retrieving the near duplicates are contained in FVC_text_queries.csv.

**Downloading YouTube videos**

Install the downloader dependency:

```bash
python -m pip install -r requirements.txt
```

Download every URL listed in `FVC.csv` sequentially into
`/media/yingqichao/Lenovo/FVC/<label>/`:

```bash
python scripts/download_fvc_youtube.py
```

By default, files are named with their cascade id, for example
`/media/yingqichao/Lenovo/FVC/fake/f14.mp4`. The script checks existing output
files and `/media/yingqichao/Lenovo/FVC/downloaded.txt`, skips rows that were
already downloaded, shows a `tqdm` progress bar, and waits a random `0` to `1`
seconds between download attempts.

Download or test a single URL:

```bash
python scripts/download_fvc_youtube.py --url "https://www.youtube.com/watch?v=jGuDpD-Y-1s"
```

You can also select rows by cascade id, for example:

```bash
python scripts/download_fvc_youtube.py --id f14
```

The script writes `/media/yingqichao/Lenovo/FVC/download_report.csv`. To change
the delay range, pass `--interval-min` and `--interval-max`.

For age-restricted or private videos, run the downloader with cookies from a
browser where you are already signed in to an age-verified YouTube account:

```bash
python scripts/download_fvc_youtube.py --cookies-from-browser chrome
```

Firefox works too:

```bash
python scripts/download_fvc_youtube.py --cookies-from-browser firefox
```

If your signed-in session is not in the default browser profile, pass the
profile name, for example:

```bash
python scripts/download_fvc_youtube.py --cookies-from-browser "chrome:Profile 1"
```

Alternatively, export a Netscape-format `cookies.txt` file and pass it with
`--cookies /path/to/cookies.txt`.

Some original seed URLs are old, unavailable, or age-gated. To keep `FVC.csv` as
the source of labels while trying same-cascade, same-label YouTube reposts when a
seed URL fails, pass `FVC_dup.csv` as a fallback source:

```bash
python scripts/download_fvc_youtube.py \
  --cookies-from-browser chrome+GNOMEKEYRING \
  --fallback-duplicates FVC_dup.csv
```

Audit the alignment between the seed videos, near-duplicates, and query metadata
with:

```bash
python scripts/audit_fvc_alignment.py
```

**Title-only NVIDIA LLM inference**

The repo includes a title-only inference script that reads `event_title` from
`FVC_text_queries.csv`, asks the NVIDIA/OpenAI-compatible chat API to predict
only `real` or `fake`, stores the model reasoning, and compares the prediction
against the CSV `label` column.

The system prompt lives in `scripts/fvc_language_prompt.txt`. You can edit that
file directly or pass another prompt with `--prompt-file`.

Credentials are read from `.env` or the shell environment. `.env` is intentionally
ignored by git. A local `.env` can be created from `.env.example`:

```bash
cp .env.example .env
```

Set either `NVIDIA_API_KEY` or `OPENAI_API_KEY` in `.env`. The default endpoint
and model match the NVIDIA inference API pattern used in the related
FakeSV-Thinking-Dev repo:

```text
NVIDIA_API_URL=https://inference-api.nvidia.com/v1/chat/completions
NVIDIA_MODEL=openai/openai/gpt-5.1
```

Run a one-row smoke test:

```bash
python scripts/infer_fvc_titles_nvidia.py \
  --id f0 \
  --output outputs/fvc_title_llm_predictions_smoke.json
```

Run all 380 seed titles:

```bash
python scripts/infer_fvc_titles_nvidia.py \
  --resume \
  --output outputs/fvc_title_llm_predictions.json
```

The output JSON contains the prompt, local title-pattern analysis, per-row
prediction, reasoning, raw model response, and correctness summary. The prompt
uses only the title text and includes moderate corpus-level shortcut cues for
obvious fake patterns such as paranormal/cryptid wording, impossible spectacle
claims, and highly sensational military-strike phrasing. Some rows in
`FVC_text_queries.csv` contain unquoted commas; the script repairs obvious
comma-split `event_title` fragments before sending the title to the model.

**License and acknowledgement**

The video dataset is provided under the Attribution-NonCommercial-ShareAlike 4.0 International [(CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

The video dataset is supported by the [InVID project](https://www.invid-project.eu/), which is funded by the European Commission under contract number 687786.

If you use this video dataset for your research, please include a citation to the following paper: Papadopoulou, O., Zampoglou, M., Papadopoulos, S., & Kompatsiaris, Y. (2018). [A Corpus of Debunked and Verified User-Generated Videos](https://mklab.iti.gr/results/fake-video-corpus/OIR.pdf). Online Information Review. Accepted for publication.

    @article{papadopoulou2018corpus,
      author = "Papadopoulou, Olga and Zampoglou, Markos and Papadopoulos, Symeon and Kompatsiaris, Ioannis",
      title = "A corpus of debunked and verified user-generated videos",
      journal = "Online Information Review",
      doi = "10.1108/OIR-03-2018-0101",
      year={2018},
      publisher={Emerald Publishing Limited}
    }


If you encounter any issues in this process, please get in touch with Olga Papadopoulou <olgapapa@iti.gr>.
