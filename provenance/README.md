# Provenance of the scored submissions

Each folder holds the code, Dockerfile and weight metadata that were encrypted into the pull request
(the weights themselves are release assets; see `tools/fetch_weights.py`).

| Folder | Pull request | Commit | Weights |
|---|---|---|---|
| `pr34_v1/` | #34 | 8a05f77 | v1 soup |
| `pr35_v2/` | #35 | 079c843 | v2 soup |
| `pr38_v2+v2s1/` | #38 | 4bc2a56 | v2 and v2s1 soups, `weights/ensemble.json` |
| `pr39_v2+v4/` | #39 | b47bc3f | v2 and v4 soups, `weights/ensemble.json` |

`submission_log.txt` lists the SHA-256 of every file that was encrypted, including the weights and the
encrypted parts, at the time of each submission. PR #34 and #35 used the same code; #38 and #39 use the
later code that adds ensemble support (identical outputs for a single model).
