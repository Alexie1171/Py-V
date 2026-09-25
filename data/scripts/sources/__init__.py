"""
sources — PY-V Data Pipeline v2
One module per training-data source. Each exposes
    iter_records(cfg: dict, stats: Counter) -> Iterator[dict]
yielding PY-V records (instruction / output / metadata incl. task label).
`cfg` is that source's block under dataset_v2.sources in config.yaml.
"""

from data.scripts.sources import (
    bug_fix,
    code_search_net,
    commitpack_refactor,
    glaive,
    improve_synthetic,
    long_file_fix,
    old_github,
    opencodeinstruct,
    self_oss_instruct,
)

SOURCES = {
    "opencodeinstruct":  opencodeinstruct.iter_records,
    "self_oss_instruct": self_oss_instruct.iter_records,
    "old_github":        old_github.iter_records,
    "glaive":            glaive.iter_records,
    "code_search_net":   code_search_net.iter_records,
    "bug_fix":             bug_fix.iter_records,            # runs code — Colab only
    "improve_synthetic":   improve_synthetic.iter_records,  # runs code — Colab only
    "commitpack_refactor": commitpack_refactor.iter_records,
    "long_file_fix":       long_file_fix.iter_records,      # runs code — Colab only
}
