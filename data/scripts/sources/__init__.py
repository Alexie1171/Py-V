"""
sources — PY-V Data Pipeline v2
One module per training-data source. Each exposes
    iter_records(cfg: dict, stats: Counter) -> Iterator[dict]
yielding PY-V records (instruction / output / metadata incl. task label).
`cfg` is that source's block under dataset_v2.sources in config.yaml.
"""

from data.scripts.sources import (
    code_search_net,
    glaive,
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
}
