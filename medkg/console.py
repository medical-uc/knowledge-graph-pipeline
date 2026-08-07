"""Progress reporting shared by every stage.

The pipeline is long-running and, between artifacts, almost entirely silent:
Stage 2 issues one LLM call per chunk, Stage 3 loads FAISS and SapBERT before
it links anything, and Stage 7 materializes a closure over the whole corpus.
Minutes of that are indistinguishable from a hang, so each stage narrates what
it is doing through the helpers here instead of printing on its own.

Progress is reported as batched counts on ordinary lines (`20/96 chunks
tagged`). Carriage-return redraws and progress bars are unreadable on
terminals that do not support them and worse than useless in a captured log,
so neither is used.

Messages are nested by indentation: a stage header sits at the left margin,
the steps inside it one level in, and per-step counts and results one level
further. `set_base_indent_level` shifts the whole scheme, which is what lets a
corpus run print its stages underneath a per-document banner.
"""
from __future__ import annotations

import math
import time

INDENT = "  "
STAGE_COUNT = 8

_base_indent_level = 0
_quiet = False


def set_quiet(quiet: bool = True) -> bool:
    """Silence (or unsilence) every message emitted through this module.

    Returns the previous setting, so a caller that only wants to borrow a
    stage as a library function can restore whatever was in force.
    """
    global _quiet
    previous, _quiet = _quiet, quiet
    return previous


def is_quiet() -> bool:
    """Whether messages are currently being suppressed."""
    return _quiet


def set_base_indent_level(level: int) -> int:
    """Indent every subsequent message by `level` further levels.

    Returns the previous level so the caller can restore it. Negative levels
    are clamped to zero.
    """
    global _base_indent_level
    previous, _base_indent_level = _base_indent_level, max(0, level)
    return previous


def write(message: str = "", indent_level: int = 0) -> None:
    """Print one line at `indent_level`, unless output is silenced.

    Flushed on every call: stdout is block-buffered when redirected to a file,
    and progress that only appears once the run has ended is not progress.
    """
    if _quiet:
        return
    prefix = INDENT * (_base_indent_level + indent_level) if message else ""
    print(prefix + message, flush=True)


def announce_stage(number: int, title: str, detail: str = "") -> None:
    """Announce the stage that is starting: `stage 2/8  extract  ...`.

    `number` is the stage's position in the pipeline, not a count of the
    stages this particular run happens to execute, so a `from-stage3` run
    still names Stage 4 as Stage 4.
    """
    write("")
    write(f"stage {number}/{STAGE_COUNT}  {title}"
          + (f"  {detail}" if detail else ""))


def announce_step(message: str) -> None:
    """Announce a unit of work within the running stage."""
    write(message, indent_level=1)


def announce_detail(message: str) -> None:
    """Report a count, a result or a note underneath the running step."""
    write(message, indent_level=2)


def progress_interval(total_count: int, target_update_count: int = 10) -> int:
    """How many items to process between progress lines.

    Chosen so a loop reports roughly `target_update_count` times, rounded to
    the nearest 1, 2 or 5 times a power of ten. That rounding is what keeps
    the counts readable (`20/96`, `40/96`) rather than arithmetically exact
    and unscannable (`9/96`, `18/96`). Never returns less than 1.
    """
    if total_count <= target_update_count:
        return 1
    rough = total_count / target_update_count
    magnitude = 10 ** int(math.floor(math.log10(rough)))
    candidates = [multiple * magnitude for multiple in (1, 2, 5, 10)]
    return int(min(candidates, key=lambda size: abs(math.log(size / rough))))


def report_progress(noun: str, completed_count: int, total_count: int,
                    interval: int = None) -> None:
    """Print `20/96 chunks tagged` on batch boundaries and on the last item.

    Call it once per item and it decides whether a line is warranted, so no
    caller has to carry the batching arithmetic. `interval` overrides the
    batch size `progress_interval` would pick. An empty total prints nothing.
    """
    if total_count <= 0:
        return
    every = interval or progress_interval(total_count)
    if completed_count % every and completed_count != total_count:
        return
    announce_detail(f"{completed_count}/{total_count} {noun}")


def with_progress(items, noun: str, interval: int = None):
    """Yield each of `items`, reporting progress as the loop consumes them.

    The count is reported after the body has run, including for the last item,
    so a loop full of `continue`s does not have to repeat a progress call in
    every branch. `items` must be a sized sequence. `noun` and `interval` mean
    what they do in `report_progress`.
    """
    total = len(items)
    for completed_count, item in enumerate(items, start=1):
        yield item
        report_progress(noun, completed_count, total, interval)


def format_count(count: int, noun: str, plural: str = None) -> str:
    """`1, 'span'` -> `1 span`; `4, 'span'` -> `4 spans`.

    Thousands are separated, because the counts this reports on (atoms,
    triples) routinely run to seven figures. `plural` overrides the default
    `noun + "s"`.
    """
    word = noun if count == 1 else (plural or noun + "s")
    return f"{count:,} {word}"


def format_duration(seconds: float) -> str:
    """`8.2` -> `8.2s`; `92` -> `1m 32s`; `4210` -> `1h 10m`."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def announce_finished(stage_title: str, started_at: float,
                      summary: str = "") -> None:
    """Close a stage with its wall-clock cost and, optionally, its yield.

    `started_at` is a `time.time()` reading taken when the stage began. The
    timing is worth printing on its own: it is what tells an operator whether
    a slow run is slow in the LLM passes, which cost money, or in the local
    ones, which do not.
    """
    elapsed = format_duration(max(0.0, time.time() - started_at))
    announce_step(f"{stage_title} done in {elapsed}"
                  + (f": {summary}" if summary else ""))
