"""Leakage firewall tests — Step 0.3.

The plan requires that these be seen to FAIL on a deliberately poisoned split
before they are trusted. `test_poisoned_*` are those tests: each injects known
held-out material and asserts the firewall catches it. A silent pass on a
poisoned split would mean the firewall is decorative.
"""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from biodreamer.data.firewall import (
    LeakageError,
    assert_clean,
    check_assay_ids,
    check_instance,
    check_sequences,
    load_registry,
    translate,
)

REGISTRY = "splits/heldout.yaml"
PG_REF = "data/raw/proteingym/DMS_substitutions_reference.csv"

# verified by mmseqs in step 0.3; see splits/heldout.yaml
KNOWN_LEAKS = [
    "SPG1_STRSG_Olson_2014",
    "SPG1_STRSG_Wu_2016",
    "SPG2_STRSG_Tsuboyama_2023_5UBS",
    "F7YBW8_MESOW_Aakre_2015",
    "F7YBW8_MESOW_Ding_2023",
    "DYR_ECOLI_Nguyen_2023",
    "DYR_ECOLI_Thompson_2019",
]


@pytest.fixture(scope="module")
def registry():
    return load_registry(REGISTRY)


@pytest.fixture(scope="module")
def pg_sequences():
    """ProteinGym assay id -> target sequence, if the reference file is present."""
    try:
        df = pd.read_csv(PG_REF)
    except FileNotFoundError:
        pytest.skip("ProteinGym reference not downloaded; see DATA_REGISTER.md")
    return dict(zip(df.DMS_id, df.target_seq.astype(str), strict=True))


# ---- registry integrity ----


def test_registry_loads_and_is_complete(registry):
    assert registry.identity_threshold == pytest.approx(0.70)
    names = {e["name"] for e in registry.held_out}
    assert {"TrpB4", "TEV", "GB1", "ParD2", "ParD3"} <= names
    assert set(KNOWN_LEAKS) == registry.banned_assays
    assert registry.banned_flip2 == {"gb1", "trpb"}


def test_phoq_is_not_in_the_registry(registry):
    """PhoQ was withdrawn in 0.2; it must not creep back as a held-out landscape."""
    assert "PhoQ" not in {e["name"] for e in registry.held_out}


def test_every_leak_in_registry_is_reachable(registry):
    """Each declared leak must name a real source and a plausible identity."""
    for entry in registry.held_out:
        for leak in entry.get("leaks_into") or []:
            assert leak["source"] in {"ProteinGym", "FLIP2"}
            assert 0.5 <= leak["identity"] <= 1.0
            if leak["source"] == "ProteinGym":
                assert leak["id"] in registry.banned_assays


# ---- id-level checks ----


def test_clean_split_passes(registry):
    clean = ["BLAT_ECOLX_Stiffler_2015", "CALM1_HUMAN_Weile_2017"]
    assert check_assay_ids(clean, registry) == []
    assert_clean(training_ids=clean, registry=registry)  # must not raise


@pytest.mark.parametrize("assay", KNOWN_LEAKS)
def test_poisoned_split_is_caught_for_every_known_leak(registry, assay):
    """THE POISON TEST. Each known leak, injected alone, must be caught."""
    poisoned = ["BLAT_ECOLX_Stiffler_2015", assay]
    leaks = check_assay_ids(poisoned, registry)
    assert len(leaks) == 1 and leaks[0].identifier == assay
    with pytest.raises(LeakageError, match=assay):
        assert_clean(training_ids=poisoned, registry=registry)


def test_flip2_gb1_and_trpb_are_banned(registry):
    """FLIP2 was added as a training source and contains two held-out landscapes."""
    leaks = check_assay_ids(["aav", "gb1", "trpb", "ired"], registry, source="FLIP2")
    assert {x.identifier for x in leaks} == {"gb1", "trpb"}


# ---- sequence-level checks: the renamed-copy case ----


def test_sequence_check_catches_a_renamed_copy(registry):
    """An id check cannot catch the same protein under a different name."""
    gb1 = registry.sequences(".")["GB1"]
    poisoned = {"INNOCENT_LOOKING_NAME_2024": gb1}
    leaks = check_sequences(poisoned, registry, ".")
    assert leaks and leaks[0].held_out == "GB1"
    assert leaks[0].identity == pytest.approx(1.0)


def test_sequence_check_passes_an_unrelated_protein(registry):
    unrelated = {"SOME_ASSAY": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSG"}
    assert check_sequences(unrelated, registry, ".") == []


def test_dhfr_nucleotide_entry_is_translated(registry):
    """SSMuLA's DHFR is DNA; an untranslated check silently misses DYR_ECOLI."""
    seqs = registry.sequences(".")
    dhfr = seqs.get("DHFR", "")
    assert dhfr.startswith("MISLIAALAVDRVIGME"), "DHFR was not translated"
    assert set(dhfr) - set("ACGT"), "DHFR still looks like nucleotides"


def test_translate_handles_stop_codons():
    assert translate("ATGATCAGTTAA") == "MIS"
    assert translate("ATGATCAGT") == "MIS"


@pytest.mark.parametrize("assay", ["SPG1_STRSG_Olson_2014", "DYR_ECOLI_Thompson_2019"])
def test_real_proteingym_sequences_are_flagged(registry, pg_sequences, assay):
    """End-to-end against the real corpus, not synthetic strings."""
    leaks = check_sequences({assay: pg_sequences[assay]}, registry, ".")
    assert leaks, f"{assay} should be flagged by sequence identity"


def test_the_whole_proteingym_corpus_has_no_unknown_leaks(registry, pg_sequences):
    """Regression guard: a new leak appearing here means the registry is stale."""
    allowed = registry.banned_assays
    clean = {k: v for k, v in pg_sequences.items() if k not in allowed}
    leaks = check_sequences(clean, registry, ".")
    assert not leaks, (
        "sequence check found leaks outside the registry — update splits/heldout.yaml:\n  "
        + "\n  ".join(str(x) for x in leaks)
    )


# ---- two-instance rule ----


def test_planning_instance_allowed_in_phase_6(registry):
    assert check_instance("planning", "6.3", registry) == []


def test_benchmark_instance_refused_in_phase_6(registry):
    """A benchmark checkpoint may train on banned data, so it must never reach H4."""
    leaks = check_instance("benchmark", "6.3", registry)
    assert leaks and "may not produce" in leaks[0].reason
    with pytest.raises(LeakageError):
        assert_clean(instance="benchmark", phase="6.3", registry=registry)


def test_unknown_instance_is_refused(registry):
    assert check_instance("typo", "1.1", registry)


def test_registry_file_is_valid_yaml():
    yaml.safe_load(open(REGISTRY).read())
