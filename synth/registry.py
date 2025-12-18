from .LN_synth import dm_ln_synthesize
from .BN_synth import dm_bn_synthesize
from .coreset_construction import (
    random_ipc_synthesize,
    vq_synthesize,
    voronoi_synthesize,
    gonzalez_synthesize,
)
from .BatchNorm_stats_synth import batchnorm_stats_synthesize
from .moments_synth import dm_moment_synthesize

SYNTH_REGISTRY = {
    # -------------------------
    # Condensation methods
    # -------------------------
    "dm_ln": {
        "fn": dm_ln_synthesize,
        "type": "condensation",
        "teacher_required": False,
    },
    "dm_bn": {
        "fn": dm_bn_synthesize,
        "type": "condensation",
        "teacher_required": False,
    },

    # -------------------------
    # Teacher-based synthesis
    # -------------------------
    "bn_stats": {
        "fn": batchnorm_stats_synthesize,
        "type": "teacher_matching",
        "teacher_required": True,
    },

    # -------------------------
    # Coreset baselines
    # -------------------------
    "random_ipc": {
        "fn": random_ipc_synthesize,
        "type": "coreset",
        "teacher_required": False,
    },
    "vq": {
        "fn": vq_synthesize,
        "type": "coreset",
        "teacher_required": False,
    },
    "voronoi": {
        "fn": voronoi_synthesize,
        "type": "coreset",
        "teacher_required": False,
    },
    "gonzalez": {
        "fn": gonzalez_synthesize,
        "type": "coreset",
        "teacher_required": False,
    },

    # -------------------------
    # Moments synthesis
    # -------------------------
    "dm_moments": {
    "fn": dm_moment_synthesize,
    "type": "condensation",
    "teacher_required": False,
}
}


def synthesize(synth_type, data, config):
    if synth_type not in SYNTH_REGISTRY:
        raise ValueError(
            f"Unknown synth type '{synth_type}'. "
            f"Available: {list(SYNTH_REGISTRY.keys())}"
        )

    entry = SYNTH_REGISTRY[synth_type]

    if entry["teacher_required"]:
        assert "teacher" in data, (
            f"Synth '{synth_type}' requires a trained teacher model "
            f"in data['teacher']"
        )

    return entry["fn"](data, config)