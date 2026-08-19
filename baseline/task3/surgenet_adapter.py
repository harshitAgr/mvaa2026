#!/usr/bin/env python3
"""Thin adapter around the pinned upstream SurgeNet / MetaFormerFPN code.

Upstream: https://github.com/timjaspers0801/surgenet, vendored as a pinned git submodule at
`third_party/surgenet` (commit `54831cfacf3a6cac0bbf5119f72b9cb9c31c3e82` -- see
the repository README). In the Docker image, the same two files (`metaformer.py`,
`LICENSE.txt`) are mirrored byte-for-byte at `docker/code/third_party/surgenet/` (the build
context is scoped to `docker/`, so the repo-root submodule is out of reach for `docker build`;
see the repository README).

This file is the ONLY code we author that touches the upstream API surface.
Upstream files are never edited. This module:
  (a) puts the pinned upstream directory on `sys.path`;
  (b) defensively pre-registers `timm.models.layers` / `timm.models.registry` aliases *before*
      importing the upstream module, in case a future timm release drops the deprecated
      compat modules that upstream's `metaformer.py` imports from (timm 1.0.26/1.0.27, the
      versions this repo pins in Docker/venv respectively, both still expose them with only a
      FutureWarning -- verified 2026-07-10 -- so this is a no-op today, kept for robustness);
  (c) re-exports `caformer_s18` and `MetaFormerFPN` unchanged;
  (d) provides `load_surgenet_into_model(model, ckpt_path)`, a hardened *local-file-only*
      loader (never fetches a URL) implementing every documented load check (its missing-keys
      audit is the authoritative, always-on StarReLU guard), and `run_golden_activation_test(...)`,
      the Sec.4.1 fixture -- see its docstring for what each of its three sub-checks proves.

THE CRITICAL CORRECTNESS REQUIREMENT: stock timm `caformer_s18` uses StarReLU; the SurgeNet
checkpoint was pretrained with plain `nn.ReLU`. Callers MUST construct with
`pretrained='SurgeNet'` (see `model_factory.py`'s `metaformerfpn` branch). This module's
`load_surgenet_into_model` raises loudly if it detects the wrong (StarReLU) config was used
(activation-affine keys showing up in `missing_keys`).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SURGENET_DIR = _REPO_ROOT / "third_party" / "surgenet"

if not (_SURGENET_DIR / "metaformer.py").exists():
    raise ImportError(
        f"Pinned upstream SurgeNet repo not found at {_SURGENET_DIR}. "
        "On the host: run `git submodule update --init third_party/surgenet`. "
        "In Docker: confirm docker/code/third_party/surgenet/metaformer.py was copied "
        "(see the repository README)."
    )

if str(_SURGENET_DIR) not in sys.path:
    # append (not insert at position 0): the vendored upstream directory ships bare, generic
    # module names (metaformer.py, convnextv2.py, pvtv2.py, load_models.py) that would otherwise
    # take import priority over any same-named module of ours/a dependency's for the rest of the
    # process. Appending yields priority to everything already on sys.path (our own baseline/
    # code, site-packages, etc.) while still making the upstream import below resolve -- Python
    # just finds it later in the search instead of first.
    sys.path.append(str(_SURGENET_DIR))


def _ensure_deprecated_timm_shims() -> None:
    """Pre-register `timm.models.layers` / `timm.models.registry` module aliases so upstream's
    `from timm.models.layers import trunc_normal_, DropPath` and
    `from timm.models.registry import register_model` resolve, WITHOUT editing upstream.
    No-op on timm 1.0.26/1.0.27 (both still ship these deprecated modules); guards against a
    future timm release removing them.
    """
    try:
        import timm.models.layers  # noqa: F401
    except ImportError:
        import timm.layers as _timm_layers

        sys.modules["timm.models.layers"] = _timm_layers

    try:
        import timm.models.registry  # noqa: F401
    except ImportError:
        import timm.models as _timm_models

        if not hasattr(_timm_models, "register_model"):
            raise ImportError(
                "timm.models has no register_model attribute; cannot build the deprecated "
                "timm.models.registry shim needed by the pinned upstream surgenet code. "
                "timm API has drifted further than this adapter anticipates -- update the shim."
            )
        _shim = types.ModuleType("timm.models.registry")
        _shim.register_model = _timm_models.register_model
        sys.modules["timm.models.registry"] = _shim


_ensure_deprecated_timm_shims()

# Upstream module import. `metaformer.py` is unmodified; only the path/shims above make this
# import resolve.
import metaformer as _upstream_metaformer  # type: ignore  # noqa: E402

caformer_s18 = _upstream_metaformer.caformer_s18
MetaFormerFPN = _upstream_metaformer.MetaFormerFPN

__all__ = [
    "caformer_s18",
    "MetaFormerFPN",
    "SurgeNetLoadError",
    "load_surgenet_into_model",
    "run_golden_activation_test",
]


# ---------------------------------------------------------------------------
# Hardened local-checkpoint loader
# ---------------------------------------------------------------------------

_EXPECTED_KEY_COUNT = 152
_REQUIRED_KEY = "downsample_layers.0.conv.weight"
_FORBIDDEN_PREFIXES = ("teacher.", "student.", "backbone.", "module.", "head.")
# StarReLU (wrong/default config) affine params: SepConv's act1 (stages 0-1) and Mlp's act
# (all stages) both use `s * relu(x)**2 + b` under the default config, with learnable
# `.scale`/`.bias`. The correct 'SurgeNet' (ReLU) config has no such params at all, so if the
# encoder was built with the wrong config these keys exist on the model but are absent from
# the checkpoint -> they show up in `missing_keys`.
_ACTIVATION_AFFINE_MARKERS = (".act.scale", ".act.bias", ".act1.scale", ".act1.bias")


class SurgeNetLoadError(ValueError):
    """Raised when a SurgeNet checkpoint fails one of the hardened-loader checks."""


def _get_encoder(model: nn.Module) -> nn.Module:
    return model.metaformer if hasattr(model, "metaformer") else model


def load_surgenet_into_model(model: nn.Module, ckpt_path: str | Path) -> dict:
    """Load the SurgeNetXL teacher checkpoint into `model`'s CaFormer encoder.

    `model` may be a `MetaFormerFPN` (encoder = `model.metaformer`) or a bare `caformer_s18`
    instance. Local file only -- never fetches a URL. `model` MUST already have been
    constructed with `pretrained='SurgeNet'` (the ReLU config); this function verifies that
    and raises `SurgeNetLoadError` / `ValueError` loudly otherwise. Implements checks 2-4 of
    the load checks (1, 5, 6 are the golden-activation test / forward-shape assert / CLI-flag
    assert, done by the caller -- see `run_golden_activation_test` and
    `scripts/train_t3_caformerxl.py`).
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SurgeNet checkpoint not found: {ckpt_path}")

    encoder = _get_encoder(model)

    # --- 2. Right file / right convention -----------------------------------
    src_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(src_sd, dict) or not all(torch.is_tensor(v) for v in src_sd.values()):
        raise SurgeNetLoadError(
            f"{ckpt_path} is not a flat tensor state_dict (got {type(src_sd)!r}). "
            "Use the 89 MB *_teacher.pth checkpoint, not the 733 MB full train-state checkpoint."
        )
    if len(src_sd) != _EXPECTED_KEY_COUNT:
        raise SurgeNetLoadError(
            f"{ckpt_path}: expected exactly {_EXPECTED_KEY_COUNT} keys (flat CaFormer-S18 "
            f"backbone state_dict), got {len(src_sd)}. Wrong checkpoint file?"
        )
    if _REQUIRED_KEY not in src_sd:
        raise SurgeNetLoadError(f"{ckpt_path}: missing required key {_REQUIRED_KEY!r}.")
    bad_prefixed = sorted(k for k in src_sd if k.startswith(_FORBIDDEN_PREFIXES))
    if bad_prefixed:
        raise SurgeNetLoadError(
            f"{ckpt_path}: found key(s) with forbidden prefix {_FORBIDDEN_PREFIXES} "
            f"(e.g. {bad_prefixed[:5]}) -- looks like a DINO teacher/student wrapper checkpoint, "
            "not the plain backbone state_dict this loader expects."
        )

    # --- fingerprint: snapshot every encoder param/buffer before loading ----
    init_snapshot = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    model_keys = set(init_snapshot.keys())

    # --- reject any source key the encoder doesn't recognize ----------------
    unrecognized_src_keys = sorted(set(src_sd.keys()) - model_keys)
    if unrecognized_src_keys:
        raise ValueError(
            f"{ckpt_path}: {len(unrecognized_src_keys)} unrecognized source key(s) not present "
            f"on the encoder, e.g. {unrecognized_src_keys[:5]}. Refusing to load -- this usually "
            "means the wrong model class/config was built, or upstream metaformer.py has "
            "drifted from the pinned commit."
        )

    result = encoder.load_state_dict(src_sd, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    # --- 4. Missing-keys audit ------------------------------------------------
    # This block (not run_golden_activation_test) is the AUTHORITATIVE StarReLU guard: it is
    # the one check that runs on every real load (train and inference), independent of any
    # golden-test fixture, and it fails loudly the instant the encoder was built with the wrong
    # (StarReLU/'ImageNet') config, because that config's `.act(1).scale`/`.bias` params can
    # never be satisfied by this checkpoint and so always land in `missing_keys`.
    if unexpected:
        raise SurgeNetLoadError(f"{ckpt_path}: unexpected_keys should be empty, got {unexpected}")
    activation_missing = [k for k in missing if any(m in k for m in _ACTIVATION_AFFINE_MARKERS)]
    if activation_missing:
        raise SurgeNetLoadError(
            f"{ckpt_path}: activation-affine key(s) found in missing_keys: {activation_missing}. "
            "This means the encoder was built with StarReLU (the wrong/default config), not the "
            "ReLU SurgeNet config. Construct with pretrained='SurgeNet'."
        )
    non_head_missing = [k for k in missing if not k.startswith("head.")]
    if non_head_missing:
        raise SurgeNetLoadError(
            f"{ckpt_path}: missing_keys must be head-only, got non-head missing key(s) "
            f"{non_head_missing}."
        )

    # --- 3. Loaded-vs-init fingerprint: every consumed param actually changed, and
    # every source tensor was consumed exactly once (mapped + intentionally-dropped == count) --
    consumed = set(src_sd.keys())
    dropped_intentionally = model_keys - consumed  # should equal the head-only `missing` set
    if dropped_intentionally != set(missing):
        raise SurgeNetLoadError(
            "Internal consistency check failed: (encoder keys - checkpoint keys) != missing_keys. "
            f"diff={dropped_intentionally.symmetric_difference(set(missing))}"
        )
    if len(consumed) + len(dropped_intentionally) != len(model_keys):
        raise SurgeNetLoadError(
            "Internal consistency check failed: mapped + intentionally-dropped != total encoder "
            f"keys ({len(consumed)} + {len(dropped_intentionally)} != {len(model_keys)})."
        )
    still_at_init = [
        k for k in consumed if torch.equal(encoder.state_dict()[k], init_snapshot[k])
    ]
    if still_at_init:
        raise SurgeNetLoadError(
            f"{ckpt_path}: {len(still_at_init)} loaded param(s) are bit-identical to their random "
            f"init (no-op load?), e.g. {still_at_init[:5]}."
        )

    # --- 5. Full-res forward-shape assertion --------------------------------
    # Guards the FPN stride-4 hazard in code, not just empirically: only applies when `model`
    # is the full MetaFormerFPN (encoder + FPN decoder), not a bare caformer_s18 encoder, since
    # only the former produces a per-pixel segmentation map.
    if hasattr(model, "FPN"):
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                probe_out = model(torch.zeros(1, 3, 448, 800))
        finally:
            model.train(was_training)
        expected_classes = getattr(encoder, "num_classes", 1)
        expected_shape = (1, expected_classes, 448, 800)
        if tuple(probe_out.shape) != expected_shape:
            raise SurgeNetLoadError(
                f"{ckpt_path}: post-load forward-shape check failed -- model(1,3,448,800) "
                f"returned shape {tuple(probe_out.shape)}, expected {expected_shape}. This guards "
                "the FPN stride-4 hazard, in case `interpolation` is ever touched."
            )

    return {
        "num_loaded": len(consumed),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


_STARRELU_DISCRIMINATION_MIN_MAX_DIFF = 1e-2  # reviewer-measured per-stage max|diff| ~= [0.13, 7.06, 103.66, 0.96]


def run_golden_activation_test(
    ckpt_path: str | Path,
    num_classes: int = 1,
    image_size: tuple[int, int] = (224, 224),
    seed: int = 0,
) -> list[float]:
    """The activation-identity 'killer' check. Three independent sub-checks:

    (A) Loader-wiring check (ReLU vs ReLU cross-build). Builds a from-scratch reference
    `caformer_s18(pretrained='SurgeNet')`, loads the checkpoint directly into it (a plain
    `strict=False` load, independent of `load_surgenet_into_model`'s extra bookkeeping),
    separately builds a `MetaFormerFPN(pretrained='SurgeNet')` and loads the SAME checkpoint into
    its `.metaformer` encoder via `load_surgenet_into_model`, then runs one fixed random input
    through both and asserts per-stage max|diff| < 1e-4. Both builds use the identical (ReLU)
    activation config, so this by itself CANNOT discriminate ReLU from StarReLU -- what it proves
    is that `load_surgenet_into_model`'s key mapping, stage order, and tensor-layout handling
    (no transpose error, no stage off-by-one) reproduce a plain direct `state_dict` load exactly.

    (B) StarReLU-discrimination check (NEW -- this is what actually earns the "proves
    ReLU-not-StarReLU" claim). Builds a THIRD model, `caformer_s18(pretrained='ImageNet')` (the
    default/wrong StarReLU config), loads the SAME checkpoint into it with `strict=False`, and
    asserts its features on the same fixed input differ from the (A) reference by max|diff| >
    1e-2 on at least one stage. Since the checkpoint carries no StarReLU affine params at all,
    this load leaves the StarReLU build's `.scale`/`.bias` at random init while the shared conv/
    attention weights are identical to the ReLU build -- so a large measured difference here
    proves this checkpoint actually produces materially different activations under ReLU vs
    StarReLU, i.e. that the test fixture can discriminate the two configs. (Reviewer-measured
    per-stage max|diff| ~= [0.13, 7.06, 103.66, 0.96].)

    (C) Zero-StarReLU-params check (NEW). Asserts the ReLU-config (A) reference has NO parameter
    whose name matches a StarReLU affine marker (`*.act.scale`, `*.act1.scale`, etc.) -- i.e. the
    'SurgeNet' config is structurally incapable of using StarReLU, independent of what got loaded.

    Raises `AssertionError` if any of (A)/(B)/(C) fails. NOTE: (A) is necessary but NOT
    sufficient to prove ReLU-not-StarReLU on its own -- see (B)/(C). The authoritative,
    always-on StarReLU guard for real loads is `load_surgenet_into_model`'s missing-keys audit
    (its activation-affine-in-missing-keys check), not this golden test; this golden test is a
    fixture-level proof that the loader machinery and the config choice are both correct.
    """
    ckpt_path = Path(ckpt_path)
    src_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    reference = caformer_s18(num_classes=num_classes, pretrained="SurgeNet", pretrained_weights=None)
    ref_result = reference.load_state_dict(src_sd, strict=False)
    if ref_result.unexpected_keys:
        raise SurgeNetLoadError(f"golden test reference: unexpected_keys={ref_result.unexpected_keys}")
    if any(not k.startswith("head.") for k in ref_result.missing_keys):
        raise SurgeNetLoadError(f"golden test reference: non-head missing_keys={ref_result.missing_keys}")

    # --- (C) zero-StarReLU-params check: the ReLU config must have none of these params at all --
    starrelu_params_on_reference = [
        n for n, _ in reference.named_parameters() if any(m in n for m in _ACTIVATION_AFFINE_MARKERS)
    ]
    if starrelu_params_on_reference:
        raise AssertionError(
            "golden test (C) FAILED: the 'SurgeNet' (ReLU) config reference has StarReLU affine "
            f"parameter(s) {starrelu_params_on_reference}; expected none. This means "
            "pretrained='SurgeNet' is not actually selecting the plain-ReLU token mixer/MLP config."
        )

    candidate = MetaFormerFPN(num_classes=num_classes, pretrained="SurgeNet", pretrained_weights=None)
    load_surgenet_into_model(candidate, ckpt_path)

    reference.eval()
    candidate.metaformer.eval()

    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 3, *image_size, generator=gen)

    with torch.no_grad():
        _, ref_feats = reference.forward_features(x)
        _, cand_feats = candidate.metaformer.forward_features(x)

    if len(ref_feats) != len(cand_feats):
        raise AssertionError(f"stage count mismatch: {len(ref_feats)} vs {len(cand_feats)}")

    # --- (A) loader-wiring check: ReLU-vs-ReLU cross-build must match almost exactly ---
    max_diffs = []
    for i, (rf, cf) in enumerate(zip(ref_feats, cand_feats)):
        if rf.shape != cf.shape:
            raise AssertionError(f"stage {i} shape mismatch: {rf.shape} vs {cf.shape}")
        d = (rf - cf).abs().max().item()
        max_diffs.append(d)
        if d >= 1e-4:
            raise AssertionError(
                f"golden activation test (A) FAILED at stage {i}: max|diff|={d} >= 1e-4. This means "
                "the ReLU (SurgeNet) config was not used correctly somewhere in the build/load path."
            )

    # --- (B) StarReLU-discrimination check: the wrong (StarReLU) config must differ materially ---
    wrong_config = caformer_s18(num_classes=num_classes, pretrained="ImageNet", pretrained_weights=None)
    wrong_result = wrong_config.load_state_dict(src_sd, strict=False)
    if wrong_result.unexpected_keys:
        raise SurgeNetLoadError(
            f"golden test (B) StarReLU build: unexpected_keys={wrong_result.unexpected_keys}"
        )
    wrong_config.eval()
    with torch.no_grad():
        _, wrong_feats = wrong_config.forward_features(x)
    if len(wrong_feats) != len(ref_feats):
        raise AssertionError(f"stage count mismatch (B): {len(ref_feats)} vs {len(wrong_feats)}")

    starrelu_diffs = []
    for i, (rf, wf) in enumerate(zip(ref_feats, wrong_feats)):
        if rf.shape != wf.shape:
            raise AssertionError(f"stage {i} shape mismatch (B): {rf.shape} vs {wf.shape}")
        starrelu_diffs.append((rf - wf).abs().max().item())
    if max(starrelu_diffs) <= _STARRELU_DISCRIMINATION_MIN_MAX_DIFF:
        raise AssertionError(
            f"golden test (B) FAILED: ReLU-config reference and StarReLU-config (wrong) build "
            f"produced near-identical features (per-stage max|diff|={starrelu_diffs}, max over "
            f"stages <= {_STARRELU_DISCRIMINATION_MIN_MAX_DIFF}). This means the fixture cannot "
            "actually distinguish ReLU from StarReLU, so (A) passing would not prove anything."
        )

    return max_diffs
