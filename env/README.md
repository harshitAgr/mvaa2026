# Environments and dependencies

Three environments are involved. They are deliberately separate — the training environments track
current CUDA, while the inference container pins an older CUDA line for the evaluation hardware.

| Environment | Used for | Pin file | Python |
|---|---|---|---|
| **nnU-Net** | Task 1 + Task 2 training | [`requirements-nnunet.txt`](requirements-nnunet.txt) | 3.12 |
| **PyTorch / T3** | Task 3 training | [`requirements-t3.txt`](requirements-t3.txt) | 3.12 |
| **Container** | inference (all 3 tasks) | [`../docker/requirements.txt`](../docker/requirements.txt) | 3.12 (`python:3.12-slim`) |

```bash
python -m venv .venv-nnunet && .venv-nnunet/bin/pip install -r env/requirements-nnunet.txt
python -m venv .venv        && .venv/bin/pip install -r env/requirements-t3.txt
```

Point the Task 1/2 scripts at the first with `NNUNET_VENV=/path/to/.venv-nnunet`; the Task 3
driver uses `.venv` by default, overridable with `PYTHON=/path/to/python`.

## Key versions

| Package | nnU-Net env (T1/T2) | T3 env | Container | Note |
|---|---|---|---|---|
| `torch` | 2.11.0+**cu128** | 2.11.0+**cu128** | 2.5.1+**cu124** | The evaluation server is an NVIDIA V100 (**sm_70**). The cu128 wheels dropped sm_70, so the image must be built on the cu124 line; the Dockerfile asserts this at build time. |
| `torchvision` | 0.26.0+cu128 | 0.26.0+cu128 | 0.20.1+cu124 | follows `torch` |
| `timm` | **1.0.22** | **1.0.27** | 1.0.26 | the three differ — see the conflict below |
| `nnunetv2` | 2.7.0 | — | 2.7.0 | |
| `segmentation-models-pytorch` | — | 0.5.0 | 0.5.0 | |
| `monai` | — | 1.6.0 | 1.5.2 | |

The two training environments genuinely pin different `timm` versions; that is the point of keeping
them separate, not an inconsistency. Read each column against its own requirements file.

The pin files are captured from the machine that produced the submitted models, so a few packages
have moved on since the paper was written (the paper cites MONAI 1.5; the T3 environment now
resolves 1.6.0). Nothing in the deployed inference path depends on MONAI — the container pins
1.5.2 — so this affects training-side utilities only.

## Two things that will bite you

**1. `timm` conflicts with nnU-Net.** Task 3 needs `timm >= 1.0.26` for its DINOv2 graph, but
nnU-Net's `dynamic-network-architectures==0.4.3` pins `timm < 1.0.23`. Installing both in one
environment leaves pip unsatisfiable. The container resolves it by installing everything else
first and then forcing timm last:

```dockerfile
RUN pip install --no-deps timm==1.0.26
```

pip warns rather than errors, and this is the exact stack that produced the scored output. If you
train Tasks 1/2 and Task 3 in separate environments — as above — you never hit this.

**2. The container must build offline.** The evaluation runs with no network, so anything normally
fetched at runtime has to exist in the image. The Dockerfile pre-caches the ImageNet encoder
weights for `efficientnet-b4` and `mit_b3` at build time so `smp` can construct those
architectures offline. Those caches are overwritten by the trained `state_dict` at load and
contribute nothing to the output; they exist only so model construction does not reach for the
network. The deployed Task 3 path does not use either encoder.

## Hardware

Training was done on a single NVIDIA RTX PRO 6000. An nnU-Net fold peaks around 11.8 GB, so
several fit concurrently. The evaluation target is a 12 GB V100 with a budget of ~10 s per case.
