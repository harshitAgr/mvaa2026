# MVAA 2026 — unified test-phase inference image (all 3 tasks).
# Base: slim Python; the cu124 torch wheel bundles CUDA libs, so only the host NVIDIA
# driver (via nvidia-container-toolkit) is needed at runtime. The official eval server
# is a V100 (sm_70, CUDA 12.4); cu124 wheels include sm_70 (the cu128 line DROPPED it),
# so we target cu124 here. NOTE: this image therefore CANNOT be GPU-tested on our Blackwell
# (sm_120) dev box — validate CPU-only first, then confirm on an sm_70 runner.
FROM python:3.12-slim

# MVAA_T2_DISABLE_TTA=1 is the image default: Task 2 is excluded from the final-test ranking
# (every eligible team receives the same Task 2 score, because the underlying data is public),
# so its 8x mirror TTA is wasted compute. Dropping it makes the slowest task ~30% faster with
# no ranking impact. Override with `-e MVAA_T2_DISABLE_TTA=0` to reproduce the TTA-on masks
# (e.g. for the raw metrics reported in the paper).
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    nnUNet_results=/opt/weights/nnunet \
    nnUNet_raw=/tmp/nnraw \
    nnUNet_preprocessed=/tmp/nnprep \
    MVAA_T3_CKPT_A=/opt/weights/t3/imagenet_dr_epoch070.pt \
    MVAA_T3_CKPT_B=/opt/weights/t3/surgenetxl_best.pt \
    MVAA_T3_THRESHOLD=0.45 \
    nnUNet_compile=0 \
    MVAA_T2_DISABLE_TTA=1 \
    TORCH_HOME=/opt/cache/torch \
    HF_HOME=/opt/cache/hf

# libgomp1 for scipy/scikit-image OpenMP; nothing else needed (no cv2).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# torch first, pinned to the cu124 build for V100 / sm_70 compatibility (the eval server).
# 2.5.1 satisfies every pin in requirements.txt (tightest lower bound: monai 1.5.2 -> torch>=2.4.1;
# nnunetv2 2.7.0 -> torch>=2.1.2,!=2.9.*). We compile no custom CUDA kernels, so the prebuilt
# cu124 wheels' bundled sm_70 SASS is all we need — no TORCH_CUDA_ARCH_LIST.
RUN pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124
# Sanity gate: confirm we got a CUDA 12.4 (cu124) build, not a CPU wheel. get_arch_list() can't be
# checked here — it returns [] with no GPU in the build sandbox — so sm_70 coverage is verified
# post-build via `docker run --gpus all ... get_arch_list()`. cu124 wheels include sm_70 (cu128
# dropped it); torch.version.cuda is static and needs no GPU.
RUN python -c "import torch; cu=torch.version.cuda; print('torch', torch.__version__, '| cuda', cu); assert cu and cu.startswith('12.4'), f'expected a cu124 CUDA build, got cuda={cu} (CPU wheel?)'"

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# T2 uses the custom nnUNetTrainer_SafeMirror (mirror_axes=(0,2), excludes the
# anatomically-unsafe A-P axis for the mitral leaflet labels). nnUNetv2_predict
# resolves trainer classes via recursive_find_python_class() rooted at
# nnunetv2/training/nnUNetTrainer/, so the file must live inside that tree —
# mirrors what scripts/nnunet_t2/install_trainers.sh does for the host venvs.
COPY code/nnunet_trainers /tmp/nnunet_trainers
RUN NNUNET_DIR=$(python -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))") && \
    mkdir -p "$NNUNET_DIR/training/nnUNetTrainer/variants/data_augmentation" && \
    cp /tmp/nnunet_trainers/nnUNetTrainer_SafeMirror.py "$NNUNET_DIR/training/nnUNetTrainer/variants/data_augmentation/" && \
    python -c "from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainer_SafeMirror import nnUNetTrainer_SafeMirror; print('nnUNetTrainer_SafeMirror import OK')"

# T3 needs timm 1.0.26, but nnU-Net's dynamic-network-architectures 0.4.3 pins timm<1.0.23.
# The host venv runs both together; reproduce that by upgrading timm post-hoc (pip warns, not errors).
# This is the exact stack that produced our scored T3 output.
RUN pip install --no-deps timm==1.0.26

# Pre-cache the ImageNet EfficientNet-B4 encoder weights into the image so the T3
# ImageNet model builds OFFLINE at runtime (the challenge test container has no network).
# The trained state_dict overwrites these, but building the arch requires the cache to exist.
RUN python -c "import segmentation_models_pytorch as smp; smp.UnetPlusPlus(encoder_name='efficientnet-b4', encoder_weights='imagenet', in_channels=3, classes=1)" \
    && echo "pre-cached efficientnet-b4 imagenet weights to $TORCH_HOME"
# Same for the SegFormer MiT-B3 encoder (T3 3-way arch-diversity member) — its trained state_dict
# overwrites these, but smp.Segformer(encoder_weights='imagenet') needs the cache to build offline.
RUN python -c "import segmentation_models_pytorch as smp; smp.Segformer(encoder_name='mit_b3', encoder_weights='imagenet', in_channels=3, classes=1)" \
    && echo "pre-cached mit_b3 imagenet weights to $TORCH_HOME"

# Install the optional T1 all-data trainer after dependency/cache layers. A standard build still
# selects fold0; candidate builds choose this trainer/fold through immutable build args.
COPY code/t1_trainers/nnUNetTrainer_T1AllData250.py /tmp/t1_trainers/nnUNetTrainer_T1AllData250.py
RUN NNUNET_DIR=$(python -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))") && \
    mkdir -p "$NNUNET_DIR/training/nnUNetTrainer/variants/sampling" && \
    cp /tmp/t1_trainers/nnUNetTrainer_T1AllData250.py "$NNUNET_DIR/training/nnUNetTrainer/variants/sampling/" && \
    python -c "from nnunetv2.training.nnUNetTrainer.variants.sampling.nnUNetTrainer_T1AllData250 import nnUNetTrainer_T1AllData250; print('nnUNetTrainer_T1AllData250 import OK')"

# Configuration selectors. A plain build selects the Dataset512 self-train 5-fold T1 ensemble
# plus the two-member T3 fusion at weight 0.60. Override at build time with --build-arg, or at
# run time with -e. See the repository README for the exact args used by the deployed image.
ARG MVAA_T1_TRAINER=nnUNetTrainer_250epochs
ARG MVAA_T1_FOLD="0 1 2 3 4"
ARG MVAA_T1_FUSION_LARGEPATCH=0
ARG MVAA_T1_LP_FOLD="0 1 2 3 4"
ARG MVAA_T3_WEIGHT_A=0.6
ARG MVAA_T3_WEIGHT_B=0.3
ARG MVAA_T3_WEIGHT_C=0.1
ARG MVAA_T3_CKPT_A_PATH=/opt/weights/t3/imagenet_dr_epoch070.pt
ARG MVAA_T3_CKPT_C_PATH=""
ARG MVAA_T3_PRETRAINED_WEIGHTS_PATH=""
ARG MVAA_T3_PRETRAINED_FP32_WEIGHTS_PATH=""
ARG MVAA_T3_SURGENETDINO_CV2=0
ARG MVAA_T3_SURGENETDINO_TRI=0
ARG MVAA_T3_SURGENETDINO_PURE_TRI=0
ENV MVAA_T1_TRAINER=${MVAA_T1_TRAINER} \
    MVAA_T1_FOLD=${MVAA_T1_FOLD} \
    MVAA_T1_FUSION_LARGEPATCH=${MVAA_T1_FUSION_LARGEPATCH} \
    MVAA_T1_LP_FOLD=${MVAA_T1_LP_FOLD} \
    MVAA_T3_WEIGHT_A=${MVAA_T3_WEIGHT_A} \
    MVAA_T3_WEIGHT_B=${MVAA_T3_WEIGHT_B} \
    MVAA_T3_WEIGHT_C=${MVAA_T3_WEIGHT_C} \
    MVAA_T3_CKPT_A=${MVAA_T3_CKPT_A_PATH} \
    MVAA_T3_CKPT_C=${MVAA_T3_CKPT_C_PATH} \
    MVAA_T3_SURGENETDINO_CV2=${MVAA_T3_SURGENETDINO_CV2} \
    MVAA_T3_SURGENETDINO_TRI=${MVAA_T3_SURGENETDINO_TRI} \
    MVAA_T3_SURGENETDINO_PURE_TRI=${MVAA_T3_SURGENETDINO_PURE_TRI}
ENV MVAA_T3_PRETRAINED_WEIGHTS=${MVAA_T3_PRETRAINED_WEIGHTS_PATH}
ENV MVAA_T3_PRETRAINED_FP32_WEIGHTS=${MVAA_T3_PRETRAINED_FP32_WEIGHTS_PATH}
# Keep this selector in the final configuration layer so infrastructure retries reuse the
# already-proven CUDA/PyTorch dependency layers. CPU diagnostics override it with -e MVAA_DEVICE=cpu.
ENV MVAA_DEVICE=cuda

# Application code (scripts/ + baseline/task3/ for the T3 ensemble) and weights.
COPY code /opt/mvaa
COPY weights /opt/weights
COPY run_inference.py /opt/run_inference.py
COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

WORKDIR /opt/mvaa
# Defaults: read /input, write /output, run all tasks. Override args at `docker run`.
ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["--input", "/input", "--output", "/output", "--tasks", "1,2,3"]
