# x86_64 dGPU image for the safety monitoring pipeline.
#
#   docker compose -f docker/compose.yml up --build
#
# The Jetson build installs onto the host with scripts/setup.sh. This image does the same work in
# layers, with three things deliberately NOT baked in:
#
#   engines  TensorRT engines are specific to the GPU architecture, the driver and the TensorRT
#            version that built them. One built here would be wrong everywhere else, so
#            docker/entrypoint.sh builds them on first boot against the GPU actually present.
#   media    454 MB of site footage. Mounted, never carried between deployments.
#   weights  The VLM/LLM GGUFs are ~12 GB and optional. Fetched to a volume when enabled.
#
# ONNX *is* baked, in a separate stage below: it is architecture-independent, and exporting it
# needs torch + ultralytics, ~2.5 GB that has no business in a runtime image.

# =================================================================================================
# Stage 1 — detector weights -> ONNX
# =================================================================================================
# Deliberately CPU torch, for the same reason scripts/setup.sh gives: this stage only ever traces
# a model to ONNX. TensorRT does every actual inference, so a CUDA torch build would be a
# multi-gigabyte download for zero benefit — and here it would be discarded at the stage boundary
# anyway.
FROM python:3.12-slim AS onnx-export

WORKDIR /src
# ultralytics imports opencv, which links libGL and libglib even for a pure export that never
# opens an image. Without them the stage dies on `import ultralytics` with a bare
# "libGL.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision
COPY requirements/export.txt requirements/export.txt
RUN pip install --no-cache-dir -r requirements/export.txt

# export_models.py resolves paths from its own location (ROOT = parent.parent), so the script has
# to sit at scripts/ for models/ to land where the nvinfer configs expect it.
COPY scripts/export_models.py scripts/export_models.py
RUN python3 scripts/export_models.py all

# Leaves: models/{ppe,fire}/model/*.onnx, models/{ppe,fire}/config/labels.txt,
#         models/export_summary.json  — the class-index contract the parser and configs share.


# =================================================================================================
# Stage 2 — runtime
# =================================================================================================
# `samples` rather than `triton`: project_skill.md §4 settles nvinfer over nvinferserver, so the
# Triton server is dead weight. Verified on this image: the pyservicemaker wheel, trtexec, g++ and
# the DeepStream headers the custom parser needs are all present.
FROM nvcr.io/nvidia/deepstream:9.1-samples-multiarch

ENV DEBIAN_FRONTEND=noninteractive \
    ISMS_ROOT=/opt/isms \
    PYTHONUNBUFFERED=1

# ffmpeg        media handling and the file-mode clip cut (-c copy)
# libmosquitto1 libnvds_nvmultiobjecttracker.so dlopens it. Without it the tracker fails to load
#               with "Failed to initilaize low level lib" — a message that names the tracker and
#               not the missing library.
# redis-tools   redis-cli only. The server runs as its own container; nothing here shells out to
#               manage it, but several scripts gate on `redis-cli ping`.
# python3-venv  the services venv below
# bsdextrautils  supplies `column`, which build_engines.sh ends by pretty-printing its CSV with.
#                Absent, that last line returns 127 and a perfectly good engine build reports
#                failure. (The entrypoint checks the engine files rather than the exit code for
#                the same reason, but the table is worth having in the log.)
# libpython3.12   the SHARED python library. The samples image ships the interpreter but not
#                 libpython3.12.so.1.0, and pyservicemaker's _pydeepstream extension links
#                 against it — so the wheel installs cleanly, `pip show` reports it present, and
#                 the first `import pyservicemaker` fails on a missing .so.
# cuda-cudart-dev / cuda-crt   headers only, for models/parser. The samples image ships the CUDA
#   runtime but not its headers: the parser includes NvInfer.h -> cuda_runtime_api.h (cudart-dev)
#   -> crt/host_defines.h (crt). It links -lnvinfer alone, so both are compile-time only.
#
#   The 13-2 suffix tracks the CUDA the base image carries. A base image on a different CUDA fails
#   here with "Unable to locate package", at build time and by name, which is the right place to
#   find out.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg mediainfo bc curl git make cmake build-essential \
        python3-venv python3-yaml redis-tools libmosquitto1 \
        cuda-cudart-dev-13-2 cuda-crt-13-2 libpython3.12 bsdextrautils \
    && rm -rf /var/lib/apt/lists/*

# The base image ships a dpkg database that claims libvpx9, libx264, libx265, libmp3lame and
# friends are installed, while the image build deleted the actual .so files to save space. apt
# therefore considers them satisfied and skips them, and the first thing to notice is ffprobe
# dying with "libvpx.so.9: cannot open shared object file" — from a package dpkg reports as `ii`.
#
# DeepStream ships this script for exactly that: it is a list of `apt-get install --reinstall`
# for the stripped packages, maintained by NVIDIA alongside the image. Running theirs rather than
# reconstructing the list here means a base-image change that strips something new is handled.
#
# Without it the pipeline still runs — DeepStream decodes through its own elements — and every
# evidence clip fails, because clip_service.py cuts them with ffmpeg.
RUN bash /opt/nvidia/deepstream/deepstream/user_additional_install.sh \
    && rm -rf /var/lib/apt/lists/* \
    && ffprobe -version > /dev/null

# pyservicemaker ships in the image but is NOT pre-installed. It goes into system python because
# that is where app/safety_pipeline.py imports it from — the same split the Jetson build has, and
# the reason app/events.py speaks RESP over a plain socket rather than importing a redis client.
RUN pip install --no-cache-dir --break-system-packages \
        /opt/nvidia/deepstream/deepstream/service-maker/python/pyservicemaker*.whl \
        pyyaml

WORKDIR ${ISMS_ROOT}

# Subject crops (`--rgb-capture`). The pipeline pulls the violating subject out of the live frame
# via Buffer.extract() -> DLPack -> torch, so torch has to be CUDA-enabled and in SYSTEM python,
# beside pyservicemaker. ~3.5 GB, hence the switch: --build-arg WITH_RGB_CAPTURE=0 leaves it out
# and the VLM falls back to context frames.
#
# Only RTSP mode needs it. In file mode the crop is cut from the source .mp4 at the incident PTS
# and no torch is involved.
ARG WITH_RGB_CAPTURE=1
RUN if [ "$WITH_RGB_CAPTURE" = "1" ]; then \
        pip install --no-cache-dir --break-system-packages \
            torch --index-url https://download.pytorch.org/whl/cu126 \
        && pip install --no-cache-dir --break-system-packages opencv-python-headless \
        && python3 -c "import torch, cv2; from torch.utils.dlpack import from_dlpack" ; \
    fi

# Services venv, kept out of system python for the reason above. Copied first so a source edit
# does not invalidate the dependency layer.
COPY requirements/ requirements/
RUN python3 -m venv build/venv-services \
    && build/venv-services/bin/pip install --no-cache-dir --upgrade pip \
    && build/venv-services/bin/pip install --no-cache-dir -r requirements/services.txt

# mediamtx republishes the tiled output as RTSP/WebRTC/HLS. It lives in the image rather than in
# its own container because services/api.py drives it by shelling out to scripts/serve_rtsp.sh —
# a separate container would break the dashboard's live-view toggle.
ARG MEDIAMTX_VER=v1.20.0
RUN mkdir -p build/bin \
    && curl -fsSL "https://github.com/bluenviron/mediamtx/releases/download/${MEDIAMTX_VER}/mediamtx_${MEDIAMTX_VER}_linux_amd64.tar.gz" \
       | tar xz -C build/bin mediamtx \
    && build/bin/mediamtx --version

# Source. Last, so iterating on it rebuilds only what follows.
COPY app/ app/
COPY services/ services/
COPY scripts/ scripts/
COPY configs/ configs/
COPY dashboard/ dashboard/
COPY tools/ tools/
COPY tests/ tests/
COPY models/ models/
COPY .env.example README.md ./

# The ONNX and the label files from stage 1. labels.txt is the class-index contract that both the
# parser and the nvinfer configs depend on, so it travels with the graph it was read off.
COPY --from=onnx-export /src/models/ models/

# The custom YOLO output parser. nvinfer dlopens this .so to turn each detector's raw tensor into
# boxes; both nvinfer configs name it by path, so the pipeline cannot start without it.
RUN make -C models/parser \
    && test -f models/parser/libnvds_yolo_parser.so

# models/ is a volume at runtime, so that a ~10 minute engine build survives a container replace.
# A volume shadows the image's own copy of the directory: Docker seeds it from the image the first
# time and never again, so a rebuilt parser or a re-exported ONNX would silently never reach a
# deployment that already had the volume. The image therefore keeps a pristine copy here, and the
# entrypoint refreshes everything EXCEPT the engines out of it on every start — image content
# tracks the image, engines stay where the build left them.
RUN cp -a models /opt/isms/models.dist

# Deliberately no .env. scripts/env.sh's load_env() sources it and would overwrite the operator's
# credentials with this file's empty ones — an image that ships `TELEGRAM_BOT_TOKEN=` clobbers a
# real token injected through the environment. Absent, load_env leaves what compose set alone.
RUN mkdir -p data logs media build

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# 8080 dashboard + API · 8554 RTSP · 8888 HLS · 8889 WebRTC · 9997 mediamtx control API
EXPOSE 8080 8554 8888 8889 9997

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["serve"]
