# Builds the custom YOLO bbox parsers used by both nvinfer instances.
#   make            -> libnvds_yolo_parser.so
#
# CUDA_VER is auto-detected from the installed toolkit so this keeps working across JetPack
# bumps; override it explicitly if more than one toolkit is installed.

CUDA_VER ?= $(shell ls -d /usr/local/cuda-*/ 2>/dev/null | sed 's#.*cuda-##; s#/##' | sort -V | tail -1)
DS_ROOT  ?= /opt/nvidia/deepstream/deepstream

CC       := g++
CFLAGS   := -Wall -Wextra -std=c++14 -O2 -shared -fPIC
CFLAGS   += -I$(DS_ROOT)/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include
LFLAGS   := -Wl,--start-group -lnvinfer -Wl,--end-group

TARGET   := libnvds_yolo_parser.so
SRC      := nvdsinfer_custom_yolo.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	@echo "CUDA_VER=$(CUDA_VER)  DS_ROOT=$(DS_ROOT)"
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)
	@echo "built $@"
	@nm -D --defined-only $@ | grep NvDsInferParseYolo || \
	  (echo "ERROR: parser symbols missing from $@"; exit 1)

clean:
	rm -f $(TARGET)

.PHONY: all clean
