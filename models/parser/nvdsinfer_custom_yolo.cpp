/*
 * Custom nvinfer bbox parsers for the two YOLO heads used by this demo.
 *
 * Neither layout ships with DeepStream, and the two models do NOT share a head:
 *
 *   ppe   YOLO11n   output [batch, 4+num_classes, anchors]  channel-major, PRE-NMS
 *                   -> NvDsInferParseYoloPreNMS   with cluster-mode=2 (DeepStream runs NMS)
 *
 *   fire  YOLO26s   output [batch, max_det, 6]              row-major,     POST-NMS
 *                   -> NvDsInferParseYoloPostNMS  with cluster-mode=4 (no clustering)
 *
 * Getting this pairing wrong is the classic way a DeepStream YOLO integration produces zero
 * boxes or a screen full of garbage, so both parsers log their observed dims once on the first
 * frame — if the shape doesn't match what the config assumes, it says so instead of failing
 * silently.
 *
 * Coordinates are emitted in NETWORK INPUT space (0..networkInfo.width/height). nvinfer applies
 * the transform to frame space itself, accounting for maintain-aspect-ratio and padding.
 */

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseYoloPreNMS(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

extern "C" bool NvDsInferParseYoloPostNMS(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

namespace {

inline float clampf(float v, float lo, float hi)
{
  return v < lo ? lo : (v > hi ? hi : v);
}

/* Per-class threshold, falling back to the global one when the config omits per-class entries. */
inline float thresholdFor(NvDsInferParseDetectionParams const& p, unsigned int classId)
{
  if (classId < p.perClassPreclusterThreshold.size())
    return p.perClassPreclusterThreshold[classId];
  return 0.25f;
}

/* Emit one detection, clamped to the network input rectangle. Boxes that clamp away to nothing
 * are dropped rather than passed on as degenerate zero-area rectangles. */
inline void emit(std::vector<NvDsInferObjectDetectionInfo>& out,
                 unsigned int classId, float conf,
                 float left, float top, float width, float height,
                 NvDsInferNetworkInfo const& net)
{
  const float netW = static_cast<float>(net.width);
  const float netH = static_cast<float>(net.height);

  float l = clampf(left, 0.0f, netW);
  float t = clampf(top, 0.0f, netH);
  float r = clampf(left + width, 0.0f, netW);
  float b = clampf(top + height, 0.0f, netH);

  if (r - l < 1.0f || b - t < 1.0f)
    return;

  /* Zero-initialise: leaving rotation_angle uninitialised renders every box tilted. */
  NvDsInferObjectDetectionInfo obj{};
  obj.classId = classId;
  obj.detectionConfidence = conf;
  obj.left = l;
  obj.top = t;
  obj.width = r - l;
  obj.height = b - t;
  out.push_back(obj);
}

}  // namespace

/* ------------------------------------------------------------------------------------------
 * PRE-NMS  [4 + num_classes, anchors], channel-major.
 *
 * value(c, a) = buf[c * anchors + a]
 *   c = 0..3          -> cx, cy, w, h   (network input pixels)
 *   c = 4..4+nc-1     -> per-class score, already sigmoid-activated
 *
 * There is no separate objectness channel in the v8/v11 head — the class score IS the
 * confidence. Only the best-scoring class per anchor is emitted; DeepStream's NMS
 * (cluster-mode=2) does the rest.
 * ---------------------------------------------------------------------------------------- */
extern "C" bool NvDsInferParseYoloPreNMS(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    fprintf(stderr, "[YoloPreNMS] no output layers\n");
    return false;
  }

  NvDsInferLayerInfo const& layer = outputLayersInfo[0];
  const NvDsInferDims& d = layer.inferDims;
  if (d.numDims < 2) {
    fprintf(stderr, "[YoloPreNMS] expected 2 dims (channels, anchors), got %u\n", d.numDims);
    return false;
  }

  const int channels = static_cast<int>(d.d[0]);
  const int anchors = static_cast<int>(d.d[1]);
  const int numClasses = channels - 4;

  static bool logged = false;
  if (!logged) {
    logged = true;
    printf("[YoloPreNMS] dims={%d, %d} -> %d classes, %d anchors\n",
           channels, anchors, numClasses, anchors);
    if (numClasses <= 0 || anchors <= 0)
      fprintf(stderr, "[YoloPreNMS] IMPLAUSIBLE SHAPE — this model is probably POST-NMS. "
                      "Use NvDsInferParseYoloPostNMS with cluster-mode=4.\n");
    if (numClasses != static_cast<int>(detectionParams.numClassesConfigured))
      fprintf(stderr, "[YoloPreNMS] WARNING: model has %d classes but config declares %u. "
                      "Fix num-detected-classes.\n",
              numClasses, detectionParams.numClassesConfigured);
  }
  if (numClasses <= 0 || anchors <= 0)
    return false;

  const float* buf = static_cast<const float*>(layer.buffer);
  objectList.reserve(64);

  for (int a = 0; a < anchors; ++a) {
    /* Best class for this anchor. */
    int bestClass = -1;
    float bestScore = 0.0f;
    for (int k = 0; k < numClasses; ++k) {
      const float s = buf[(4 + k) * anchors + a];
      if (s > bestScore) { bestScore = s; bestClass = k; }
    }
    if (bestClass < 0 || bestScore < thresholdFor(detectionParams, bestClass))
      continue;

    const float cx = buf[0 * anchors + a];
    const float cy = buf[1 * anchors + a];
    const float w  = buf[2 * anchors + a];
    const float h  = buf[3 * anchors + a];

    emit(objectList, static_cast<unsigned int>(bestClass), bestScore,
         cx - w * 0.5f, cy - h * 0.5f, w, h, networkInfo);
  }

  return true;
}

/* ------------------------------------------------------------------------------------------
 * POST-NMS  [max_det, 6], row-major: (x1, y1, x2, y2, conf, classId).
 *
 * Already clustered by the model, so this is a straight translation into DeepStream's struct
 * and the config must use cluster-mode=4. Rows are score-sorted and zero-padded, so the first
 * row below threshold ends the useful data.
 * ---------------------------------------------------------------------------------------- */
extern "C" bool NvDsInferParseYoloPostNMS(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    fprintf(stderr, "[YoloPostNMS] no output layers\n");
    return false;
  }

  NvDsInferLayerInfo const& layer = outputLayersInfo[0];
  const NvDsInferDims& d = layer.inferDims;
  if (d.numDims < 2) {
    fprintf(stderr, "[YoloPostNMS] expected 2 dims (max_det, 6), got %u\n", d.numDims);
    return false;
  }

  const int maxDet = static_cast<int>(d.d[0]);
  const int stride = static_cast<int>(d.d[1]);

  static bool logged = false;
  if (!logged) {
    logged = true;
    printf("[YoloPostNMS] dims={%d, %d}\n", maxDet, stride);
    if (stride != 6)
      fprintf(stderr, "[YoloPostNMS] expected stride 6 (x1,y1,x2,y2,conf,cls) but got %d — "
                      "this model is probably PRE-NMS. Use NvDsInferParseYoloPreNMS with "
                      "cluster-mode=2.\n", stride);
  }
  if (stride != 6)
    return false;

  const float* buf = static_cast<const float*>(layer.buffer);
  objectList.reserve(32);

  for (int i = 0; i < maxDet; ++i) {
    const float* row = buf + i * stride;
    const float conf = row[4];
    const unsigned int classId = static_cast<unsigned int>(row[5] + 0.5f);

    /* Rows arrive score-sorted; the first sub-threshold row means the rest are padding. */
    if (conf < thresholdFor(detectionParams, classId))
      break;

    emit(objectList, classId, conf,
         row[0], row[1], row[2] - row[0], row[3] - row[1], networkInfo);
  }

  return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloPreNMS);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloPostNMS);
