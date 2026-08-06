// Minimal CUPTI Activity API tracer for ARGUS demos.
// Collects CONCURRENT_KERNEL records: name, streamId, duration_ms.
//
// Build (CUDA devel image)::
//   KERNEL_T=$(grep -oE 'CUpti_ActivityKernel[0-9]+' \
//                "$CUDA_HOME/include/cupti_activity.h" | sort -V | tail -1)
//   g++ -shared -fPIC -O2 -DARGUS_KERNEL_T=$KERNEL_T \
//       -I"$CUDA_HOME/include" playground/argus_cupti_tracer.cpp \
//       -L"$CUDA_HOME/lib64" -lcupti -o libargus_cupti_tracer.so

#include <cupti.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#ifndef ARGUS_KERNEL_T
#define ARGUS_KERNEL_T CUpti_ActivityKernel4
#endif

namespace {

constexpr size_t kNameLen = 128;
constexpr size_t kBufSize = 8 * 1024 * 1024;

struct Record {
  char name[kNameLen];
  uint32_t stream_id;
  double duration_ms;
};

std::mutex g_mu;
std::vector<Record> g_records;
bool g_started = false;

#define ARGUS_CUPTI_CHECK(call)                                                  \
  do {                                                                           \
    CUptiResult _st = (call);                                                    \
    if (_st != CUPTI_SUCCESS) {                                                  \
      const char* _err = nullptr;                                                \
      cuptiGetResultString(_st, &_err);                                          \
      std::fprintf(stderr, "CUPTI error at %s:%d: %s\n", __FILE__, __LINE__,     \
                   _err ? _err : "?");                                           \
      return -1;                                                                 \
    }                                                                            \
  } while (0)

void push_kernel(const char* name, uint32_t stream_id, uint64_t start_ns, uint64_t end_ns) {
  if (end_ns <= start_ns) {
    return;
  }
  Record r{};
  if (name == nullptr) {
    name = "<unknown>";
  }
  std::snprintf(r.name, kNameLen, "%s", name);
  r.stream_id = stream_id;
  r.duration_ms = static_cast<double>(end_ns - start_ns) / 1e6;
  std::lock_guard<std::mutex> lock(g_mu);
  g_records.push_back(r);
}

void CUPTIAPI buffer_requested(uint8_t** buffer, size_t* size, size_t* max_num_records) {
  *size = kBufSize;
  *buffer = static_cast<uint8_t*>(std::malloc(*size));
  *max_num_records = 0;
}

void CUPTIAPI buffer_completed(CUcontext /*ctx*/, uint32_t /*stream_id*/, uint8_t* buffer,
                               size_t /*size*/, size_t valid_size) {
  if (valid_size > 0) {
    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, valid_size, &record) == CUPTI_SUCCESS) {
      if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
          record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
        auto* k = reinterpret_cast<ARGUS_KERNEL_T*>(record);
        push_kernel(k->name, k->streamId, k->start, k->end);
      }
    }
  }
  std::free(buffer);
}

}  // namespace

extern "C" {

int argus_cupti_start(void) {
  if (g_started) {
    return 0;
  }
  {
    std::lock_guard<std::mutex> lock(g_mu);
    g_records.clear();
  }
  ARGUS_CUPTI_CHECK(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
  ARGUS_CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  g_started = true;
  return 0;
}

int argus_cupti_stop(void) {
  if (!g_started) {
    return 0;
  }
  ARGUS_CUPTI_CHECK(cuptiActivityFlushAll(0));
  ARGUS_CUPTI_CHECK(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  g_started = false;
  return 0;
}

void argus_cupti_clear(void) {
  std::lock_guard<std::mutex> lock(g_mu);
  g_records.clear();
}

size_t argus_cupti_count(void) {
  std::lock_guard<std::mutex> lock(g_mu);
  return g_records.size();
}

int argus_cupti_get(size_t idx, char* name_out, size_t name_len, uint32_t* stream_id,
                    double* duration_ms) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (idx >= g_records.size()) {
    return -1;
  }
  const Record& r = g_records[idx];
  if (name_out != nullptr && name_len > 0) {
    std::snprintf(name_out, name_len, "%s", r.name);
  }
  if (stream_id != nullptr) {
    *stream_id = r.stream_id;
  }
  if (duration_ms != nullptr) {
    *duration_ms = r.duration_ms;
  }
  return 0;
}

const char* argus_cupti_kernel_struct(void) {
#define ARGUS_STR_(x) #x
#define ARGUS_STR(x) ARGUS_STR_(x)
  return ARGUS_STR(ARGUS_KERNEL_T);
#undef ARGUS_STR
#undef ARGUS_STR_
}

}  // extern "C"
