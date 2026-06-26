//===-- test/globalbenchmarks/ScopedPerfCounters.h ---------------*- C++ -*-===//
//
// ScopedPerfCounters — count CPU hardware events for one code region only
// ============================================================================
//
// A tiny RAII wrapper around Linux perf_event_open(2). Unlike running the whole
// process under `perf stat`, this measures *only* the region between Start() and
// Stop(), so one-off costs outside the region (here: the BVH build, which differs
// 5x between the SweepSAH and binned-SAH builders) do not contaminate the query
// counters we actually want to compare.
//
// Counters are opened as a single group (scheduled together, so their ratios are
// consistent) in user space only (exclude_kernel/exclude_hv). Each counter is
// opened independently and silently skipped if the kernel/PMU refuses it, so the
// header degrades gracefully on restricted hosts (perf_event_paranoid, VMs, ...).
//
// Usage:
//   ScopedPerfCounters pc("plain (v2) query");
//   pc.Start();
//   ... region to measure ...
//   pc.Stop();
//   pc.Report();   // prints to stderr; also accumulates across Start/Stop pairs
//
// Only compiled meaningfully on Linux; elsewhere the methods are no-ops.

#ifndef VECGEOM_TEST_SCOPEDPERFCOUNTERS_H_
#define VECGEOM_TEST_SCOPEDPERFCOUNTERS_H_

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#if defined(__linux__)
#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>

class ScopedPerfCounters {
public:
  explicit ScopedPerfCounters(std::string label) : fLabel(std::move(label))
  {
    // (name, type, config). HW_CACHE configs are packed (id | op<<8 | result<<16).
    AddCounter("cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    AddCounter("instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    AddCounter("branches", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS);
    AddCounter("branch-misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
    AddCounter("L1d-read-miss", PERF_TYPE_HW_CACHE,
               PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                   (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
    AddCounter("LLC-read-miss", PERF_TYPE_HW_CACHE,
               PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                   (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
  }

  ~ScopedPerfCounters()
  {
    for (auto &c : fCounters)
      if (c.fd >= 0) close(c.fd);
  }

  void Start()
  {
    for (auto &c : fCounters) {
      if (c.fd < 0) continue;
      ioctl(c.fd, PERF_EVENT_IOC_RESET, 0);
      ioctl(c.fd, PERF_EVENT_IOC_ENABLE, 0);
    }
  }

  void Stop()
  {
    for (auto &c : fCounters) {
      if (c.fd < 0) continue;
      ioctl(c.fd, PERF_EVENT_IOC_DISABLE, 0);
      uint64_t v = 0;
      if (read(c.fd, &v, sizeof(v)) == sizeof(v)) c.accum += v;
    }
  }

  void Report() const
  {
    std::fprintf(stderr, "[perf] %-24s", fLabel.c_str());
    uint64_t cycles = 0, insns = 0;
    for (auto const &c : fCounters) {
      if (c.fd < 0) continue;
      if (c.name == "cycles") cycles = c.accum;
      if (c.name == "instructions") insns = c.accum;
    }
    for (auto const &c : fCounters) {
      if (c.fd < 0) {
        std::fprintf(stderr, "  %s=NA", c.name.c_str());
        continue;
      }
      std::fprintf(stderr, "  %s=%llu", c.name.c_str(), (unsigned long long)c.accum);
    }
    if (cycles && insns) std::fprintf(stderr, "  IPC=%.3f", double(insns) / double(cycles));
    std::fprintf(stderr, "\n");
  }

private:
  struct Counter {
    std::string name;
    int fd       = -1;
    uint64_t accum = 0;
  };

  static long perf_event_open(struct perf_event_attr *attr, pid_t pid, int cpu, int group_fd, unsigned long flags)
  {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
  }

  void AddCounter(const char *name, uint32_t type, uint64_t config)
  {
    struct perf_event_attr attr;
    std::memset(&attr, 0, sizeof(attr));
    attr.type           = type;
    attr.size           = sizeof(attr);
    attr.config         = config;
    attr.disabled       = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv     = 1;
    Counter c;
    c.name = name;
    c.fd   = (int)perf_event_open(&attr, 0, -1, -1, 0);
    if (c.fd < 0)
      std::fprintf(stderr, "[perf] warning: counter '%s' unavailable (%s)\n", name, std::strerror(errno));
    fCounters.push_back(std::move(c));
  }

  std::string fLabel;
  std::vector<Counter> fCounters;
};

#else // !__linux__

class ScopedPerfCounters {
public:
  explicit ScopedPerfCounters(std::string) {}
  void Start() {}
  void Stop() {}
  void Report() const {}
};

#endif

#endif // VECGEOM_TEST_SCOPEDPERFCOUNTERS_H_
