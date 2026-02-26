#ifndef KERNELFORGE_PROFILING_H
#define KERNELFORGE_PROFILING_H

#ifdef KERNELFORGE_ENABLE_PROFILING

    #include <chrono>
    #include <cstdio>

namespace kf {

// High-resolution timer utilities
using ProfileClock = std::chrono::high_resolution_clock;
using ProfileTime = std::chrono::time_point<ProfileClock>;

// Inline helper to compute elapsed time in seconds
inline double elapsed_seconds(ProfileTime start, ProfileTime end) {
    return std::chrono::duration<double>(end - start).count();
}

    // Macros for timing blocks
    #define PROFILE_START(name) auto __profile_start_##name = kf::ProfileClock::now()

    #define PROFILE_END(name, accumulator)                                                    \
        do {                                                                                  \
            auto __profile_end_##name = kf::ProfileClock::now();                              \
            accumulator += kf::elapsed_seconds(__profile_start_##name, __profile_end_##name); \
        } while (0)

    #define PROFILE_SCOPE_START(name) auto __profile_scope_##name = kf::ProfileClock::now()

    #define PROFILE_SCOPE_END(name) kf::ProfileClock::now()

}  // namespace kf

#else

    // No-op macros when profiling is disabled (zero overhead)
    #define PROFILE_START(name)
    #define PROFILE_END(name, accumulator)
    #define PROFILE_SCOPE_START(name)
    #define PROFILE_SCOPE_END(name)

#endif  // KERNELFORGE_ENABLE_PROFILING

#endif  // KERNELFORGE_PROFILING_H
