//
// Runtime ISA tier for non-IQK quant vec_dot dispatch on x86 (GBCU groundwork).
//
#pragma once

#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Ordered for >= comparisons in dispatchers.
enum ggml_quant_dot_x86_isa {
    GGML_QUANT_DOT_X86_ISA_SCALAR = 0,
    GGML_QUANT_DOT_X86_ISA_SSE2   = 1,
    GGML_QUANT_DOT_X86_ISA_SSSE3  = 2,
    GGML_QUANT_DOT_X86_ISA_AVX    = 3,
    GGML_QUANT_DOT_X86_ISA_AVX2   = 4,
};

// Best ISA tier for this process (CPUID on x86; SCALAR elsewhere). Cached after first call.
GGML_API enum ggml_quant_dot_x86_isa ggml_quant_dot_x86_isa(void);

#ifdef __cplusplus
}
#endif
