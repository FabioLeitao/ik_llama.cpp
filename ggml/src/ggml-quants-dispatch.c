//
// x86 CPU feature detection for quant vec_dot dispatch (runtime, not compile-time stubs).
//
#include "ggml-quants-dispatch.h"

#include "ggml.h"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

static void ggml_quant_dot_cpuid(int leaf, int subleaf, uint32_t regs[4]) {
#if defined(_MSC_VER)
    __cpuidex((int *) regs, leaf, subleaf);
#else
    __cpuid_count(leaf, subleaf, regs[0], regs[1], regs[2], regs[3]);
#endif
}

static enum ggml_quant_dot_x86_isa ggml_quant_dot_x86_isa_detect(void) {
    uint32_t r0[4], r7[4];

    ggml_quant_dot_cpuid(0, 0, r0);
    if (r0[0] == 0) {
        return GGML_QUANT_DOT_X86_ISA_SCALAR;
    }

    ggml_quant_dot_cpuid(1, 0, r0);
    const uint32_t ecx = r0[2];
    const uint32_t edx = r0[3];

    const int has_sse2  = (edx >> 26) & 1;
    const int has_ssse3 = (ecx >>  9) & 1;
    const int has_avx   = (ecx >> 28) & 1;

    int has_avx2 = 0;
    if (r0[0] >= 7) {
        ggml_quant_dot_cpuid(7, 0, r7);
        has_avx2 = (r7[1] >> 5) & 1;
    }

    // OS must enable XSAVE for YMM; skip AVX* if not (conservative).
    const int os_avx = has_avx && ((ecx >> 27) & 1) && ((ecx >> 26) & 1);
    if (has_avx2 && os_avx) {
        return GGML_QUANT_DOT_X86_ISA_AVX2;
    }
    if (os_avx) {
        return GGML_QUANT_DOT_X86_ISA_AVX;
    }
    if (has_ssse3) {
        return GGML_QUANT_DOT_X86_ISA_SSSE3;
    }
    if (has_sse2) {
        return GGML_QUANT_DOT_X86_ISA_SSE2;
    }
    return GGML_QUANT_DOT_X86_ISA_SCALAR;
}

#endif // x86

enum ggml_quant_dot_x86_isa ggml_quant_dot_x86_isa(void) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    static enum ggml_quant_dot_x86_isa cached = GGML_QUANT_DOT_X86_ISA_SCALAR;
    static int init = 0;
    if (!init) {
        cached = ggml_quant_dot_x86_isa_detect();
        init = 1;
    }
    return cached;
#else
    return GGML_QUANT_DOT_X86_ISA_SCALAR;
#endif
}
