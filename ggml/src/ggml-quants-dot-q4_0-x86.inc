// Included from ggml-quants.c after x86 SIMD helpers (mul_sum_i8_pairs, etc.).

static void ggml_vec_dot_q4_0_q8_0_generic(int nb, int ib, const block_q4_0 * x, const block_q8_0 * y, float * sumf) {
    const int qk = QK8_0;

    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        const int sumi = sumi0 + sumi1;
        *sumf += sumi*GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d);
    }
}

#if defined(__AVX2__)
static void ggml_vec_dot_q4_0_q8_0_avx2(int nb, int * ib, const block_q4_0 * x, const block_q8_0 * y, float * sumf) {
    __m256 acc = _mm256_setzero_ps();

    for (int i = *ib; i < nb; ++i) {
        const __m256 d = _mm256_set1_ps(GGML_FP16_TO_FP32(x[i].d) * GGML_FP16_TO_FP32(y[i].d));

        __m256i qx = bytes_from_nibbles_32(x[i].qs);
        const __m256i off = _mm256_set1_epi8(8);
        qx = _mm256_sub_epi8(qx, off);

        __m256i qy = _mm256_loadu_si256((const __m256i *) y[i].qs);
        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        acc = _mm256_fmadd_ps(d, q, acc);
    }

    *sumf += hsum_float_8(acc);
    *ib = nb;
}
#endif

#if defined(__AVX__)
static void ggml_vec_dot_q4_0_q8_0_avx(int nb, int * ib, const block_q4_0 * x, const block_q8_0 * y, float * sumf) {
    __m256 acc = _mm256_setzero_ps();

    for (int i = *ib; i < nb; ++i) {
        const __m256 d = _mm256_set1_ps(GGML_FP16_TO_FP32(x[i].d) * GGML_FP16_TO_FP32(y[i].d));

        const __m128i lowMask = _mm_set1_epi8(0xF);
        const __m128i off = _mm_set1_epi8(8);

        const __m128i tmp = _mm_loadu_si128((const __m128i *) x[i].qs);

        __m128i bx_0 = _mm_and_si128(lowMask, tmp);
        __m128i by_0 = _mm_loadu_si128((const __m128i *) y[i].qs);
        bx_0 = _mm_sub_epi8(bx_0, off);
        const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

        bx_0 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp, 4));
        by_0 = _mm_loadu_si128((const __m128i *) (y[i].qs + 16));
        bx_0 = _mm_sub_epi8(bx_0, off);
        const __m128i i32_1 = mul_sum_i8_pairs(bx_0, by_0);

        __m256 p = _mm256_cvtepi32_ps(MM256_SET_M128I(i32_0, i32_1));
        acc = _mm256_add_ps(_mm256_mul_ps(d, p), acc);
    }

    *sumf += hsum_float_8(acc);
    *ib = nb;
}
#endif

#if defined(__SSSE3__)
// Available in ggml-quants.c only when SSSE3 is the highest x86 tier at compile time; duplicate for dispatch builds with AVX/AVX2.
static inline float ggml_hsum_float_4x4(const __m128 a, const __m128 b, const __m128 c, const __m128 d) {
    __m128 res_0 = _mm_hadd_ps(a, b);
    __m128 res_1 = _mm_hadd_ps(c, d);
    __m128 res   = _mm_hadd_ps(res_0, res_1);
    res = _mm_hadd_ps(res, res);
    res = _mm_hadd_ps(res, res);
    return _mm_cvtss_f32(res);
}

static void ggml_vec_dot_q4_0_q8_0_ssse3(int nb, int * ib, const block_q4_0 * x, const block_q8_0 * y, float * sumf) {
    const __m128i lowMask = _mm_set1_epi8(0xF);
    const __m128i off = _mm_set1_epi8(8);

    __m128 acc_0 = _mm_setzero_ps();
    __m128 acc_1 = _mm_setzero_ps();
    __m128 acc_2 = _mm_setzero_ps();
    __m128 acc_3 = _mm_setzero_ps();

    int i = *ib;
    for (; i + 1 < nb; i += 2) {
        _mm_prefetch(&x[i] + sizeof(block_q4_0), _MM_HINT_T0);
        _mm_prefetch(&y[i] + sizeof(block_q8_0), _MM_HINT_T0);

        const __m128 d_0_1 = _mm_set1_ps(GGML_FP16_TO_FP32(x[i].d) * GGML_FP16_TO_FP32(y[i].d));

        const __m128i tmp_0_1 = _mm_loadu_si128((const __m128i *) x[i].qs);

        __m128i bx_0 = _mm_and_si128(lowMask, tmp_0_1);
        __m128i by_0 = _mm_loadu_si128((const __m128i *) y[i].qs);
        bx_0 = _mm_sub_epi8(bx_0, off);
        const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

        __m128i bx_1 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_0_1, 4));
        __m128i by_1 = _mm_loadu_si128((const __m128i *) (y[i].qs + 16));
        bx_1 = _mm_sub_epi8(bx_1, off);
        const __m128i i32_1 = mul_sum_i8_pairs(bx_1, by_1);

        const __m128 d_2_3 = _mm_set1_ps(GGML_FP16_TO_FP32(x[i + 1].d) * GGML_FP16_TO_FP32(y[i + 1].d));

        const __m128i tmp_2_3 = _mm_loadu_si128((const __m128i *) x[i + 1].qs);

        __m128i bx_2 = _mm_and_si128(lowMask, tmp_2_3);
        __m128i by_2 = _mm_loadu_si128((const __m128i *) y[i + 1].qs);
        bx_2 = _mm_sub_epi8(bx_2, off);
        const __m128i i32_2 = mul_sum_i8_pairs(bx_2, by_2);

        __m128i bx_3 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_2_3, 4));
        __m128i by_3 = _mm_loadu_si128((const __m128i *) (y[i + 1].qs + 16));
        bx_3 = _mm_sub_epi8(bx_3, off);
        const __m128i i32_3 = mul_sum_i8_pairs(bx_3, by_3);

        __m128 p0 = _mm_cvtepi32_ps(i32_0);
        __m128 p1 = _mm_cvtepi32_ps(i32_1);
        __m128 p2 = _mm_cvtepi32_ps(i32_2);
        __m128 p3 = _mm_cvtepi32_ps(i32_3);

        acc_0 = _mm_add_ps(_mm_mul_ps(d_0_1, p0), acc_0);
        acc_1 = _mm_add_ps(_mm_mul_ps(d_0_1, p1), acc_1);
        acc_2 = _mm_add_ps(_mm_mul_ps(d_2_3, p2), acc_2);
        acc_3 = _mm_add_ps(_mm_mul_ps(d_2_3, p3), acc_3);
    }

    *sumf += ggml_hsum_float_4x4(acc_0, acc_1, acc_2, acc_3);
    *ib = i;
}
#endif

static void ggml_vec_dot_q4_0_q8_0_x86(int nb, int * ib, const block_q4_0 * x, const block_q8_0 * y, float * sumf) {
    const enum ggml_quant_dot_x86_isa isa = ggml_quant_dot_x86_isa();

#if defined(__AVX2__)
    if (isa >= GGML_QUANT_DOT_X86_ISA_AVX2) {
        ggml_vec_dot_q4_0_q8_0_avx2(nb, ib, x, y, sumf);
        return;
    }
#endif
#if defined(__AVX__)
    if (isa >= GGML_QUANT_DOT_X86_ISA_AVX) {
        ggml_vec_dot_q4_0_q8_0_avx(nb, ib, x, y, sumf);
        return;
    }
#endif
#if defined(__SSSE3__)
    if (isa >= GGML_QUANT_DOT_X86_ISA_SSSE3) {
        ggml_vec_dot_q4_0_q8_0_ssse3(nb, ib, x, y, sumf);
        ggml_vec_dot_q4_0_q8_0_generic(nb, *ib, x, y, sumf);
        return;
    }
#endif
    // SSE2 (and below): scalar reference for now; dedicated SSE2 kernel is a follow-up.
    ggml_vec_dot_q4_0_q8_0_generic(nb, *ib, x, y, sumf);
}
