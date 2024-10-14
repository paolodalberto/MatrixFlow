#include <cstddef>
#include <cstdint>
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__)
#include <x86intrin.h>
#endif

#ifdef __FAST_MATH__
#error Compensated summation is unsafe with -ffast-math (/fp:fast)
#endif

using std::size_t;
using std::uint32_t;

// Kahan summation
inline static void kadd(double& sum, double& c, double y) {
  y -= c;
  auto t = sum + y;
  c = (t - sum) - y;
  sum = t;
}

inline static void kadd(__m256d& sumV, __m256d& cV, __m256d y) {
  y = _mm256_sub_pd(y, cV);
  __m256d t = _mm256_add_pd(sumV, y);
  cV = _mm256_sub_pd(_mm256_sub_pd(t, sumV), y);
  sumV = t;
}

#ifdef __AVX512F__
inline static void kadd(__m512d& sumV, __m512d& cV, __m512d y) {
  y = _mm512_sub_pd(y, cV);
  __m512d t = _mm512_add_pd(sumV, y);
  cV = _mm512_sub_pd(_mm512_sub_pd(t, sumV), y);
  sumV = t;
}
#endif

// Loads the first N floats starting at p.
inline static __m128 mm_load_partial_ps(const float* p, size_t N) {
  // This uses BMI2, which doesn't exist in Intel Sandy/Ivy Bridge or AMD
  // Jaguar/Puma/Bulldozer/Piledriver/Steamroller, but does exist in all other
  // AVX-supporting CPUs.
  uint32_t k1 = _bzhi_u32(-1, N); // set N low bits
  k1 = _pdep_u32(k1, 0x80808080); // deposit the set bits into high bit of each byte
  __m128i k2 = _mm_set1_epi32(k1); // broadcast to vector
  k2 = _mm_cvtepi8_epi32(k2); // expand 8-bit els into 32-bit els
  return _mm_maskload_ps(p, k2);
}

#ifdef __AVX512F__
inline static __m256 mm256_load_partial_ps(const float* p, size_t N) {
  uint8_t k1 = _bzhi_u32(0xFF, N); // set N low bits
  return _mm256_maskz_load_ps(k1, p);
}
#endif

// Zeros r
inline static void zeroVecArr(__m256d* r, size_t N) {
  for (size_t i = 0; i < N; i++) r[i] = _mm256_setzero_pd();
}

#ifdef __AVX512F__
inline static void zeroVecArr(__m512d* r, size_t N) {
  for (size_t i = 0; i < N; i++) r[i] = _mm512_setzero_pd();
}
#endif

double fastAccurate(const float* values, size_t len) {
  constexpr size_t NACC = 8; // 8 vector accumulators
  constexpr size_t ELS_PER_VEC = 4; // 4 doubles per 256b vec
  const float* const end = values + len;

  double sum = 0., c = 0.;
  // Align to 16B boundary
  while (reinterpret_cast<uintptr_t>(values) % 16 && values < end) {
    kadd(sum, c, *values);
    values++;
  }

  __m256d sumV[NACC], cV[NACC];
  zeroVecArr(sumV, NACC);
  zeroVecArr(cV, NACC);
  // Continue the compensation from the alignment loop.
  sumV[0] = _mm256_setr_pd(sum, 0., 0., 0.);
  cV[0] = _mm256_setr_pd(c, 0., 0., 0.);

  // Main vectorized loop:
  while (values < end - NACC * ELS_PER_VEC) {
    for (size_t i = 0; i < NACC; i++) {
      __m256d y = _mm256_cvtps_pd(_mm_load_ps(values));
      kadd(sumV[i], cV[i], y);
      values += ELS_PER_VEC;
    }
  }

  // Up to NACC * ELS_PER_VEC values remain.
  while (values < end - ELS_PER_VEC) {
    __m256d y = _mm256_cvtps_pd(_mm_load_ps(values));
    kadd(sumV[0], cV[0], y);
    values += ELS_PER_VEC;
  }

  // Up to ELS_PER_VEC values remain. Use masked loads.
  __m256d y = _mm256_cvtps_pd(mm_load_partial_ps(values, end - values));
  kadd(sumV[0], cV[0], y);

  // Fold the accumulators together.
  for (size_t i = 1; i < NACC; i++)
    kadd(sumV[0], cV[0], sumV[i]);

  // Horizontally add the elements of sumV[0].
  // (Consider using compensated summation here, but we can probably assume that
  // all of our accumulators are similar magnitude at this point.)
  __m128d lo = _mm256_castpd256_pd128(sumV[0]);
  __m128d hi = _mm256_extractf128_pd(sumV[0], 1);
  lo = _mm_add_pd(lo, hi); // 0+2, 1+3
  __m128d hi64 = _mm_unpackhi_pd(lo, lo);
  return _mm_cvtsd_f64(_mm_add_sd(lo, hi64)); // 0+2+1+3
}

#ifdef __AVX512F__
double fastAccurateAVX512(const float* values, size_t len) {
  constexpr size_t NACC = 8; // 8 vector accumulators
  constexpr size_t ELS_PER_VEC = 8; // 8 doubles per 512b vec
  const float* const end = values + len;

  double sum = 0., c = 0.;
  // Align to 32B boundary
  while (reinterpret_cast<uintptr_t>(values) % 32 && values < end) {
    kadd(sum, c, *values);
    values++;
  }

  __m512d sumV[NACC], cV[NACC];
  zeroVecArr(sumV, NACC);
  zeroVecArr(cV, NACC);
  // Continue the compensation from the alignment loop.
  sumV[0] = _mm512_setr_pd(sum, 0., 0., 0., 0., 0., 0., 0.);
  cV[0] = _mm512_setr_pd(c, 0., 0., 0., 0., 0., 0., 0.);

  // Main vectorized loop:
  while (values < end - NACC * ELS_PER_VEC) {
    for (size_t i = 0; i < NACC; i++) {
      __m512d y = _mm512_cvtps_pd(_mm256_load_ps(values));
      kadd(sumV[i], cV[i], y);
      values += ELS_PER_VEC;
    }
  }
  
  // Up to NACC * ELS_PER_VEC = 64 values remain.
  while (values < end - ELS_PER_VEC) {
      __m512d y = _mm512_cvtps_pd(_mm256_load_ps(values));
      kadd(sumV[0], cV[0], y);
      values += ELS_PER_VEC;
  }

  // Up to ELS_PER_VEC values remain.
  __m512d y = _mm512_cvtps_pd(mm256_load_partial_ps(values, end - values));
  kadd(sumV[0], cV[0], y);

  // Fold the accumulators together.
  for (size_t i = 1; i < NACC; i++)
    kadd(sumV[0], cV[0], sumV[i]);

  // Horizontally add the elements of sumV[0].
  // (Consider using compensated summation here, but we can probably assume that
  // all of our accumulators are similar magnitude at this point.)
  // (Note: this intrinsic doesn't map to a single instruction.)
  return _mm512_reduce_add_pd(sumV[0]);
}
#endif
