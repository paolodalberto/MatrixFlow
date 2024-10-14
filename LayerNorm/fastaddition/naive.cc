#include <cstddef>

float naive(const float* values, std::size_t len) {
  float sum = 0.f;
  for (const float* const end = values + len; values < end; values++) {
    sum += *values;
  }
  return sum;
}
