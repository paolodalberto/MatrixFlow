#include <cstddef>
#include <cmath>
#ifdef __FAST_MATH__
#error Compensated summation is unsafe with -ffast-math (/fp:fast)
#endif

inline static void kadd(double& sum, double& c, double y) {
  y -= c;
  auto t = sum + y;
  c = (t - sum) - y;
  sum = t;
}

double accurate(const float* values, std::size_t len) {
  double sum = 0, c = 0;
  for (const float* const end = values + len; values < end; values++) {
    auto y = static_cast<double>(*values);
    kadd(sum, c, y);
  }
  return sum + c;
}
