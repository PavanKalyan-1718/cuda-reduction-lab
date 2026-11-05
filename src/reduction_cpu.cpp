#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  const std::size_t count = argc > 1 ? std::stoull(argv[1]) : 1'000'000;
  if (count == 0) {
    throw std::invalid_argument("count must be positive");
  }

  std::vector<float> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    values[index] = static_cast<float>(index + 1);
  }

  const auto start = std::chrono::steady_clock::now();
  const double observed = std::accumulate(values.begin(), values.end(), 0.0);
  const auto stop = std::chrono::steady_clock::now();
  const double expected = static_cast<double>(count) * (count + 1) / 2.0;
  const double milliseconds = std::chrono::duration<double, std::milli>(stop - start).count();
  const double gib_per_second = (count * sizeof(float)) / (milliseconds / 1000.0) / (1024.0 * 1024.0 * 1024.0);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "elements=" << count << " sum=" << observed << " expected=" << expected
            << " elapsed_ms=" << milliseconds << " effective_GiB_s=" << gib_per_second << "\n";
  return std::abs(observed - expected) < 0.5 ? 0 : 1;
}

