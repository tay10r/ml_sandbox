#include "color_generator.h"

ColorGenerator::ColorGenerator(const int seed)
  : m_rng(seed)
{
}

auto
ColorGenerator::generate() -> Vec3
{
  constexpr float threshold = 0.005;

  constexpr float K = threshold * threshold;

  Vec3 color{ 0, 0, 0 };

  std::uniform_real_distribution<float> dist(0.1, 1.0);

  while (true) {

    color = Vec3(dist(m_rng), dist(m_rng), dist(m_rng));

    for (const auto& existing : m_existing) {
      const auto delta = existing - color;
      if (dot(delta, delta) > K)
        break;
    }

    m_existing.emplace_back(color);

    break;
  }

  return color;
}
