#pragma once

#include <bvh/v2/vec.h>

#include <random>
#include <vector>

class ColorGenerator final
{
public:
  using Vec3 = bvh::v2::Vec<float, 3>;

  ColorGenerator(int seed);

  Vec3 generate();

private:
  std::vector<Vec3> m_existing;

  std::mt19937 m_rng;
};
