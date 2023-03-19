#pragma once

#include "image.h"

#include <bvh/v2/ray.h>
#include <bvh/v2/vec.h>

#include <glm/glm.hpp>

#include <random>

#include <cmath>
#include <cstdint>

template<typename T>
class Image;

class Scene;

class Renderer final
{
public:
  using Vec3 = bvh::v2::Vec<float, 3>;

  using Ray = bvh::v2::Ray<float, 3>;

  struct Result final
  {
    Image<Vec3> albedo;

    Image<Vec3> noisy_color;

    Image<Vec3> color;

    Image<Vec3> normal;

    Image<Vec3> depth;

    Image<Vec3> segmentation;

    Image<unsigned char> stencil;

    Result(int w, int h)
      : albedo(w, h)
      , noisy_color(w, h)
      , color(w, h)
      , normal(w, h)
      , depth(w, h)
      , segmentation(w, h)
      , stencil(w, h)
    {
    }
  };

  explicit Renderer(int w, int h, int seed);

  Result render(const Scene& scene, const Vec3& cameraPos);

  void setSkyColors(const std::uint32_t lo, const std::uint32_t hi)
  {
    auto toFlt = [](const std::uint32_t color, std::uint32_t bitShift) -> float {
      return ((color >> bitShift) & 0xff) / 255.0f;
    };

    setSkyColors({ toFlt(lo, 16), toFlt(lo, 8), toFlt(lo, 0) }, { toFlt(hi, 16), toFlt(hi, 8), toFlt(hi, 0) });
  }

  void setSkyColors(const Vec3& lo, const Vec3& hi);

protected:
  using Rng = std::minstd_rand;

  struct SurfaceInfo final
  {
    Vec3 albedo;

    Vec3 depth;

    Vec3 normal;

    Vec3 segmentation;

    bool objectMask;
  };

  SurfaceInfo getSurfaceInfo(const Scene& scene, Ray& ray);

  Vec3 trace(const Scene& scene, Ray& ray, Rng& rng, int depth);

  Vec3 onMiss(const Ray& ray);

  static Vec3 sampleHemisphere(Rng& rng, const Vec3& n);

private:
  std::vector<Rng> m_rngs;

  const int m_width;

  const int m_height;

  const int m_maxDepth{ 5 };

  const float m_minDistance{ 15.0f };

  const float m_maxDistance{ 100.0f };

  const float m_fov = std::tan(glm::radians(45.0f) * 0.5f);

  Vec3 m_skyLow{ 1.0f, 1.0f, 1.0f };

  Vec3 m_skyHigh{ 0.5f, 0.7f, 1.0f };
};
