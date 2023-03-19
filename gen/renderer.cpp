#include "renderer.h"

#include "scene.h"

#include <limits>

#include <bvh/v2/executor.h>
#include <bvh/v2/thread_pool.h>

Renderer::Renderer(const int w, const int h, const int seed)
  : m_rngs(w * h)
  , m_width(w)
  , m_height(h)
{
  std::mt19937 rng(seed);

  for (int i = 0; i < (w * h); i++)
    m_rngs[i] = Rng(rng());
}

auto
Renderer::render(const Scene& scene, const Vec3& cameraPos) -> Result
{
  using Ray = Scene::Ray;

  Result result(m_width, m_height);

  const auto u_scale{ 1.0f / static_cast<float>(m_width) };
  const auto v_scale{ 1.0f / static_cast<float>(m_height) };

  bvh::v2::ThreadPool thread_pool;

  bvh::v2::ParallelExecutor executor(thread_pool);

  const std::size_t pixel_count = m_width * m_height;

  const Vec3 worldUp(0, 1, 0);

  const Vec3 cameraTarget(0, 12, 0);
  const Vec3 cameraDir = normalize(cameraTarget - cameraPos);
  const Vec3 cameraRight = cross(cameraDir, worldUp);
  const Vec3 cameraUp = cross(cameraRight, cameraDir);

  const float aspect = static_cast<float>(m_width) / static_cast<float>(m_height);

  auto generateRay = [&](const float u, const float v) -> Ray {
    const float dx = (u * 2.0f - 1.0f) * m_fov * aspect;
    const float dy = (1.0f - v * 2.0f) * m_fov;
    return Ray(cameraPos, normalize(cameraDir + cameraUp * dy + cameraRight * dx), 0, m_maxDistance);
  };

  executor.for_each(0, pixel_count, [&](const std::size_t begin, const std::size_t end) {
    for (auto i = begin; i < end; i++) {

      const auto x = i % m_width;
      const auto y = i / m_width;

      // First we get surface info from the center pixel.
      // We get color separately, since it requires multi sampling.

      {
        const auto u = (static_cast<float>(x) + 0.5f) * u_scale;
        const auto v = (static_cast<float>(y) + 0.5f) * v_scale;

        auto ray = generateRay(u, v);

        const auto surfaceInfo{ getSurfaceInfo(scene, ray) };

        result.albedo[i] = surfaceInfo.albedo;

        result.depth[i] = surfaceInfo.depth;

        result.normal[i] = surfaceInfo.normal;

        result.segmentation[i] = surfaceInfo.segmentation;

        result.stencil[i] = surfaceInfo.objectMask ? 0xff : 0;
      }

      // Now we get color

      std::uniform_real_distribution<float> uv_dist(0, 1);

      constexpr int low_spp{ 16 };

      for (int j = 0; j < low_spp; j++) {

        const auto u = (static_cast<float>(x) + uv_dist(m_rngs[i])) * u_scale;
        const auto v = (static_cast<float>(y) + uv_dist(m_rngs[i])) * v_scale;

        auto ray = generateRay(u, v);

        const auto color = trace(scene, ray, m_rngs[i], 0);

        result.noisy_color[i] = result.noisy_color[i] + color * (1.0f / static_cast<float>(low_spp));
      }

      const int high_spp{ 256 };

      for (int j = 0; j < high_spp; j++) {

        const auto u = (static_cast<float>(x) + uv_dist(m_rngs[i])) * u_scale;
        const auto v = (static_cast<float>(y) + uv_dist(m_rngs[i])) * v_scale;

        auto ray = generateRay(u, v);

        const auto color = trace(scene, ray, m_rngs[i], 0);

        result.color[i] = result.color[i] + color * (1.0f / static_cast<float>(high_spp));
      }
    }
  });

  return result;
}

namespace {

using Vec3 = bvh::v2::Vec<float, 3>;

Vec3
mix(const Vec3& a, const Vec3& b, const float alpha)
{
  return a + (b - a) * alpha;
}

Vec3
depthToColor(const float depth, const float minDepth, const float maxDepth)
{
  if (depth < minDepth)
    return Vec3(0, 0, 0);

  if (depth > maxDepth)
    return Vec3(0, 0, 0);

  const float depthScale = 1.0f / (maxDepth - minDepth);

  const float alpha = (depth - minDepth) * depthScale;

  if (alpha <= 0.5)
    return mix(Vec3(1, 0, 0), Vec3(0, 1, 0), alpha * 2.0f);
  else
    return mix(Vec3(0, 1, 0), Vec3(0, 0, 1), (alpha - 0.5f) * 2.0f);
}

} // namespace

auto
Renderer::getSurfaceInfo(const Scene& scene, Ray& ray) -> SurfaceInfo
{
  auto hit{ scene.intersect(ray) };

  if (!hit)
    return SurfaceInfo{ onMiss(ray), Vec3(0, 0, 0), -ray.dir, Vec3(0, 0, 0) };

  const auto albedo = hit->albedo;

  const auto emission = hit->emission;

  const auto normal = hit->normal;

  return { albedo,
           depthToColor(ray.tmax, m_minDistance, m_maxDistance),
           (normal + Vec3(1.0f)) * 0.5f,
           hit->segmentation,
           hit->objectMask };
}

void
Renderer::setSkyColors(const Vec3& lo, const Vec3& hi)
{
  m_skyLow = lo;
  m_skyHigh = hi;
}

auto
Renderer::trace(const Scene& scene, Ray& ray, Rng& rng, int depth) -> Vec3
{
  if (depth > m_maxDepth)
    return Vec3(0, 0, 0);

  auto hit{ scene.intersect(ray) };

  if (!hit)
    return onMiss(ray);

  const auto albedo = hit->albedo;

  const auto emission = hit->emission;

  const auto normal = hit->normal;

  const auto next_dir = sampleHemisphere(rng, normal);

  const auto next_org = ray.org + (ray.dir * (ray.tmax - 0.001f));

  auto second_ray{ Ray(next_org, next_dir, 0.0f, std::numeric_limits<float>::infinity()) };

  return albedo * trace(scene, second_ray, rng, depth + 1) + emission;
}

auto
Renderer::onMiss(const Ray& ray) -> Vec3
{
  const auto up = Vec3(0, 1, 0);
  const auto level = (dot(ray.dir, up) + 1.0f) * 0.5f;
  return m_skyLow + (m_skyHigh - m_skyLow) * level;
}

auto
Renderer::sampleHemisphere(Rng& rng, const Vec3& n) -> Vec3
{
  std::uniform_real_distribution<float> dist(-1, 1);

  while (true) {
    const auto v{ Vec3(dist(rng), dist(rng), dist(rng)) };
    if (dot(v, v) <= 1.0f) {
      if (dot(v, n) < 0)
        return normalize(v * ((dot(v, n) < 0) ? -1.0f : 1.0f));
    }
  }

  return Vec3(0, 1, 0);
}
