#include "scene.h"

#include <bvh/v2/default_builder.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/thread_pool.h>

#include <fstream>
#include <random>

#include <cstdint>

namespace {

std::vector<uint8_t>
read_whole_file(const char* path)
{
  std::vector<uint8_t> data;

  std::ifstream file(path, std::ios::binary | std::ios::in);

  if (!file.good())
    return data;

  file.seekg(0, std::ios::end);

  const auto file_end = file.tellg();

  if (file_end == -1l)
    return data;

  file.seekg(0, std::ios::beg);

  const auto file_size = static_cast<std::size_t>(file_end);

  data.resize(file_size);

  file.read(reinterpret_cast<char*>(data.data()), data.size());

  return data;
}

} // namespace

bool
Scene::loadModel(const char* path, const Vec3& albedo, const Vec3& emission, const Vec3& segmentation)
{
  const auto data{ read_whole_file(path) };

  constexpr std::size_t header_size{ 84 };

  if (data.size() < header_size)
    return false;

  const std::size_t tri_count{ *reinterpret_cast<const std::uint32_t*>(data.data() + 80) };

  constexpr std::size_t bytes_per_tri = 50;

  if (data.size() < (header_size + bytes_per_tri * tri_count))
    return false;

  Model model;

  for (std::size_t i = 0; i < tri_count; i++) {

    const float* tri_ptr = reinterpret_cast<const float*>(data.data() + header_size + i * bytes_per_tri);

    const float norm_x = tri_ptr[0];
    const float norm_y = tri_ptr[1];
    const float norm_z = tri_ptr[2];

    const float a_x = tri_ptr[3];
    const float a_y = tri_ptr[4];
    const float a_z = tri_ptr[5];

    const float b_x = tri_ptr[6];
    const float b_y = tri_ptr[7];
    const float b_z = tri_ptr[8];

    const float c_x = tri_ptr[9];
    const float c_y = tri_ptr[10];
    const float c_z = tri_ptr[11];

    const auto a = Vec3(a_x, a_y, a_z);
    const auto b = Vec3(b_x, b_y, b_z);
    const auto c = Vec3(c_x, c_y, c_z);

    model.normals.emplace_back(Vec3(norm_x, norm_y, norm_z));

    model.primitives.emplace_back(Model::Tri(a, b, c));
  }

  model.albedo = albedo;

  model.emission = emission;

  model.segmentation = segmentation;

  m_models.emplace_back(std::move(model));

  return true;
}

namespace {

bool
is_empty_spot(const std::vector<Scene::Vec3>& spots, const Scene::Vec3& spot, const float distance_threshold)
{
  const float d_sq = distance_threshold * distance_threshold;

  for (const auto& existing_spot : spots) {

    const auto delta = existing_spot - spot;

    const auto delta_sq = dot(delta, delta);

    if (delta_sq < d_sq)
      return false;
  }

  return true;
}

} // namespace

void
Scene::instanceRange(const std::size_t offset,
                     const std::size_t count,
                     const glm::mat4& transform,
                     const std::optional<Vec3>& albedoOverride,
                     const bool objectMask)
{
  for (std::size_t i = 0; i < count; i++)
    instance(m_models[i + offset], transform, albedoOverride, objectMask);
}

namespace {

bvh::v2::PrecomputedTri<float>
instanceTri(const bvh::v2::Tri<float, 3>& tri, const glm::mat4& transform)
{
  const auto p0 = transform * glm::vec4(tri.p0[0], tri.p0[1], tri.p0[2], 1.0f);
  const auto p1 = transform * glm::vec4(tri.p1[0], tri.p1[1], tri.p1[2], 1.0f);
  const auto p2 = transform * glm::vec4(tri.p2[0], tri.p2[1], tri.p2[2], 1.0f);

  using Vec3 = bvh::v2::Vec<float, 3>;

  return bvh::v2::Tri<float, 3>(Vec3(p0.x, p0.y, p0.z), Vec3(p1.x, p1.y, p1.z), Vec3(p2.x, p2.y, p2.z));
}

} // namespace

void
Scene::instance(const Model& model,
                const glm::mat4& transform,
                const std::optional<Vec3>& albedoOverride,
                const bool objectMask)
{
  const auto albedo = albedoOverride.has_value() ? albedoOverride.value() : model.albedo;

  for (std::size_t i = 0; i < model.primitives.size(); i++) {

    const auto t = instanceTri(model.primitives[i], transform);

    m_primitives.emplace_back(t);

    m_attributes.emplace_back(Attributes{ model.normals[i], albedo, model.emission, model.segmentation, objectMask });
  }
}

void
Scene::commit()
{
  using BBox = bvh::v2::BBox<float, 3>;

  if (m_primitives.empty())
    return;

  bvh::v2::ThreadPool thread_pool;

  bvh::v2::ParallelExecutor executor(thread_pool);

  std::vector<BBox> bboxes(m_primitives.size());

  std::vector<Vec3> centers(m_primitives.size());

  executor.for_each(0, m_primitives.size(), [&](const std::size_t begin, const std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      bboxes[i] = m_primitives[i].get_bbox();
      centers[i] = m_primitives[i].get_center();
    }
  });

  typename bvh::v2::DefaultBuilder<Node>::Config config;

  config.quality = bvh::v2::DefaultBuilder<Node>::Quality::High;

  m_bvh = bvh::v2::DefaultBuilder<Node>::build(thread_pool, bboxes, centers, config);
}
