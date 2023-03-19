#pragma once

#include <glm/glm.hpp>

#include <bvh/v2/bvh.h>
#include <bvh/v2/node.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/tri.h>

#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <vector>

struct Model final
{
  using Vec3 = bvh::v2::Vec<float, 3>;

  using Tri = bvh::v2::Tri<float, 3>;

  std::vector<Tri> primitives;

  std::vector<Vec3> normals;

  Vec3 albedo;

  Vec3 emission;

  Vec3 segmentation;
};

class Scene final
{
public:
  using Vec3 = bvh::v2::Vec<float, 3>;

  using Tri = bvh::v2::PrecomputedTri<float>;

  using Node = bvh::v2::Node<float, 3>;

  using Bvh = bvh::v2::Bvh<Node>;

  using Ray = bvh::v2::Ray<float, 3>;

  struct Attributes final
  {
    Vec3 normal;

    Vec3 albedo;

    Vec3 emission;

    Vec3 segmentation;

    bool objectMask;
  };

  struct Hit final
  {
    Vec3 normal;

    Vec3 albedo;

    Vec3 emission;

    Vec3 segmentation;

    bool objectMask;
  };

  bool loadModel(const char* path, const Vec3& albedo, const Vec3& emission, const Vec3& segmentation);

  void instanceRange(std::size_t offset,
                     std::size_t count,
                     const glm::mat4& transform = glm::mat4(1.0f),
                     const std::optional<Vec3>& albedoOverride = std::nullopt,
                     bool objectMask = false);

  void instanceSingle(std::size_t index,
                      const glm::mat4& transform,
                      const std::optional<Vec3>& albedoOverride,
                      bool objectMask)
  {
    instanceRange(index, 1, transform, albedoOverride, objectMask);
  }

  void randomize(int instance_count, int seed);

  void commit();

  void clear()
  {
    m_primitives.clear();

    m_attributes.clear();
  }

  std::size_t primitiveCount() const { return m_primitives.size(); }

  std::optional<Hit> intersect(Ray& ray) const
  {
    constexpr std::size_t stack_size{ 64 };

    bvh::v2::SmallStack<Bvh::Index, stack_size> stack;

    constexpr auto use_robust_traversal{ false };

    constexpr auto invalid_id = std::numeric_limits<std::size_t>::max();

    auto primitive_id = invalid_id;

    m_bvh.intersect<false, use_robust_traversal>(
      ray, m_bvh.get_root().index, stack, [&](const std::size_t begin, const std::size_t end) {
        auto hit_flag{ false };
        for (std::size_t i = begin; i < end; i++) {
          const std::size_t j = m_bvh.prim_ids[i];
          if (auto hit = m_primitives[j].intersect(ray)) {
            primitive_id = j;
            hit_flag = true;
          }
        }
        return hit_flag;
      });

    if (primitive_id == invalid_id)
      return std::nullopt;

    const auto& attrib = m_attributes[primitive_id];

    // flip normal if needed. don't feel like checking the model normals

    auto normal = (dot(ray.dir, attrib.normal) < 0.0f) ? attrib.normal : -attrib.normal;

    return Hit{ normal, attrib.albedo, attrib.emission, attrib.segmentation, attrib.objectMask };
  }

  std::size_t modelCount() const { return m_models.size(); }

protected:
  void instance(const Model& model,
                const glm::mat4& transform,
                const std::optional<Vec3>& albedoOverride,
                bool objectMask);

private:
  std::vector<Tri> m_primitives;

  std::vector<Attributes> m_attributes;

  Bvh m_bvh;

  std::vector<Model> m_models;
};
