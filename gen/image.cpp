#include "image.h"

#include "third_party/stb_image_write.h"

#include <algorithm>

namespace {

template<typename T>
auto
min(const bvh::v2::Vec<T, 3>& a, const bvh::v2::Vec<T, 3>& b) -> bvh::v2::Vec<T, 3>
{
  return { std::min(a[0], b[0]), std::min(a[1], b[1]), std::min(a[2], b[2]) };
}

template<typename T>
auto
max(const bvh::v2::Vec<T, 3>& a, const bvh::v2::Vec<T, 3>& b) -> bvh::v2::Vec<T, 3>
{
  return { std::max(a[0], b[0]), std::max(a[1], b[1]), std::max(a[2], b[2]) };
}

} // namespace

bool
savePng(const Image<bvh::v2::Vec<float, 3>>& image, const char* path)
{
  const int w = image.width();
  const int h = image.height();

  if ((w <= 0) || (h <= 0))
    return false;

  using Vec3 = bvh::v2::Vec<float, 3>;

  std::unique_ptr<unsigned char[]> data(new unsigned char[w * h * 3]);

  for (int i = 0; i < w * h; i++) {

    const auto c = image[i];

    const auto safe_c = min(max(c * 255.0f, Vec3(0, 0, 0)), Vec3(255, 255, 255));

    auto* pixel = data.get() + i * 3;

    pixel[0] = static_cast<unsigned char>(static_cast<int>(safe_c[0]));
    pixel[1] = static_cast<unsigned char>(static_cast<int>(safe_c[1]));
    pixel[2] = static_cast<unsigned char>(static_cast<int>(safe_c[2]));
  }

  return !!stbi_write_png(path, w, h, 3, data.get(), w * 3);
}

bool
savePng(const Image<unsigned char>& image, const char* path)
{
  const int w = image.width();
  const int h = image.height();

  if ((w <= 0) || (h <= 0))
    return false;

  return !!stbi_write_png(path, w, h, 1, &image[0], w);
}
