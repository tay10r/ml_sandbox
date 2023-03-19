#pragma once

#include <bvh/v2/vec.h>

#include <memory>

template<typename T>
class Image final
{
public:
  Image(const int w, const int h)
    : m_color(new T[w * h]{})
    , m_width(w)
    , m_height(h)
  {
  }

  int width() const { return m_width; }

  int height() const { return m_height; }

  T& operator[](const int index) { return m_color.get()[index]; }

  const T& operator[](const int index) const { return m_color.get()[index]; }

private:
  std::unique_ptr<T[]> m_color;
  const int m_width;
  const int m_height;
};

bool
savePng(const Image<bvh::v2::Vec<float, 3>>& image, const char* path);

bool
savePng(const Image<unsigned char>& image, const char* path);
