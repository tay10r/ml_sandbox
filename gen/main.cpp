#include "color_generator.h"
#include "image.h"
#include "renderer.h"
#include "scene.h"

#include <glm/gtx/transform.hpp>

#include <bvh/v2/vec.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include <cmath>
#include <cstdlib>

namespace {

std::string
createDataPath(const char* folderPath, const char* name, const int i, const char* ext)
{
  std::ostringstream imagePathStream;
  imagePathStream << folderPath << '/';
  imagePathStream << std::setw(5) << std::setfill('0') << i;
  imagePathStream << '_';
  imagePathStream << name;
  imagePathStream << ext;
  return imagePathStream.str();
}

struct ModelInfo final
{
  std::string path;

  bvh::v2::Vec<float, 3> albedo;

  bvh::v2::Vec<float, 3> emission;
};

class Program final
{
public:
  using Vec3 = bvh::v2::Vec<float, 3>;

  Program(const int w, const int h, const int seed)
    : m_colorGenerator(seed)
    , m_renderer(w, h, seed)
    , m_rng(seed)
  {
    loadModels();

    if (!std::filesystem::exists("train"))
      std::filesystem::create_directory("train");

    if (!std::filesystem::exists("test"))
      std::filesystem::create_directory("test");
  }

  void run()
  {
    for (std::size_t i = 0; i < 80; i++) {
      std::uniform_int_distribution<std::size_t> d(m_objectModelOffset, m_objectModelOffset + m_objectModelCount - 1);
      const auto objectIndex{ d(m_rng) };
      runSimulation(objectIndex, "train");
      std::cout << "Generated training set " << i << " of 80." << std::endl;
    }

    for (std::size_t i = 0; i < 20; i++) {
      std::uniform_int_distribution<std::size_t> d(m_objectModelOffset, m_objectModelOffset + m_objectModelCount - 1);
      const auto objectIndex{ d(m_rng) };
      runSimulation(objectIndex, "test");
      std::cout << "Generated test set " << i << " of 20." << std::endl;
    }
  }

protected:
  void runSimulation(const std::size_t objectIndex, const char* folderPath)
  {
    // Conservation of energy:
    //       mgh = (1/2)mv^2
    //        gh = (1/2)v^2
    //       2gh = v^2
    // sqrt(2gh) = v
    //
    // (1/2)gt^2 + vt + x0 = x
    //
    // Total time (quadratic equation) = 3.499

    std::uniform_int_distribution<int> skyDist(0, 2);

    switch (skyDist(m_rng)) {
      case 0:
        m_renderer.setSkyColors(0xffffff, 0x7fcfff);
        break;
      case 1:
        m_renderer.setSkyColors(0xe15b00, 0x7a96bc);
        break;
      case 2:
        m_renderer.setSkyColors(0x182c6b, 0x010216);
        break;
    }

    const auto albedo = m_colorGenerator.generate();

    std::uniform_real_distribution<float> camXDist(-40, -25);
    std::uniform_real_distribution<float> camYDist(4, 6);
    std::uniform_real_distribution<float> camZDist(-2, 2);

    const Vec3 cameraPos(camXDist(m_rng), camYDist(m_rng), camZDist(m_rng));

    const float totalTime = 3.12984f;

    const float initialVelocity = 15.336f;

    const float angularVelocity = 360.0f * 3.0f / totalTime;

    const float g = -9.8;

    const float dt = 1.0f / 15.0f;

    const int totalSteps = static_cast<int>(totalTime / dt);

    float velocity = initialVelocity;

    float position = 0;

    float angle = 0;

    for (int i = 0; i < totalSteps; i++) {

      m_scene.clear();

      m_scene.instanceRange(m_staticModelOffset, m_staticModelCount);

      angle += angularVelocity * dt;

      position += velocity * dt + 0.5 * g * dt * dt;

      velocity += g * dt;

      m_scene.instanceSingle(objectIndex,
                             glm::translate(glm::vec3(0.0f, position, 0.0f)) *
                               glm::rotate(glm::radians(angle), glm::vec3(0, 1, 0)),
                             albedo,
                             true);

      m_scene.commit();

      const auto render_result{ m_renderer.render(m_scene, cameraPos) };

      savePng(render_result.noisy_color, createDataPath(folderPath, "noisy", m_stepIndex, ".png").c_str());
      savePng(render_result.color, createDataPath(folderPath, "color", m_stepIndex, ".png").c_str());
      savePng(render_result.albedo, createDataPath(folderPath, "albedo", m_stepIndex, ".png").c_str());
      savePng(render_result.normal, createDataPath(folderPath, "normal", m_stepIndex, ".png").c_str());
      savePng(render_result.depth, createDataPath(folderPath, "depth", m_stepIndex, ".png").c_str());
      savePng(render_result.segmentation, createDataPath(folderPath, "segmentation", m_stepIndex, ".png").c_str());
      savePng(render_result.stencil, createDataPath(folderPath, "stencil", m_stepIndex, ".png").c_str());

      saveYolo(render_result.stencil, objectIndex, folderPath);

      m_stepIndex++;
    }
  }

  void saveYolo(const Image<unsigned char>& stencil, const std::size_t objectIndex, const char* folderPath)
  {
    int xMin = stencil.width() + 1;
    int xMax = -1;
    int yMin = stencil.height() + 1;
    int yMax = -1;

    for (int y = 0; y < stencil.height(); y++) {
      for (int x = 0; x < stencil.width(); x++) {
        if (!stencil[y * stencil.width() + x])
          continue;
        xMin = std::min(xMin, x);
        xMax = std::max(xMax, x);
        yMin = std::min(yMin, y);
        yMax = std::max(yMax, y);
      }
    }

    if ((xMax < 0) || (yMax < 0))
      return;

    const auto path = createDataPath(folderPath, "annotation", m_stepIndex, ".txt");

    const int w = (xMax - xMin) + 1;
    const int h = (yMax - yMin) + 1;

    std::ofstream file(path.c_str());

    file << objectIndex << ' ' << xMin << ' ' << yMin << ' ' << w << ' ' << h << std::endl;
  }

  void loadModels()
  {
    std::vector<ModelInfo> models;

    models.emplace_back(ModelInfo{ MODEL_PATH "/room.stl", { 1, 1, 1 }, { 0, 0, 0 } });
    models.emplace_back(ModelInfo{ MODEL_PATH "/ejection_tunnel.stl", { 0, 1, 0 }, { 0, 0, 0 } });
    models.emplace_back(ModelInfo{ MODEL_PATH "/big_sphere.stl", m_colorGenerator.generate(), { 0, 0, 0 } });
    models.emplace_back(ModelInfo{ MODEL_PATH "/little_sphere.stl", m_colorGenerator.generate(), { 0, 0, 0 } });
    models.emplace_back(ModelInfo{ MODEL_PATH "/cone.stl", m_colorGenerator.generate(), { 0, 0, 0 } });
    models.emplace_back(ModelInfo{ MODEL_PATH "/left_shelf.stl", { 0.787, 0.129, 0 }, { 0, 0, 0 } });
    models.emplace_back(ModelInfo{ MODEL_PATH "/big_cube.stl", m_colorGenerator.generate(), { 0, 0, 0 } });
    models.emplace_back(ModelInfo{ MODEL_PATH "/little_cube.stl", m_colorGenerator.generate(), { 0, 0, 0 } });
    models.emplace_back(ModelInfo{ MODEL_PATH "/right_shelf.stl", { 0.787, 0.129, 0 }, { 0, 0, 0 } });
    models.emplace_back(ModelInfo{ MODEL_PATH "/torus.stl", m_colorGenerator.generate(), { 0, 0, 0 } });

    m_staticModelOffset = 0;

    for (const auto& m : models) {

      if (m_scene.loadModel(m.path.c_str(), m.albedo, m.emission, m_colorGenerator.generate()))
        std::cout << "Loaded '" << m.path << "'." << std::endl;
    }

    m_staticModelCount = m_scene.modelCount() - m_staticModelOffset;

    m_objectModelOffset = m_scene.modelCount();

    m_scene.loadModel(MODEL_PATH "/buddha.stl", m_colorGenerator.generate(), { 0, 0, 0 }, m_colorGenerator.generate());
    m_scene.loadModel(MODEL_PATH "/bunny.stl", m_colorGenerator.generate(), { 0, 0, 0 }, m_colorGenerator.generate());
    m_scene.loadModel(MODEL_PATH "/dragon.stl", m_colorGenerator.generate(), { 0, 0, 0 }, m_colorGenerator.generate());
    m_scene.loadModel(MODEL_PATH "/monkey.stl", m_colorGenerator.generate(), { 0, 0, 0 }, m_colorGenerator.generate());
    m_scene.loadModel(MODEL_PATH "/teapot.stl", m_colorGenerator.generate(), { 0, 0, 0 }, m_colorGenerator.generate());

    m_objectModelCount = m_scene.modelCount() - m_objectModelOffset;
  }

private:
  ColorGenerator m_colorGenerator;

  Renderer m_renderer;

  std::mt19937 m_rng;

  Scene m_scene;

  std::size_t m_staticModelOffset{ 0 };

  std::size_t m_staticModelCount{ 0 };

  std::size_t m_objectModelOffset{ 0 };

  std::size_t m_objectModelCount{ 0 };

  int m_stepIndex{ 0 };
};

} // namespace

int
main()
{
  Program program(256, 256, 1234);

  program.run();

  std::cout << "Done." << std::endl;

  return EXIT_SUCCESS;
}
