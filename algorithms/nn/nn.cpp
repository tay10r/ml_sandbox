#include "nn.h"

#include <cmath>

namespace nn {

Dense::Dense(const std::size_t inputCount, const std::size_t outputCount)
  : LayerBase<Dense>(inputCount, outputCount)
  , m_weights(inputCount * outputCount)
{
}

void
Dense::forwardPass(const float* input, float* output) const
{
  for (std::size_t i = 0; i < outputCount(); i++) {

    auto sum{ 0.0f };

    for (std::size_t j = 0; j < inputCount(); j++) {

      const auto wOffset = i * inputCount();

      sum += m_weights[wOffset + j] * input[j];
    }

    output[i] = sum;
  }
}

ReLU::ReLU(const std::size_t size)
  : LayerBase<ReLU>(size, size)
{
}

void
ReLU::forwardPass(const float* input, float* output) const
{
  for (std::size_t i = 0; i < inputCount(); i++)
    output[i] = (input[i] < 0) ? 0 : input[i];
}

float
MeanSquaredError::eval(const float* actual, const float* expected, std::size_t size) const
{
  float sum{ 0 };

  for (std::size_t i = 0; i < size; i++) {
    const float delta = actual[i] - expected[i];
    sum += delta * delta;
  }

  using namespace std;

  return sqrt(sum / static_cast<float>(size));
}

Network::Network(std::vector<LayerPtr> layers)
  : m_layers(std::move(layers))
{
  if (m_layers.size() > 0) {

    m_buffers.emplace_back(m_layers[0]->inputCount());

    for (std::size_t i = 0; i < m_layers.size(); i++)
      m_buffers.emplace_back(m_layers[i]->outputCount());
  }
}

std::size_t
Network::inputCount() const
{
  return m_layers.empty() ? 0 : m_layers[0]->inputCount();
}

std::size_t
Network::outputCount() const
{
  return m_layers.empty() ? 0 : m_layers.back()->outputCount();
}

float*
Network::getInput()
{
  return m_buffers.empty() ? nullptr : m_buffers[0].data();
}

const float*
Network::getOutput() const
{
  return m_buffers.empty() ? nullptr : m_buffers.back().data();
}

void
Network::forwardPass()
{
  for (std::size_t i = 0; i < m_layers.size(); i++)
    m_layers[i]->forwardPass(m_buffers[i].data(), m_buffers[i + 1].data());
}

void
NetworkBuilder::addDense(const std::size_t inputs, const std::size_t outputs)
{
  m_layers.emplace_back(new Dense(inputs, outputs));
}

void
NetworkBuilder::addReLU()
{
  const auto size = m_layers.at(m_layers.size() - 1)->outputCount();

  m_layers.emplace_back(new ReLU(size));
}

Network
NetworkBuilder::build()
{
  return Network{ std::move(m_layers) };
}

} // namespace nn
