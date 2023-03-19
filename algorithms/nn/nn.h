#pragma once

#include <memory>
#include <vector>

#include <cstddef>

namespace nn {

/// <summary>
/// The base class of a neural network layer.
/// </summary>
class Layer
{
public:
  Layer() = default;

  Layer(const Layer&) = default;

  Layer(Layer&&) = default;

  Layer& operator=(const Layer&) = default;

  Layer& operator=(Layer&&) = default;

  virtual ~Layer() = default;

  virtual void forwardPass(const float* input, float* output) const = 0;

  virtual std::size_t inputCount() const noexcept = 0;

  virtual std::size_t outputCount() const noexcept = 0;
};

/// <summary>
/// A base class for layers.
/// </summary>
template<typename Derived>
class LayerBase : public Layer
{
public:
  /// <summary>
  /// Constructs a new layer base, allocating the weights used for this layer.
  /// </summary>
  /// <param name="inputCount">
  /// The number of inputs for this layer.
  /// </param>
  /// <param name="outputCount">
  /// The number of outputs for this layer.
  /// </param>
  LayerBase(std::size_t inputCount, std::size_t outputCount)
    : m_inputCount(inputCount)
    , m_outputCount(outputCount)
  {
  }

  LayerBase(const LayerBase&) = default;

  LayerBase(LayerBase&&) = default;

  LayerBase& operator=(const LayerBase&) = default;

  LayerBase& operator=(LayerBase&&) = default;

  std::size_t inputCount() const noexcept override { return m_inputCount; }

  std::size_t outputCount() const noexcept override { return m_outputCount; }

private:
  std::size_t m_inputCount;

  std::size_t m_outputCount;
};

class Dense final : public LayerBase<Dense>
{
public:
  Dense(std::size_t inputCount, std::size_t outputCount);

  void forwardPass(const float* input, float* output) const override;

private:
  /// <summary>
  /// The weights associated with each unique pairing of input and output.
  /// </summary>
  std::vector<float> m_weights;
};

class ReLU final : public LayerBase<ReLU>
{
public:
  ReLU(std::size_t size);

  using LayerBase<ReLU>::LayerBase;

  void forwardPass(const float* input, float* output) const override;
};

class Sigmoid final : public LayerBase<Sigmoid>
{
public:
  using LayerBase<Sigmoid>::LayerBase;
};

class Softmax final : public LayerBase<Softmax>
{
public:
  using LayerBase<Softmax>::LayerBase;
};

class Loss
{
public:
  Loss() = default;

  Loss(const Loss&) = default;

  Loss(Loss&&) = default;

  Loss& operator=(const Loss&) = default;

  Loss& operator=(Loss&&) = default;

  virtual ~Loss() = default;

  virtual float eval(const float* actual, const float* expected, std::size_t size) const = 0;
};

class MeanSquaredError final : public Loss
{
public:
  float eval(const float* actual, const float* expected, std::size_t size) const override;
};

class NetworkBuilder;

class Network final
{
public:
  using LayerPtr = std::unique_ptr<Layer>;

  explicit Network(std::vector<LayerPtr> layers);

  std::size_t inputCount() const;

  std::size_t outputCount() const;

  void forwardPass();

  float* getInput();

  const float* getOutput() const;

private:
  std::vector<LayerPtr> m_layers;

  std::vector<std::vector<float>> m_buffers;
};

class NetworkBuilder final
{
public:
  using LayerPtr = Network::LayerPtr;

  void addDense(std::size_t inputs, std::size_t outputs);

  void addReLU();

  void addSigmoid();

  void addSoftmax();

  Network build();

private:
  std::vector<LayerPtr> m_layers;
};

} // namespace nn
