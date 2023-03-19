#include "nn.h"

#include <iostream>

#include <cmath>
#include <cstdlib>

namespace {

void
linspace(const std::size_t n, float* output)
{
  for (std::size_t i = 0; i < n; i++)
    output[i] = static_cast<float>(i) / static_cast<float>(n);
}

template<typename Function>
void
generate(const std::size_t n, const float* input, float* output, Function function)
{
  for (std::size_t i = 0; i < n; i++)
    output[i] = function(input[i]);
};

} // namespace

int
main()
{
  const int N = 100;

  nn::NetworkBuilder builder;

  builder.addDense(N, N * 2);
  builder.addReLU();
  builder.addDense(N * 2, N);
  builder.addReLU();

  auto network = builder.build();

  auto* input = network.getInput();

  linspace(N, input);

  network.forwardPass();

  std::vector<float> expected(N);

  generate(N, input, expected.data(), [](const float x) -> float { return std::exp(-std::pow(x - 2, 2)); });

  nn::MeanSquaredError mse;

  const float loss = mse.eval(network.getOutput(), expected.data(), expected.size());

  std::cout << "Loss: " << loss << std::endl;

  return EXIT_SUCCESS;
}
