from torchvision.models import vgg16

from PIL import Image
from numpy import float32, uint8

from torch import tensor, zeros, zeros_like
from torch import argmax, sum, pow
from torch import flatten, reshape, clamp, squeeze, unsqueeze
from torch import no_grad



input_file    = "data.jpg"
output_file   = "data_dreamt.jpg"
dream_layer   = 30
hm_iterations = 100



model = vgg16(pretrained=True).eval()
image = float32(Image.open(input_file)).reshape(1, 3, 229, 229)
input = tensor(image, requires_grad=True)

print(f'selected dream layer: {model.features[dream_layer]}')


for iteration in range(hm_iterations):

    outputs = [input]
    for layer in model.features[:dream_layer]:
        outputs.append(layer(outputs[-1]))

    output = outputs[-1]
    flat_output = flatten(output)
    flat_label = zeros_like(flat_output)
    flat_label[argmax(flat_output)] += 1
    label = reshape(flat_label, output.size())
    loss = sum(pow(output - label, 2))

    loss.backward()
    with no_grad():
        input -= input.grad * 1e-5
        input.grad = None

    input = clamp(input, min=0, max=255).detach().requires_grad_(True)

    print(iteration, float(loss))


with open(output_file, 'wb+') as file:
    Image.fromarray(uint8(input.detach().numpy()).reshape(229, 229, 3)).save(file, 'jpeg')










