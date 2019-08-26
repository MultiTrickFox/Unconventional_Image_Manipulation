from torch import tensor, Tensor
from torch import pow, sum
from torch import clamp
from torch import no_grad

from torchvision.models import vgg19

from numpy import float32, uint8
from PIL import Image


input_file_1  = 'data.jpg'
input_file_2  = 'data2.jpg'
output_file   = 'data_styled.jpg'

hm_iterations = 20
learning_rate = 1e-3

importance_content = 1
importance_style   = .7

width_in  = 229
height_in = 229


model = vgg19(pretrained=True).eval()

image1 = float32(Image.open(input_file_1)).reshape(1, 3, width_in, height_in)
image2 = float32(Image.open(input_file_2)).reshape(1, 3, width_in, height_in)

input       = tensor(image1, requires_grad=True)
out_content = model(Tensor(image1))
out_style   = model(Tensor(image2))


for iteration in range(hm_iterations):

    output = model(input)

    gram_output = output @ output.t()
    gram_style = out_style @ out_style.t()

    loss_content = sum(pow(output - out_content, 2))
    loss_style = sum(pow(gram_output - gram_style, 2))
    loss = sum(importance_style * loss_content + importance_style * loss_style)

    loss.backward(retain_graph=True)
    with no_grad():
        input -= input.grad * learning_rate
        input.grad = None

    input = clamp(input, min=0, max=255).detach().requires_grad_(True)

    print(f'iteration: {iteration}, loss: {float(loss)}')


with open(output_file, 'wb+') as file:
    Image.fromarray(uint8(input.detach().numpy()).reshape(width_in, height_in, 3)).save(file, 'jpeg')