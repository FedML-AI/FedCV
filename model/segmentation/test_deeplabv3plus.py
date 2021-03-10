from ptflops import get_model_complexity_info

from model.segmentation.deeplabV3_plus import DeepLabV3_plus

model = DeepLabV3_plus(pretrained=True)

flops, params = get_model_complexity_info(model, (64, 513, 513), verbose=True)

print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
