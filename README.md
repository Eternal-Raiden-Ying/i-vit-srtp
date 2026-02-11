we finish vit int8 quantization (partial) deployment on NVIDIA GPU by tensorRT

to run this
1. use quant_train.py to get a quant model weight based on fp pretrained model (we use timm, you can use Google ViT as well, remember to change the transform style to fit the pretrained model)
2. run export_onnx.py to get the .onnx file, we need this to generate tensorRT engine
3. run 'onnxsim input.onnx output.onnx' in cmd to get a simplied onnx (pip install onnx first)
4. use build_trt_engine.py to build the final engine (whole tensorRT is not a must, python package for tensorRT is enough)
5. use compare.py to see the acc loss and speed up gain.
