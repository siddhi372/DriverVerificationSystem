import onnx


MODEL_PATH = "models/MiniFASNetV2.onnx"


# Load ONNX model
model = onnx.load(MODEL_PATH)

print("========== MiniFASNetV2 ==========")

print("\nINPUT:")

for input_tensor in model.graph.input:

    print("Name:", input_tensor.name)

    shape = []

    for dimension in input_tensor.type.tensor_type.shape.dim:

        if dimension.dim_value:
            shape.append(dimension.dim_value)
        else:
            shape.append("dynamic")

    print("Shape:", shape)


print("\nOUTPUT:")

for output_tensor in model.graph.output:

    print("Name:", output_tensor.name)

    shape = []

    for dimension in output_tensor.type.tensor_type.shape.dim:

        if dimension.dim_value:
            shape.append(dimension.dim_value)
        else:
            shape.append("dynamic")

    print("Shape:", shape)


print("\n===================================")