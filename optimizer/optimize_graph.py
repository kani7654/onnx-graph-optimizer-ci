import onnx
import onnxoptimizer

model = onnx.load("model.onnx")

passes = [
    "eliminate_deadend",
    "eliminate_identity",
    "eliminate_nop_dropout",
    "eliminate_unused_initializer"
]

optimized_model = onnxoptimizer.optimize(model, passes)

onnx.checker.check_model(optimized_model)
onnx.save(optimized_model, "optimized_model.onnx")

print("ONNX graph optimized and validated successfully")
