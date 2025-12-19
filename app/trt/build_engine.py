import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_path, engine_path):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        builder.max_workspace_size = 1 << 28
        builder.fp16_mode = True

        with open(onnx_path, "rb") as f:
            parser.parse(f.read())

        engine = builder.build_cuda_engine(network)

        with open(engine_path, "wb") as f:
            f.write(engine.serialize())

        print("[OK] TensorRT engine built")

if __name__ == "__main__":
    build_engine("models/xgb_model.onnx", "models/xgb_model.engine")
