from app.config import USE_TENSORRT

if USE_TENSORRT:
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    def build_engine(onnx_path, engine_path, workspace_size=(1 << 28), fp16=True):
        """
        Build a TensorRT engine from ONNX using the modern builder/config API.

        Writes a serialized engine to `engine_path`.
        """
        with trt.Builder(TRT_LOGGER) as builder:
            flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(flags)
            parser = trt.OnnxParser(network, TRT_LOGGER)

            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    raise RuntimeError("Failed to parse ONNX file")

            config = builder.create_builder_config()
            config.max_workspace_size = workspace_size
            if fp16 and hasattr(trt, 'BuilderFlag'):
                try:
                    config.set_flag(trt.BuilderFlag.FP16)
                except Exception:
                    pass

            # Build serialized network (preferred for newer TRT versions)
            if hasattr(builder, 'build_serialized_network'):
                serialized_engine = builder.build_serialized_network(network, config)
                if serialized_engine is None:
                    raise RuntimeError("Failed to build serialized engine")
                with open(engine_path, "wb") as f:
                    f.write(serialized_engine)
            else:
                # Fallback (older API)
                engine = builder.build_cuda_engine(network)
                if engine is None:
                    raise RuntimeError("Failed to build engine")
                with open(engine_path, "wb") as f:
                    f.write(engine.serialize())

            print("[OK] TensorRT engine built: {}".format(engine_path))

else:
    print("[INFO] TensorRT not available (not on Jetson)")
    
    def build_engine(*args, **kwargs):
        raise RuntimeError("TensorRT not available on this platform")
    build_engine("models/ae.onnx", "models/ae.engine")
