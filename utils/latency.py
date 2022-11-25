import torch 
import time 
import onnx 
import numpy as np 



def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def compute_latency_ms_pytorch(model, input_size, iterations=None, device=None):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model.eval()
    model = model.cuda()
    input = torch.randn(*input_size).cuda()

    with torch.no_grad():
        for _ in range(10):
            model(input)
        
        if iterations is None:
            elapsed_time = 0 
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize() 
                elapsed_time =  time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = max(iterations, int(FPS * 6))

        # testing speed 
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000 
    torch.cuda.empty_cache()
    # FPS = 1000 / latency (in ms)
    return latency


# ---------------------------------------------------------------
# TensorRT Latency 
# ---------------------------------------------------------------
try:
    import pycuda.driver as cuda 
    import pycuda.autoinit 
    import tensorrt as trt 

    def export_onnx(net, input_size, onnx_file='network.onnx', opset_version=10):
        net = net.eval()
        dummy_input = torch.randn(input_size).cuda()
        torch.onnx.export(
            net, dummy_input, onnx_file, export_params=True, 
            verbose=False, do_constant_folding=True, opset_version=opset_version,
            input_names=['input'], output_names=['output'])
        onnx_model = onnx.load(onnx_file)
        onnx.checker.check_model(onnx_model)
        return onnx_file


    class HostDeviceMem:

        def __init__(self, host_mem, device_mem):
            self.host = host_mem 
            self.device = device_mem 

        def __str__(self):
            return 'Host: \n' + str(self.host) + '\nDevice:\n' + str(self.device)
        
        def __repr__(self):
            return self.__str__() 


    def GiB(val):
        return val * 1 << 30 


    def build_engine_onnx(model_file, logger=trt.Logger(trt.Logger.INFO), mode='FP32'):
        EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(logger) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, logger) as parser:
            builder.max_workspace_size = GiB(4)
            builder.fp16_mode = True if mode == 'FP16' else False 
            with open(model_file, 'rb') as model:
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the onnx file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None 
            return builder.build_cuda_engine(network)


    def allocate_buffers(engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size 
            dtype = trt.nptype(engine.get_binding_dtype(binding)) 
            # Allocate host and device buffers. 
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings. 
            bindings.append(int(device_mem))
            # Append to the appropriate list. 
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream 


    def copy_to_host(test_image, pagelocked_buffer): 
        def normalize_image(image): 
            return image.ravel() 
        np.copyto(pagelocked_buffer, normalize_image(test_image)) 


    def do_inference_v2(context, bindings, inputs, outputs, stream, num_iters=1000):
        # Transfer input data to the GPU device 
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        stream.synchronize()
        # warmup
        for i in range(10):
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()
        # Run inference
        t_start = time.time()
        for i in range(num_iters):
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        stream.synchronize()
        elapsed_time = time.time() - t_start
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        return [out.host for out in outputs], elapsed_time 


    def compute_latency_ms_tensorrt(onnx_file, input_size, num_iters=1000, verbose=False, mode='FP32'):
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
        with build_engine_onnx(onnx_file, TRT_LOGGER, mode) as engine:
            inputs, outputs, bindings, stream = allocate_buffers(engine)
            with engine.create_execution_context() as context:
                test_image = np.random.randn(*input_size)
                copy_to_host(test_image, inputs[0].host)
                _, elapsed_time = do_inference_v2(
                    context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, num_iters=num_iters) 
                latency = elapsed_time * 1000 / num_iters  # ms  
                return latency 

except:
    pass 


def get_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)  