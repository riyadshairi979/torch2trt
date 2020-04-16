from torch2trt.torch2trt import *


@tensorrt_converter('torch.nn.ConvTranspose2d.forward')
def convert_ConvTranspose2d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    kernel_size = module.kernel_size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * 2

    stride = module.stride
    if not isinstance(stride, tuple):
        stride = (stride, ) * 2

    padding = module.padding
    if not isinstance(padding, tuple):
        padding = (padding, ) * 2

    kernel = module.weight.detach().cpu().numpy()

    bias = trt.Weights(torch_dtype_to_trt(module.weight.dtype))
    if module.bias is not None:
        bias = module.bias.detach().cpu().numpy()

    layer = ctx.network.add_deconvolution(
        input=input_trt,
        num_output_maps=module.out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias)
    layer.stride = stride
    layer.padding = padding

    # Cruise changes start here
    if hasattr(module, "output_padding") and module.output_padding != 0 and module.output_padding != (0, 0):    
        if (
            (module.padding != (1, 1) and module.padding != 1)
            or module.output_padding != (0, 1)
        ):
            raise Exception(
                "For ConvTranspose2d with output_padding only the following setting is supported. "
                "Padding has to be (1, 1) while output_padding has to be (0, 1). "
                "Normally this is used to upsample by an exact number. This logic may have further bugs, "
                "we should do something more comprehensive here and add more testing..."
            )

        layer.padding_mode = trt.PaddingMode.SAME_LOWER
    # Cruise changes end here

    if module.groups is not None:
        layer.num_groups = module.groups

    output._trt = layer.get_output(0)