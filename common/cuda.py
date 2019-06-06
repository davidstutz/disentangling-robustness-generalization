import torch
import numpy


def memory():
    """
    Get memory usage.

    :return: memory usage
    :rtype: str
    """

    index = torch.cuda.current_device()

    # results are in bytes
    # Decimal
    # Value 	Metric
    # 1000 	kB 	kilobyte
    # 1000^2 	MB 	megabyte
    # 1000^3 	GB 	gigabyte
    # 1000^4 	TB 	terabyte
    # 1000^5 	PB 	petabyte
    # 1000^6 	EB 	exabyte
    # 1000^7 	ZB 	zettabyte
    # 1000^8 	YB 	yottabyte
    # Binary
    # Value 	IEC 	JEDEC
    # 1024 	KiB 	kibibyte 	KB 	kilobyte
    # 1024^2 	MiB 	mebibyte 	MB 	megabyte
    # 1024^3 	GiB 	gibibyte 	GB 	gigabyte
    # 1024^4 	TiB 	tebibyte 	-
    # 1024^5 	PiB 	pebibyte 	-
    # 1024^6 	EiB 	exbibyte 	-
    # 1024^7 	ZiB 	zebibyte 	-
    # 1024^8 	YiB 	yobibyte 	-
    allocated = torch.cuda.max_memory_allocated(index)
    cached = torch.cuda.max_memory_cached(index)/(1024*1024)

    return '%gMiB' % BMiB(allocated + cached)


def is_cuda(mixed):
    """
    Check if model/tensor is on CUDA.

    :param mixed: model or tensor
    :type mixed: torch.nn.Module or torch.autograd.Variable or torch.Tensor
    :return: on cuda
    :rtype: bool
    """

    assert isinstance(mixed, torch.nn.Module) or isinstance(mixed, torch.autograd.Variable) \
        or isinstance(mixed, torch.Tensor), 'mixed has to be torch.nn.Module, torch.autograd.Variable or torch.Tensor'

    is_cuda = False
    if isinstance(mixed, torch.nn.Module):
        is_cuda = True
        for parameters in list(mixed.parameters()):
            is_cuda = is_cuda and parameters.is_cuda
    if isinstance(mixed, torch.autograd.Variable):
        is_cuda = mixed.is_cuda
    if isinstance(mixed, torch.Tensor):
        is_cuda = mixed.is_cuda

    return is_cuda


def estimate_size(mixed):
    """
    Estimate tensor size.

    :param tensor: tensor or model
    :type tensor: numpy.ndarray, torch.tensor, torch.autograd.Variable or torch.nn.Module
    :return: size in bits
    :rtype: int
    """

    # PyTorch types:
    # Data type 	dtype 	CPU tensor 	GPU tensor
    # 32-bit floating point 	torch.float32 or torch.float 	torch.FloatTensor 	torch.cuda.FloatTensor
    # 64-bit floating point 	torch.float64 or torch.double 	torch.DoubleTensor 	torch.cuda.DoubleTensor
    # 16-bit floating point 	torch.float16 or torch.half 	torch.HalfTensor 	torch.cuda.HalfTensor
    # 8-bit integer (unsigned) 	torch.uint8 	torch.ByteTensor 	torch.cuda.ByteTensor
    # 8-bit integer (signed) 	torch.int8 	torch.CharTensor 	torch.cuda.CharTensor
    # 16-bit integer (signed) 	torch.int16 or torch.short 	torch.ShortTensor 	torch.cuda.ShortTensor
    # 32-bit integer (signed) 	torch.int32 or torch.int 	torch.IntTensor 	torch.cuda.IntTensor
    # 64-bit integer (signed) 	torch.int64 or torch.long 	torch.LongTensor 	torch.cuda.LongTensor

    # Numpy types:
    # Data type 	Description
    # bool_ 	Boolean (True or False) stored as a byte
    # int_ 	Default integer type (same as C long; normally either int64 or int32)
    # intc 	Identical to C int (normally int32 or int64)
    # intp 	Integer used for indexing (same as C ssize_t; normally either int32 or int64)
    # int8 	Byte (-128 to 127)
    # int16 	Integer (-32768 to 32767)
    # int32 	Integer (-2147483648 to 2147483647)
    # int64 	Integer (-9223372036854775808 to 9223372036854775807)
    # uint8 	Unsigned integer (0 to 255)
    # uint16 	Unsigned integer (0 to 65535)
    # uint32 	Unsigned integer (0 to 4294967295)
    # uint64 	Unsigned integer (0 to 18446744073709551615)
    # float_ 	Shorthand for float64.
    # float16 	Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
    # float32 	Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
    # float64 	Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
    # complex_ 	Shorthand for complex128.
    # complex64 	Complex number, represented by two 32-bit floats (real and imaginary components)
    # complex128 	Complex number, represented by two 64-bit floats (real and imaginary components)

    types8 = [
        torch.uint8, torch.int8,
        numpy.int8, numpy.uint8, numpy.bool_,
    ]

    types16 = [
        torch.float16, torch.half,
        torch.int16, torch.short,
        numpy.int16, numpy.uint16, numpy.float16,
    ]

    types32 = [
        torch.float32, torch.float,
        torch.int32, torch.int,
        numpy.int32, numpy.uint32, numpy.float32,
    ]

    types64 = [
        torch.float64, torch.double,
        torch.int64, torch.long,
        numpy.int64, numpy.uint64, numpy.float64, numpy.complex64,
        numpy.int_, numpy.float_
    ]

    types128 = [
        numpy.complex_, numpy.complex128
    ]

    if isinstance(mixed, torch.nn.Module):

        size = 0
        modules = mixed.modules()
        for module in modules:
            for parameters in list(module.parameters()):
                size += estimate_size(parameters)
        return size

    if isinstance(mixed, (torch.Tensor, numpy.ndarray)):

        if mixed.dtype in types128:
            bits = 128
        elif mixed.dtype in types64:
            bits = 64
        elif mixed.dtype in types32:
            bits = 32
        elif mixed.dtype in types16:
            bits = 16
        elif mixed.dtype in types8:
            bits = 8
        else:
            assert False, 'could not identify torch.Tensor or numpy.ndarray type %s' % mixed.type()

        size = numpy.prod(mixed.shape)
        return size*bits

    elif isinstance(mixed, torch.autograd.Variable):
        return estimate_size(mixed.data)
    else:
        assert False, 'unsupported tensor size for estimating size, either numpy.ndarray, torch.tensor or torch.autograd.Variable'


def bits2MiB(bits):
    """
    Convert bits to MiB.

    :param bits: number of bits
    :type bits: int
    :return: MiB
    :rtype: float
    """

    return bits/(8*1024*1024)


def bits2MB(bits):
    """
    Convert bits to MB.

    :param bits: number of bits
    :type bits: int
    :return: MiB
    :rtype: float
    """

    return bits/(8*1000*1000)


def bytes2MiB(bytes):
    """
    Convert bytes to MiB.

    :param bytes: number of bytes
    :type bytes: int
    :return: MiB
    :rtype: float
    """

    return bytes/(1024*1024)


def bytes2MB(bytes):
    """
    Convert bytes to MB.

    :param bytes: number of bytes
    :type bytes: int
    :return: MiB
    :rtype: float
    """

    return bytes/(1000*1000)


bMiB = bits2MiB
BMiB = bytes2MiB
bMB = bits2MB
BMB = bytes2MB