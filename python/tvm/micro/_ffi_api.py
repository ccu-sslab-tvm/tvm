""" Register FFI APIs from C++ for the namespace tvm.micro. """
import tvm._ffi


tvm._ffi._init_api("micro", __name__)
