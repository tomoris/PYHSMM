from ctypes import *
lib = cdll.LoadLibrary("./awesome.so")
lib.Add.argtypes = [c_longlong, c_longlong]
print "awesome.Add(12,99) = %d" % lib.Add(12,99)
lib.Cosine.argtypes = [c_double]
lib.Cosine.restype = c_double
cos = lib.Cosine(1)
print "awesome.Cosine(1) = %f" % cos
class GoSlice(Structure):
    _fields_ = [("data", POINTER(c_void_p)),
                ("len", c_longlong), ("cap", c_longlong)]
nums = GoSlice((c_void_p * 5)(74, 4, 122, 9, 12), 5, 5)
lib.Sort.argtypes = [GoSlice]
lib.Sort.restype = None
lib.Sort(nums)
class GoString(Structure):
    _fields_ = [("p", c_char_p), ("n", c_longlong)]
lib.Log.argtypes = [GoString]
msg = GoString(b"Hello Python!", 13)
lib.Log(msg)
# See client.py on GitHub
