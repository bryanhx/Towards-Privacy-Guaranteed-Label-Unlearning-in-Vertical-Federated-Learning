from .LUV_modelnet import LUV_modelnet


def get_unlearn_method(method):
    if method == "LUV_modelnet":
        return LUV_modelnet