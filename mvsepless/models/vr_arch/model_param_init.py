import json

default_param = {}
default_param["bins"] = -1
default_param["unstable_bins"] = -1
default_param["stable_bins"] = -1
default_param["sr"] = 44100
default_param["pre_filter_start"] = -1
default_param["pre_filter_stop"] = -1
default_param["band"] = {}

N_BINS = "n_bins"


def int_keys(d):
    result_dict = {}
    for key, value in d:
        if key.isdigit():
            key = int(key)
        result_dict[key] = value
    return result_dict


class ModelParameters(object):

    def __init__(self, model_params=None):

        self.param = model_params

        for k in [
            "mid_side",
            "mid_side_b",
            "mid_side_b2",
            "stereo_w",
            "stereo_n",
            "reverse",
        ]:
            if not k in self.param:
                self.param[k] = False

        if N_BINS in self.param:
            self.param["bins"] = self.param[N_BINS]
