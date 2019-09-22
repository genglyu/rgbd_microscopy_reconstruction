import json
import numpy
import sys

sys.path.append("../../Utility")
sys.path.append("../Alignment")
from file_managing import *
# from local_transformation_estimation import *


class LocalTransformationEstimationResult:
    def __init__(self, s, t, success, conf, trans):
        self.s = s
        self.t = t
        self.success = success
        self.conf = conf
        self.trans = trans


class TransformationDataPool:
    def __init__(self, tile_info_dict, config):
        self.trans_dict = {}
        self.tile_info_dict = tile_info_dict
        self.config = config

    def save(self, path=None):
        if path is None:
            path = join(self.config["path_data"], self.config["local_trans_dict_g2o"])
        data_to_save = {}
        for (s, t) in self.trans_dict:
            if s not in data_to_save:
                data_to_save[s] = {}
            data_to_save[s][t] = {"s": self.trans_dict[(s, t)].s,
                                  "t": self.trans_dict[(s, t)].t,
                                  "success": self.trans_dict[(s, t)].success,
                                  "conf": self.trans_dict[(s, t)].conf,
                                  "trans": self.trans_dict[(s, t)].trans.tolist()}
        json.dump(data_to_save, open(path, "w"), indent=4)

    def read(self, path=None):
        if path is None:
            path = join(self.config["path_data"], self.config["local_trans_dict_g2o"])
        readed_data = json.load(open(path, "r"))
        for s in readed_data:
            for t in readed_data[s]:
                self.update_trans_pure(s_id=int(readed_data[s][t]["s"]),
                                       t_id=int(readed_data[s][t]["t"]),
                                       success=bool(readed_data[s][t]["success"]),
                                       conf=float(readed_data[s][t]["conf"]),
                                       trans=numpy.asarray(readed_data[s][t]["trans"]))

    def update_trans_pure(self, s_id, t_id, success, conf, trans):
        self.trans_dict[(s_id, t_id)] = LocalTransformationEstimationResult(s_id, t_id, success, conf, trans)

    def update_trans(self, trans_estimation:LocalTransformationEstimationResult):
        self.trans_dict[(trans_estimation.s, trans_estimation.t)] = trans_estimation

    def get_trans(self, s_id, t_id):
        try:
            local_trans_estimation = self.trans_dict[(s_id, t_id)]
        except:
            local_trans_estimation = LocalTransformationEstimationResult(s_id, t_id, False, 0, numpy.identity(4))
        return local_trans_estimation

    def get_trans_extend(self, s_id, t_id):
        l_trans_e = self.get_trans(s_id, t_id)
        return l_trans_e.success, l_trans_e.conf, l_trans_e.trans
