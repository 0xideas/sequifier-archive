
import pandas as pd
import numpy as np
import json
import onnxruntime

from sequifier.helpers import create_folder_if_not_exists
from sequifier.helpers import numpy_to_pytorch

from sequifier.config.infer_config import load_inferer_config


class Inferer(object):
    def __init__(self, model_path, project_path, id_map, map_to_id):
        self.index_map = {v:k for k,v in id_map.items()} if map_to_id else None
        self.map_to_id = map_to_id
        model_path_load = (f"{project_path}/{model_path}").replace("//", "/")
        self.ort_session = onnxruntime.InferenceSession(model_path_load)

    def infer_probs(self, x):
        """x.shape=(seq_length, any)"""
        ort_inputs = {self.ort_session.get_inputs()[0].name: x}
        ort_outs = self.ort_session.run(None, ort_inputs)
        return(ort_outs[0])

    def infer(self, x, probs=None):
        """x.shape=(seq_length, any)"""
        if probs is None:
            probs = self.infer_probs(x)
        preds = probs.argmax(1)
        if self.map_to_id:
            preds = np.array([self.index_map[i] for i in preds])
        return(preds)




def infer(args):
    config = load_inferer_config(args.config_path, args.project_path)

    model_id = config.model_path.split("/")[-1].replace(".onnx", "")

    print(f"Inferring for {model_id}")

    inference_data_path = (f"{config.project_path}/{config.inference_data_path}").replace("//", "/")
    data = pd.read_csv(inference_data_path, sep=",", decimal=".", index_col=None)
    X, _ = numpy_to_pytorch(data, config.seq_length, config.device)
    X = X.detach().cpu().numpy()
    del data

    if config.map_to_id:
        assert config.ddconfig_path is not None, "If you want to map to id, you need to provide a file path to a json that contains: {{'id_map':{...}}} to ddconfig_path"
        with open(f"{config.project_path}{config.ddconfig_path}", "r") as f:
            id_map = json.loads(f.read())["id_map"]
    else:
        id_map = None

    inferer = Inferer(config.model_path, config.project_path, id_map, config.map_to_id)

    if config.output_probabilities:
        probs = inferer.infer_probs(X.T)
        create_folder_if_not_exists(f"{config.project_path}/outputs/probabilities")
        probabilities_path = (f"{config.project_path}/outputs/probabilities/{model_id}_probabilities.csv")
        print(f"Writing probabilities to {probabilities_path}")
        pd.DataFrame(probs).to_csv(probabilities_path, sep=",", decimal=".", index=False)
        preds = inferer.infer(None, probs)
    else:
        preds = inferer.infer(X.T)

    create_folder_if_not_exists(f"{config.project_path}/outputs/predictions")
    predictions_path = (f"{config.project_path}/outputs/predictions/{model_id}_predictions.csv")
    print(f"Writing predictions to {predictions_path}")
    pd.DataFrame(preds).to_csv(predictions_path, sep=",", decimal=".", index=False)
    print("Inference complete")
