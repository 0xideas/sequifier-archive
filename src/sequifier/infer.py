
import pandas as pd
import onnxruntime
from argparse import ArgumentParser

from helpers import create_folder_if_not_exists
from helpers import numpy_to_pytorch

from config.infer_config import load_inferer_config


class Inferer(object):
    def __init__(self, model_path, project_path):
        model_path_load = (f"{project_path}/{model_path}").replace("//", "/")
        self.ort_session = onnxruntime.InferenceSession(model_path_load)

    def infer_probs(self, x):
        """x.shape=(seq_length, any)"""
        ort_inputs = {self.ort_session.get_inputs()[0].name: x}
        ort_outs = self.ort_session.run(None, ort_inputs)
        return(ort_outs[0])

    def infer(self, x):
        """x.shape=(seq_length, any)"""
        probs = self.infer_probs(x)
        preds = probs.argmax(1)
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

    inferer = Inferer(config.model_path, config.project_path)

    if config.output_probabilities:
        probs = inferer.infer_probs(X.T)
        create_folder_if_not_exists(f"{config.project_path}/outputs/probabilities")
        probabilities_path = (f"{config.project_path}/outputs/probabilities/{model_id}_probabilities.csv")
        print(f"Writing probabilities to {probabilities_path}")
        pd.DataFrame(probs).to_csv(probabilities_path, sep=",", decimal=".", index=False)
        preds = probs.argmax(1)
    else:
        preds = inferer.infer(X.T)

    create_folder_if_not_exists(f"{config.project_path}/outputs/predictions")
    predictions_path = (f"{config.project_path}/outputs/predictions/{model_id}_predictions.csv")
    print(f"Writing predictions to {predictions_path}")
    pd.DataFrame(preds).to_csv(predictions_path, sep=",", decimal=".", index=False)
    print("Inference complete")
