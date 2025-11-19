import pandas as pd
import numpy as np
import warnings
from collections import deque
from capymoa.classifier import (
    LeveragingBagging,
    HoeffdingTree,
    HoeffdingAdaptiveTree,
    AdaptiveRandomForestClassifier,
)
from capymoa.evaluation import ClassificationEvaluator
from capymoa.drift.detectors import DDM, ADWIN, ABCD

def get_models(schema, global_params, models_to_run, all_model_params):
    """
    Constrói dinamicamente o dicionário de modelos, avaliadores e
    detectores de drift com base nos parâmetros da UI.
    
    Args:
        schema: O schema do stream (necessário para os modelos).
        global_params: Um dicionário com chaves como 'WINDOW_SIZE', 'DELAY_LENGTH'.
        models_to_run: Uma lista de strings com os nomes dos modelos (ex: ["LeveragingBagging"]).
        all_model_params: Um dicionário aninhado (ex: {"LeveragingBagging": {"ensemble_size": 50}}).
    """
    
    window_size = global_params.get("WINDOW_SIZE", 500)
    delay_length = global_params.get("DELAY_LENGTH") 
    models_to_test = {}
    
    if "LeveragingBagging" in models_to_run:
        params = all_model_params.get("LeveragingBagging", {})
        models_to_test["LeveragingBagging"] = {
            "model_instance": LeveragingBagging(
                schema=schema, 
                random_seed=params.get("random_seed", 1),
                ensemble_size=params.get("ensemble_size", 100)
                # Outros parâmetros podem ser adicionados aqui
            ),
            "evaluator": ClassificationEvaluator(schema=schema, window_size=window_size),
            "drift_ddm": DDM(
                min_n_instances=params.get("ddm_min_instances", 30),
                warning_level=params.get("ddm_warning_level", 2.0),
                out_control_level=params.get("ddm_out_control_level", 3.0)
            ),
            "drift_adwin": ADWIN(delta=params.get("adwin_delta", 0.002)),
            "drift_ABCD": ABCD(
                delta_drift=params.get("abcd_delta_drift", 0.002),
                delta_warn=params.get("abcd_delta_warn", 0.01)
                # Outros parâmetros do ABCD podem ser adicionados aqui
            ),
            "results_accuracy": [],
            "results_drift_ddm": [],
            "results_drift_adwin": [],
            "results_drift_ABCD": []
        }

    if "HoeffdingAdaptiveTree" in models_to_run:
        params = all_model_params.get("HoeffdingAdaptiveTree", {})
        models_to_test["HoeffdingAdaptiveTree"] = {
            "model_instance": HoeffdingAdaptiveTree(
                schema=schema,
                random_seed=params.get("random_seed", 0),
                grace_period=params.get("grace_period", 200),
                confidence=params.get("confidence", 0.01),
                tie_threshold=params.get("tie_threshold", 0.05),
                leaf_prediction=params.get("leaf_prediction", 'NaiveBayesAdaptive'),
                nb_threshold=params.get("nb_threshold", 0)
            ),
            "evaluator": ClassificationEvaluator(schema=schema, window_size=window_size),
            "drift_ddm": DDM(
                min_n_instances=params.get("ddm_min_instances", 30),
                warning_level=params.get("ddm_warning_level", 2.0),
                out_control_level=params.get("ddm_out_control_level", 3.0)
            ),
            "drift_adwin": ADWIN(delta=params.get("adwin_delta", 0.002)),
            "drift_ABCD": ABCD(
                delta_drift=params.get("abcd_delta_drift", 0.002),
                delta_warn=params.get("abcd_delta_warn", 0.01)
            ),
            "results_accuracy": [],
            "results_drift_ddm": [],
            "results_drift_adwin": [],
            "results_drift_ABCD": []
        }

    if "AdaptiveRandomForest" in models_to_run:
        params = all_model_params.get("AdaptiveRandomForest", {})
        models_to_test["AdaptiveRandomForest"] = {
            "model_instance": AdaptiveRandomForestClassifier(
                schema=schema,
                random_seed=params.get("random_seed", 1),
                ensemble_size=params.get("ensemble_size", 100),
                max_features=params.get("max_features", 0.6),
                lambda_param=params.get("lambda_param", 6.0),
                disable_drift_detection=params.get("disable_drift_detection", False)
            ),
            "evaluator": ClassificationEvaluator(schema=schema, window_size=window_size),
            "drift_ddm": DDM(
                min_n_instances=params.get("ddm_min_instances", 30),
                warning_level=params.get("ddm_warning_level", 2.0),
                out_control_level=params.get("ddm_out_control_level", 3.0)
            ),
            "drift_adwin": ADWIN(delta=params.get("adwin_delta", 0.002)),
            "drift_ABCD": ABCD(
                delta_drift=params.get("abcd_delta_drift", 0.002),
                delta_warn=params.get("abcd_delta_warn", 0.01)
            ),
            "results_accuracy": [],
            "results_drift_ddm": [],
            "results_drift_adwin": [],
            "results_drift_ABCD": []
        }

    if "HoeffdingTree" in models_to_run:
        params = all_model_params.get("HoeffdingTree", {})
        models_to_test["HoeffdingTree"] = {
            "model_instance": HoeffdingTree(
                schema=schema, 
                random_seed=params.get("random_seed", 1),
                grace_period=params.get("grace_period", 200),
                confidence=params.get("confidence", 0.01),
                tie_threshold=params.get("tie_threshold", 0.05),
                leaf_prediction=params.get("leaf_prediction", 'NaiveBayes'),
                nb_threshold=params.get("nb_threshold", 0)
            ),
            "evaluator": ClassificationEvaluator(schema=schema, window_size=window_size),
            "drift_ddm": DDM(
                min_n_instances=params.get("ddm_min_instances", 30),
                warning_level=params.get("ddm_warning_level", 2.0),
                out_control_level=params.get("ddm_out_control_level", 3.0)
            ),
            "drift_adwin": ADWIN(delta=params.get("adwin_delta", 0.002)),
            "drift_ABCD": ABCD(
                delta_drift=params.get("abcd_delta_drift", 0.002),
                delta_warn=params.get("abcd_delta_warn", 0.01)
            ),
            "results_accuracy": [],
            "results_drift_ddm": [],
            "results_drift_adwin": [],
            "results_drift_ABCD": []
        }
        
    # Adiciona a fila de delay se necessário
    if delay_length is not None and delay_length > 0:
        log_msg = f"Aplicando um delay de {delay_length} instâncias."
        for model_name in models_to_test:
            models_to_test[model_name]["prediction_queue"] = deque(maxlen=delay_length)
    else:
        log_msg = "Nenhum delay aplicado."

    return models_to_test, log_msg