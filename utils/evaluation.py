import pandas as pd
import numpy as np
import random
from collections import deque
import streamlit as st 

from capymoa.classifier import (
    LeveragingBagging,
    HoeffdingTree,
    HoeffdingAdaptiveTree,
    AdaptiveRandomForestClassifier,
)
from capymoa.evaluation import ClassificationEvaluator
from capymoa.drift.detectors import DDM, ADWIN, ABCD

@st.cache_data
def get_attack_summary_table(df_processed, target_col):
    """
    Analisa o DataFrame processado e retorna um resumo CONCATENADO
    mostrando o início e o fim de cada TIPO de ataque.
    """
    if target_col not in df_processed.columns:
        return pd.DataFrame(columns=["Ataque", "Início (Instância)", "Fim (Instância)", "Total de Amostras"])

    attacks_df = df_processed[df_processed[target_col] != 'BENIGN'].copy()
    
    if attacks_df.empty:
        # st.info("Nenhum ataque (não-BENIGN) foi encontrado no stream processado.")
        return pd.DataFrame(columns=["Ataque", "Início (Instância)", "Fim (Instância)", "Total de Amostras"])

    attacks_df.reset_index(inplace=True) 
    
    summary = attacks_df.groupby(target_col)['index'].agg(
        Início_Instância='min',
        Fim_Instância='max',
        Total_Amostras='count'
    ).reset_index()
    
    summary.rename(columns={target_col: 'Ataque'}, inplace=True)
    
    return summary

def get_models(schema, global_params, models_to_run, all_model_params):
    window_size = global_params.get("WINDOW_SIZE", 500)
    delay_length = global_params.get("DELAY_LENGTH", None)
    
    models_to_test = {}
    
    # --- 1. LeveragingBagging ---
    if "LeveragingBagging" in models_to_run:
        params = all_model_params.get("LeveragingBagging", {})
        models_to_test["LeveragingBagging"] = {
            "model_instance": LeveragingBagging(
                schema=schema, 
                random_seed=params.get("random_seed", 1),
                ensemble_size=params.get("ensemble_size", 100)
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

    # --- 2. HoeffdingAdaptiveTree ---
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

    # --- 3. AdaptiveRandomForest ---
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

    # --- 4. HoeffdingTree ---
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
        
    if delay_length is not None and delay_length > 0:
        log_msg = f"Aplicando um delay de {delay_length} instâncias."
        for model_name in models_to_test:
            models_to_test[model_name]["prediction_queue"] = deque(maxlen=delay_length)
    else:
        log_msg = "Nenhum delay aplicado."

    return models_to_test, log_msg

def run_evaluation_stream(stream, models_to_evaluate, eval_params):
    MAX_INSTANCES = eval_params.get("MAX_INSTANCES", 10000)
    WINDOW_SIZE = eval_params.get("WINDOW_SIZE", 500)
    DELAY_LENGTH = eval_params.get("DELAY_LENGTH", None)
    LABEL_PROBABILITY = eval_params.get("LABEL_PROBABILITY", 1.0)
    
    instance_count_history = []
    
    if DELAY_LENGTH is not None and DELAY_LENGTH > 0:
        for model_name in models_to_evaluate:
            models_to_evaluate[model_name]["prediction_queue"] = deque(maxlen=DELAY_LENGTH)

    stream.restart() 
    count = 0
    while stream.has_more_instances() and count < MAX_INSTANCES:
        instance = stream.next_instance()
        is_window_boundary = (count + 1) % WINDOW_SIZE == 0
        
        yielded_metrics = {"instance": count + 1}
        
        for model_name, state in models_to_evaluate.items():
            model = state["model_instance"]
            
            prediction = model.predict(instance)
            try:
                prediction_value = prediction[0]
            except (IndexError, TypeError):
                prediction_value = prediction 
            
            state["evaluator"].update(instance.y_index, prediction_value)
            error = 0 if prediction_value == instance.y_index else 1
            
            if "window_errors" not in state:
                 state["window_errors"] = deque(maxlen=WINDOW_SIZE)
            state["window_errors"].append(error)
            
            state["drift_ddm"].add_element(error)
            if state["drift_ddm"].detected_change():
                state["results_drift_ddm"].append(count)
                state["drift_ddm"].reset()
                
            state["drift_adwin"].add_element(error)
            if state["drift_adwin"].detected_change():
                state["results_drift_adwin"].append(count)
                state["drift_adwin"].reset()

            state["drift_ABCD"].add_element(instance)
            if state["drift_ABCD"].detected_change():
                state["results_drift_ABCD"].append(count)
                state["drift_ABCD"].reset()
            
            instance_to_train = None
            if DELAY_LENGTH is not None and DELAY_LENGTH > 0:
                queue = state["prediction_queue"]
                
                if random.random() <= LABEL_PROBABILITY:
                    queue.append(instance) 
                else:
                    queue.append(None) 
                
                if len(queue) == DELAY_LENGTH:
                    delayed_item = queue.popleft() 
                    if delayed_item is not None:
                        instance_to_train = delayed_item
            else:
                if random.random() <= LABEL_PROBABILITY:
                    instance_to_train = instance
            
            if instance_to_train:
                model.train(instance_to_train)
            
            if is_window_boundary:
                if state["window_errors"]:
                    mean_error = np.mean(state["window_errors"])
                    accuracy_pct = (1.0 - mean_error) 
                    state["results_accuracy"].append(accuracy_pct)
                else:
                    state["results_accuracy"].append(1.0)
                
                yielded_metrics[model_name] = {
                    "Acurácia": state["results_accuracy"][-1], 
                    "Drift (DDM)": 1 if count in state["results_drift_ddm"] else 0,
                    "Drift (ADWIN)": 1 if count in state["results_drift_adwin"] else 0,
                    "Drift (ABCD)": 1 if count in state["results_drift_ABCD"] else 0,
                }
        
        if is_window_boundary:
            instance_count_history.append(count + 1)
            yield yielded_metrics, instance_count_history
            
        count += 1
    
    # --- Função helper CORRIGIDA (Request 1) ---
    def get_metric(metric_func):
        """
        Chama a métrica e retorna 0.0 se for None ou NaN.
        Não passa argumentos extras que possam causar erro.
        """
        try:
            val = metric_func() # Chama sem argumentos (ex: .f1_score())
            return val if pd.notna(val) else 0.0
        except Exception:
            return 0.0
    # -------------------------------------------

    final_report = {}
    for model_name, state in models_to_evaluate.items():
        evaluator = state["evaluator"]
        
        final_report[model_name] = {
            "Acurácia": get_metric(evaluator.accuracy),
            "F1-Score": get_metric(evaluator.f1_score),   # Removido weighted=True
            "Precision": get_metric(evaluator.precision), # Removido weighted=True
            "Recall": get_metric(evaluator.recall),       # Removido weighted=True
            "Kappa": get_metric(evaluator.kappa)
        }
    
    yield {
        "status": "completed", 
        "final_report": final_report, 
        "models_final_state": models_to_evaluate, 
        "instance_history": instance_count_history
    }