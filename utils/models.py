# utils/models.py

# Importe as bibliotecas necessárias para Stream Mining (Ex: scikit-multiflow ou similar)
# from skmultiflow.trees import HoeffdingTree # Exemplo de importação real

class StreamModel:
    """Classe base simulada para um modelo de Stream Mining."""
    def __init__(self, name, hyperparameters):
        self.name = name
        self.hyperparameters = hyperparameters
        self.accuracy_history = []
        self.is_trained = False

    def train_on_stream(self, data_stream, epochs):
        """Simula o treinamento com um stream de dados."""
        print(f"Iniciando treinamento do {self.name}...")
        
        # --- SIMULAÇÃO DE TREINAMENTO REAL (A SER SUBSTITUÍDO PELA SUA LÓGICA) ---
        import numpy as np
        
        # Gera dados falsos para o gráfico de acurácia
        acc_start = np.random.uniform(0.55, 0.70)
        acc_end = np.random.uniform(0.85, 0.98)
        
        # Gera a evolução da acurácia com ruído
        self.accuracy_history = np.linspace(acc_start, acc_end, epochs) + np.random.rand(epochs) * 0.03
        
        self.is_trained = True
        return self.accuracy_history

# Defina uma classe real para cada algoritmo
class HoeffdingTreeModel(StreamModel):
    def __init__(self, hyperparameters):
        super().__init__("Hoeffding Tree", hyperparameters)
        # self.model = HoeffdingTree(confidence=hyperparameters['confidence']) # Implementação real

# class AdaptiveRandomForestModel(StreamModel):
#     def __init__(self, hyperparameters):
#         super().__init__("Adaptive Random Forest", hyperparameters)
#         # self.model = AdaptiveRandomForest(n_estimators=hyperparameters['n_estimators']) # Implementação real