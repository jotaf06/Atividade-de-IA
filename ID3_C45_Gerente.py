import pandas as pd
import numpy as np
from math import log2

# Dados
data = {
    'Historico': ['Ruim', 'Desconhecido', 'Desconhecido', 'Desconhecido', 'Desconhecido', 'Desconhecido', 'Ruim', 'Ruim', 'Bom', 'Bom', 'Bom', 'Bom', 'Bom', 'Ruim', 'Bom', 'Ruim', 'Ruim', 'Desconhecido', 'Bom', 'Bom'],
    'Divida': ['Alta', 'Alta', 'Baixa', 'Baixa', 'Baixa', 'Baixa', 'Baixa', 'Baixa', 'Baixa', 'Baixa', 'Alta', 'Alta', 'Alta', 'Alta', 'Alta', 'Baixa', 'Baixa', 'Baixa', 'Alta', 'Baixa'],
    'Garantia': ['Nenhuma', 'Nenhuma', 'Nenhuma', 'Nenhuma', 'Nenhuma', 'Adequada', 'Nenhuma', 'Adequada', 'Nenhuma', 'Adequada', 'Nenhuma', 'Nenhuma', 'Nenhuma', 'Nenhuma', 'Adequada', 'Adequada', 'Adequada', 'Adequada', 'Adequada', 'Adequada'],
    'Renda': ['$0 a $15k', '$15k a $35k', '$15k a $35k', '$0 a $15k', 'Acima de $35k', 'Acima de $35k', '$0 a $15k', 'Acima de $35k', 'Acima de $35k', 'Acima de $35k', '$0 a $15k', '$15k a $35k', 'Acima de $35k', '$0 a $15k', '$0 a $15k', 'Acima de $35k', '$0 a $15k', '$0 a $15k', 'Acima de $35k', '$0 a $15k'],
    'Classe': ['Alto', 'Alto', 'Moderado', 'Alto', 'Baixo', 'Baixo', 'Alto', 'Moderado', 'Baixo', 'Baixo', 'Alto', 'Moderado', 'Baixo', 'Alto', 'Moderado', 'Moderado', 'Moderado', 'Moderado', 'Baixo', 'Moderado']
}

df = pd.DataFrame(data)

def entropy(column):
    """Calcula a entropia de uma coluna."""
    counts = column.value_counts()
    probabilities = counts / len(column)
    return -sum(p * log2(p) for p in probabilities)

def information_gain(df, feature, target):
    """Calcula o ganho de informação de um atributo."""
    feature_values = df[feature].unique()
    base_entropy = entropy(df[target])
    weighted_entropy = 0
    for value in feature_values:
        subset = df[df[feature] == value]
        weighted_entropy += (len(subset) / len(df)) * entropy(subset[target])
    return base_entropy - weighted_entropy

def gain_ratio(df, feature, target):
    """Calcula o ganho de informação relativo (gain ratio) de um atributo."""
    base_entropy = entropy(df[target])
    feature_values = df[feature].unique()
    weighted_entropy = 0
    split_info = 0
    for value in feature_values:
        subset = df[df[feature] == value]
        probability = len(subset) / len(df)
        weighted_entropy += probability * entropy(subset[target])
        split_info -= probability * log2(probability)
    return (base_entropy - weighted_entropy) / split_info if split_info != 0 else 0

def id3(df, features, target):
    """Constrói uma árvore de decisão usando o algoritmo ID3."""
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    
    if not features:
        return df[target].mode().iloc[0]
    
    gains = [information_gain(df, f, target) for f in features]
    best_feature = features[np.argmax(gains)]
    
    tree = {best_feature: {}}
    feature_values = df[best_feature].unique()
    for value in feature_values:
        subset = df[df[best_feature] == value]
        remaining_features = [f for f in features if f != best_feature]
        tree[best_feature][value] = id3(subset, remaining_features, target)
    
    return tree

def c45(df, features, target):
    """Constrói uma árvore de decisão usando o algoritmo C4.5."""
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    
    if not features:
        return df[target].mode().iloc[0]
    
    gains = [gain_ratio(df, f, target) for f in features]
    best_feature = features[np.argmax(gains)]
    
    tree = {best_feature: {}}
    feature_values = df[best_feature].unique()
    for value in feature_values:
        subset = df[df[best_feature] == value]
        remaining_features = [f for f in features if f != best_feature]
        tree[best_feature][value] = c45(subset, remaining_features, target)
    
    return tree

features = ['Historico', 'Divida', 'Garantia', 'Renda']
target = 'Classe'

# Construção da árvore de decisão usando ID3
tree_id3 = id3(df, features, target)
print("Árvore de Decisão ID3:")
print(tree_id3)

# Construção da árvore de decisão usando C4.5
tree_c45 = c45(df, features, target)
print("Árvore de Decisão C4.5:")
print(tree_c45)
