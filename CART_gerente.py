import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

# Dados de exemplo
data = {
    'Histórico de Crédito': ['Ruim', 'Desconhecido', 'Desconhecido', 'Desconhecido', 'Desconhecido', 'Desconhecido', 'Ruim', 'Ruim', 'Bom', 'Bom', 'Bom', 'Bom', 'Bom', 'Ruim', 'Bom', 'Ruim', 'Desconhecido', 'Bom', 'Bom'],
    'Dívida': ['Alta', 'Alta', 'Baixa', 'Baixa', 'Baixa', 'Baixa', 'Baixa', 'Baixa', 'Baixa', 'Baixa', 'Alta', 'Alta', 'Alta', 'Alta', 'Alta', 'Baixa', 'Baixa', 'Alta', 'Baixa'],
    'Garantia': ['Nenhuma', 'Nenhuma', 'Nenhuma', 'Nenhuma', 'Nenhuma', 'Adequada', 'Nenhuma', 'Adequada', 'Nenhuma', 'Adequada', 'Nenhuma', 'Nenhuma', 'Adequada', 'Nenhuma', 'Nenhuma', 'Adequada', 'Desconhecida', 'Nenhuma', 'Adequada'],
    'Renda': ['$0 a $15k', '$15k a $35k', '$15k a $35k', '$0 a $15k', 'Acima de $35k', 'Acima de $35k', '$0 a $15k', 'Acima de $35k', 'Acima de $35k', 'Acima de $35k', '$0 a $15k', '$15k a $35k', 'Acima de $35k', '$0 a $15k', '$0 a $15k', 'Acima de $35k', '$0 a $15k', 'Acima de $35k', '$0 a $15k'],
    'Risco': ['Alto', 'Alto', 'Moderado', 'Alto', 'Baixo', 'Baixo', 'Alto', 'Moderado', 'Baixo', 'Baixo', 'Alto', 'Moderado', 'Baixo', 'Alto', 'Moderado', 'Moderado', 'Moderado', 'Moderado', 'Moderado']
}

# Criação do DataFrame
df = pd.DataFrame(data)

# Codificação das variáveis categóricas
df_encoded = pd.get_dummies(df, columns=['Histórico de Crédito', 'Dívida', 'Garantia', 'Renda'])

# Separação das variáveis independentes (X) e dependentes (y)
X = df_encoded.drop('Risco', axis=1)
y = df['Risco']

# Criação do modelo CART
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

# Exibe a árvore de decisão no formato textual
tree_rules = export_text(model, feature_names=list(X.columns))
print(tree_rules)
