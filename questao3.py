class KnowledgeBase:
    def __init__(self):
        self.facts = set()
        self.rules = []
        self.applied_rules = set()  # Para evitar reprocessamento de regras

    def add_fact(self, fact):
        self.facts.add(fact)

    def add_rule(self, premise, conclusion, rule_type):
        self.rules.append((premise, conclusion, rule_type))

    def apply_rule(self, rule):
        premise, conclusion, rule_type = rule
        if rule_type == 'MP':  # Modus Ponens
            if all(p in self.facts for p in premise):
                if conclusion not in self.facts:
                    self.facts.add(conclusion)
                    return True
        elif rule_type == 'MT':  # Modus Tollens
            if all(p in self.facts for p in premise[:1]) and conclusion not in self.facts:
                self.facts.add(conclusion)
                return True
        elif rule_type == 'SH':  # Silogismo Hipotético
            if all(p in self.facts for p in premise[:2]):
                if conclusion not in self.facts:
                    self.facts.add(conclusion)
                    return True
        elif rule_type == 'SD':  # Silogismo Disjuntivo
            if premise[0] in self.facts or not premise[1] in self.facts:
                if conclusion not in self.facts:
                    self.facts.add(conclusion)
                    return True
        elif rule_type == 'And':  # Introdução do &
            if all(p in self.facts for p in premise):
                if conclusion not in self.facts:
                    self.facts.add(conclusion)
                    return True
        return False

    def infer(self, goal):
        stack = [rule for rule in self.rules if rule[1] == goal]
        while stack:
            current_rule = stack.pop()
            if current_rule not in self.applied_rules:
                if self.apply_rule(current_rule):
                    print(f"Aplicada regra: {current_rule}")
                self.applied_rules.add(current_rule)
            if goal in self.facts:
                return True
            # Adiciona novas regras que podem ser aplicáveis após a adição do novo fato
            stack.extend(rule for rule in self.rules if rule[1] == goal and rule not in self.applied_rules)
        return False

# Inicializando a base de conhecimento
kb = KnowledgeBase()

# Adicionando fatos
kb.add_fact('A')
kb.add_fact('B')
kb.add_fact('F')

# Adicionando regras
kb.add_rule(('A', 'B'), 'C', 'MP')
kb.add_rule('A', 'D', 'MP')
kb.add_rule(('C', 'D'), 'E', 'MP')
kb.add_rule(('B', 'E', 'F'), 'G', 'MP')
kb.add_rule(('A', 'E'), 'H', 'MP')
kb.add_rule(('D', 'E', 'H'), 'I', 'MP')

# Inferindo o objetivo
goal = 'H'
if kb.infer(goal):
    print(f"Objetivo {goal} pode ser inferido com base nas regras e fatos fornecidos.")
else:
    print(f"Objetivo {goal} não pode ser inferido.")
