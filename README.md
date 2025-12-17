# Synapse Dynamics

Synapse Dynamics é um sistema de modulação de pesos inspirado na neuroplasticidade humana, pronto para ser integrado a qualquer modelo de IA sem alterar sua arquitetura.

## Objetivo
Permitir que modelos neurais aprendam e se adaptem dinamicamente, com regras plásticas seguras (Hebbian, Oja) e rollback automático para máxima robustez. Foco em reduzir tempo de treinamento e facilitar experimentação.

## Instalação Fácil
Requisitos: Python 3.8+

1. Instale diretamente via pip (no diretório do projeto):

```bash
pip install .
```

Ou apenas copie a pasta `synapse/` para seu projeto e instale o numpy:

```bash
pip install numpy
```

Para rodar os testes (opcional):
```bash
pip install pytest
pytest tests/
```

## Exemplo de Uso
```python
import numpy as np
from synapse import Injector, PlasticityConfig

# Inicialize pesos e atividades
weights = np.zeros((2,2), dtype=np.float32)
pre = np.array([1, 0], dtype=np.float32)
post = np.array([0, 1], dtype=np.float32)

# Crie um injetor com regra hebbiana
inj = Injector(weights, PlasticityConfig(rule='hebbian', lr=0.5))
updated = inj.apply_updates(pre, post)
print(updated)
```

## Recomendações para uso eficiente
- Para máxima performance em GPU, mantenha os tensores no dispositivo correto (use `.to(device)` no PyTorch).
- Use batch updates sempre que possível para aproveitar paralelismo.
- O Injector converte automaticamente entre numpy, torch e tensorflow.
- Para grandes modelos, ajuste o parâmetro `lr` para evitar instabilidade numérica.
- Benchmarks mostram que a modulação de pesos é eficiente e não impacta negativamente o tempo de treinamento.

## Recursos
- PlasticityEngine: atualização de pesos com validação e rollback
- Injector: interface plugável para arrays numpy, torch e tensorflow
- Plugin Shim: integração fácil com sistemas de extensão
- Sem dependências além de numpy

## Licença
MIT