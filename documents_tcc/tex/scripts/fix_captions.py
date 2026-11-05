import re
captions = {
    'fig:2opt-flowchart': 'Fluxograma do algoritmo 2-opt.',
    'fig:ga-flowchart': 'Fluxograma de Algoritmo Genético.',
    'fig:ga2opt-flowchart': 'Fluxograma do algoritmo GA+2-opt.',
    'fig:crainic-taxonomy': 'Resumo dos três tipos de paralelização de Crainic e Toulouse.',
    'fig:class-diagram': 'Diagrama de classes do framework experimental.',
    'fig:stat-flowchart': 'Fluxograma de decisão para testes estatísticos.',
    'fig:results-flowchart': 'Fluxo de análise dos resultados.'
}
with open('/home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/second_draft.md', 'r') as f:
    text = f.read()
for k, v in captions.items():
    text = re.sub(rf'```\{{\.mermaid loc=assets/diagrams format=pdf caption="" caption_placeholder="true" #{k}\}}',
                  f'```{{.mermaid loc=assets/diagrams format=pdf caption="{v}" #{k}}}', text)
with open('/home/lucas_galdino/chimera/gpu_accelerated/documents_tcc/second_draft.md', 'w') as f:
    f.write(text)