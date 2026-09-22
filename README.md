# Sistema Preditivo de Palavras

Exercício-Programa da disciplina Análise Exploratória de Dados (5º semestre). O sistema lê uma frase e sugere a palavra seguinte mais provável, informando a confiança da predição.

A abordagem não é uma rede neural. É um modelo de linguagem clássico: frequências de n-grams, Teorema de Bayes e suavização de Laplace. A ideia é tornar explícito o que o modelo está calculando, e por que uma palavra ganha de outra.

## Exemplo

```text
$ python -m src.main "O aluno estudou para a prova de"
Sugestão: Estatística, Confiança: 63.58%
```

A confiança é a probabilidade posterior da palavra escolhida, já normalizada nas continuações observadas daquele contexto. Não é um chute de interface: se o histórico for raro ou ambíguo, o valor cai. Se for frequente e pouco disputado, sobe.

Para ver as três melhores hipóteses:

```text
$ python -m src.main "O aluno estudou para a prova de" --top 3
Sugestão: Estatística, Confiança: 63.58%
Sugestão: Probabilidade, Confiança: 8.67%
Sugestão: Cálculo, Confiança: 6.34%
```

## Como rodar

Requisito: Python 3.10 ou superior. Não há dependências externas; o núcleo usa só a biblioteca padrão.

```bash
git clone https://github.com/ditko-labs/sistema-preditivo-de-palavras.git
cd sistema-preditivo-de-palavras

python -m src.main "O aluno estudou para a prova de"
python -m src.main "A análise exploratória de" --top 3
python -m src.main --interactive
python -m src.main --evaluate
python -m unittest discover -s tests
```

| Flag | Efeito |
|------|--------|
| `frase` | Texto cujo próximo token será predito |
| `-i`, `--interactive` | Loop no terminal; digite `sair` para encerrar |
| `-e`, `--evaluate` | Relatório de precisão nos casos rotulados e no hold-out |
| `-k`, `--top` | Quantidade de sugestões no ranking |
| `-n`, `--order` | Ordem máxima do n-gram (padrão: 3) |
| `-a`, `--alpha` | Parâmetro da suavização de Laplace (padrão: 1.0) |
| `-c`, `--corpus` | Caminho de um corpus próprio |

## Organização do repositório

```text
src/                 código do modelo
  preprocessing.py   coleta, normalização e tokenização
  ngrams.py          contagens, vocabulário e token desconhecido
  bayes.py           prior, verossimilhança, evidência e backoff
  evaluator.py       casos rotulados e hold-out
  main.py            interface de linha de comando
data/corpus.txt      corpus acadêmico em português
tests/               testes de unidade das três camadas
```

O fluxo segue as etapas do enunciado: o corpus vira sentenças, as sentenças viram n-grams, os n-grams alimentam Bayes, e a validação mede se a palavra certa sobe no ranking.

## Lógica bayesiana

A pergunta do sistema é: dada a frase que o usuário já escreveu, qual palavra \(W_n\) maximiza a probabilidade de continuar aquele contexto?

\[
P(W_n \mid \text{contexto})
= \frac{P(\text{contexto} \mid W_n)\, P(W_n)}{P(\text{contexto})}
\]

Cada termo tem um papel distinto.

**Prior** \(P(W_n)\). Frequência da palavra no corpus. Sem contexto, o modelo devolve a palavra mais comum. Com Laplace:

\[
P(W_n) = \frac{C(W_n) + \alpha}{N + \alpha V}
\]

**Verossimilhança** \(P(\text{contexto} \mid W_n)\). Quão típico é aquele histórico aparecer imediatamente antes da palavra candidata. Para um bigram, o contexto é a palavra anterior; para um trigram, as duas anteriores.

\[
P(\text{contexto} \mid W_n)
= \frac{C(\text{contexto}, W_n) + \alpha}{C(W_n) + \alpha}
\]

O denominador \(C(W_n)+\alpha\) é o mesmo numerador do prior. O produto \(P(\text{contexto} \mid w)\, P(w)\) fica proporcional a \(C(\text{contexto}, w)+\alpha\). Depois da evidência, isso é o n-gram add-\(\alpha\): a palavra que de fato seguiu o contexto ganha da palavra só porque é comum no corpus.

**Evidência** \(P(\text{contexto})\). Não é estimada à parte. A soma dos scores não normalizados no suporte observado (palavras que já apareceram depois daquele contexto) faz o papel da evidência. Dividir por ela garante que as posteriors somem 1 e que a confiança seja uma porcentagem interpretável. Sem essa restrição de suporte, o Laplace espalharia massa por todo o vocabulário e a confiança cairia para poucos pontos percentuais sem mudar o ranking.

Sem suavização, a fórmula de Bayes com essas contagens se reduz ao estimador clássico de máxima verossimilhança do n-gram, \(C(\text{contexto}, w) / C(\text{contexto})\). A implementação mantém os três termos separados de propósito: o enunciado pede Bayes, não só a razão de contagens.

## Por que Laplace

Num corpus finito a maior parte dos trigrams possíveis nunca aparece. Sem suavização, um contexto inédito zera a verossimilhança de todas as palavras, e o modelo não consegue ranquear nada.

Laplace (add-\(\alpha\), aqui \(\alpha = 1\)) soma uma ocorrência fictícia a cada célula. Contextos novos deixam de ser impossíveis; palavras raras deixam de ser invisíveis. O preço é uma confiança mais baixa do que um modelo superajustado ao treino. Isso é desejável: o sistema admite incerteza em vez de inventar certeza.

## Backoff entre n-grams

Trigrams acertam quando o par anterior é conhecido (`prova de` tende a `estatística`). Bigrams cobrem melhor quando só a última palavra é informativa. Unigrams salvam o ranqueamento quando o contexto some.

O sistema não mistura as três ordens o tempo todo. Uma interpolação fixa com unigram empurra artigos (`a`, `o`) para o topo só porque eles são frequentes. Em vez disso, o preditor usa a maior ordem cujo contexto foi observado e mistura uma fração menor (0,20) só com a ordem imediatamente abaixo.

Se a frase for curta demais para o trigram, o contexto alto é preenchido com o marcador de início de sentença. Se o par nunca apareceu, o modelo recua para o bigram; se a última palavra também for inédita, recua para o prior.

## Bordas do problema

**Palavra fora do vocabulário.** Qualquer token que não apareceu no treino vira `<unk>` só no contexto. O modelo não sugere `<unk>`: o ranking continua restrito às palavras observadas. O contexto desconhecido cai na suavização e o backoff puxa o unigram.

**Frase vazia.** Não há histórico. A verossimilhança do unigram vale 1 e a predição vira o prior.

**Frase curta.** Uma palavra usa bigram, com um pouco de unigram. Duas ou mais usam trigram, com um pouco de bigram, se aquele contexto existir.

**Caracteres especiais e caixa.** O texto vai para minúsculas, pontuação sai, hífens viram espaço (`n-grams` vira `n` + `grams`) e acentos do português permanecem. `estatística` e `estatistica` são tokens diferentes de propósito: o corpus está em português.

**Início e fim de sentença.** Cada sentença é cercada por `<s>` e `</s>` só nas contagens. Esses marcadores não entram no vocabulário de sugestão.

**Confiança baixa.** Se as hipóteses ficam próximas, o contexto é fraco ou o corpus não decide. Olhe as três primeiras sugestões (`--top 3`) antes de tratar o primeiro lugar como resposta única.

## Escolhas de design

O projeto não usa NLTK nem spaCy. Tokenizar com expressões regulares evita download de modelos e deixa o professor rodar o repositório sem setup extra.

O corpus em `data/corpus.txt` é texto original, escrito no domínio da disciplina: estatística, probabilidade, teorema de Bayes, análise exploratória, provas e n-grams. Isso não é um atalho escondido; é o recorte do problema. Um modelo de n-grams só prediz o que viu. Um corpus genérico de jornal diluiria `prova de` em dezenas de continuações e o exemplo do enunciado deixaria de ser didático.

As camadas não se misturam. `preprocessing` não conta n-grams. `ngrams` não calcula posterior. `bayes` não lê arquivo. `evaluator` não imprime. `main` só orquestra. Isso permite trocar o corpus, o \(\alpha\) ou os pesos sem reescrever o preditor.

A confiança é a posterior, não um score arbitrário. Depois da marginalização no suporte observado, o número impresso é \(100 \times P(w \mid \text{frase})\).

## Validação

```bash
python -m src.main --evaluate
python -m unittest discover -s tests
```

Há dois eixos.

**Casos rotulados.** Dez frases escolhidas para cobrir o enunciado e as colisões do domínio (`prova de estatística` versus `prova de matemática`, `teorema de bayes`, `análise exploratória de dados`). Rodam no modelo treinado no corpus inteiro, que é o mesmo modelo da demonstração.

**Hold-out.** Cerca de 12% das sentenças com quatro tokens ou mais são reservadas. Um modelo novo treina só no restante e tenta acertar a última palavra de cada sentença deixada de fora. Essa métrica é mais honesta e, em corpus pequeno, mais dura: o hold-out pune padrões que apareceram uma única vez.

Métricas: acerto top-1 (a primeira sugestão) e top-3 (a palavra certa entre as três primeiras). Em linguagem, top-3 costuma ser o número mais justo, porque várias continuações são linguisticamente válidas.

### Casos de teste manuais

| Entrada | Esperado | Por que o modelo deve acertar |
|---------|----------|-------------------------------|
| `O aluno estudou para a prova de` | estatística | Trigram `prova de _` é o padrão mais repetido do corpus |
| `A análise exploratória de` | dados | Colocação fixa do domínio |
| `O teorema de` | bayes | Quase sempre essa continuação no texto de treino |
| `A suavização de` | laplace | Termo técnico ancorado no enunciado |
| `O modelo utiliza n` | grams | Hífen é quebrado; o contexto `n` + unigram resolve |
| `contexto inédito qualquer` | alguma palavra do vocabulário | Laplace impede probabilidade zero |
| *(frase vazia)* | palavra mais frequente | Prior puro |

Os testes de unidade em `tests/` cobrem tokenização, contagem, soma da posterior igual a 1, preferência pela continuação frequente e ausência de colapso em contexto inédito.

## Desafios da modelagem

O primeiro desafio é a esparsidade. A quantidade de trigrams possíveis cresce com \(V^3\). Num corpus de exercício a maior parte das células da matriz de transição fica vazia. Laplace resolve o zero; interpolação resolve a fragilidade.

O segundo é o vocabulário aberto. Nenhuma coleta cobre o português inteiro. Tratar o inédito como token de contexto, e nunca como sugestão, evita que o sistema “invente” uma palavra que ele não sabe pontuar.

O terceiro é a interpretação da confiança. Normalizar no vocabulário inteiro com Laplace derruba a porcentagem sem mudar quem ganha. O suporte observado deixa a confiança legível: `prova de` fica em 63,58% para `estatística`, porque matemática, probabilidade e cálculo também aparecem no corpus. O 85% do enunciado é o formato da saída, não uma meta a forçar.

O quarto é a diferença entre Bayes explícito e n-gram clássico. As duas coisas coincidem sem suavização. Com Laplace, o denominador da verossimilhança foi alinhado ao numerador do prior para o produto voltar ao add-\(\alpha\). Sem esse alinhamento, palavras só frequentes (artigos) ganhavam de continuações que de fato ocorreram.

## Limitações

O modelo não entende gramática, só estatística de sequência. Ele não generaliza para outro domínio sem outro corpus. O backoff usa pesos fixos, não Witten-Bell nem Kneser-Ney. O corpus é pequeno de propósito, para o exercício caber num repositório e rodar em segundos.

Na validação atual, os dez casos rotulados acertam 100% em top-1. O hold-out (29 sentenças reservadas) fica em 41,4% top-1 e 48,3% top-3: sentenças únicas no corpus pequeno não deixam n-gram o que memorizar. Isso é esperado, não um bug.

Se a tarefa crescer, os próximos passos naturais são um corpus maior, estimação dos pesos de backoff em um conjunto de validação e suavização que respeite a frequência dos contextos, não só um \(\alpha\) global.

## Autoria

Exercício-Programa de Análise Exploratória de Dados. Professor: Msc. Kyung Moo Kim. Semestre: 5º.
