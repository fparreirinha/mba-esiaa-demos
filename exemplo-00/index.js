import tf, { mod } from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();

    // Primeira camada da rede neural com 10 neurônios e função de ativação ReLU
    // entrada de 7 características (idade normalizada, 3 cores e 3 localizaçóes)
    
    // 80 neuronios = aqui coloquei tudo isso, pq tem pouca base de treino
    // quanto mais neuronios, mais complexa a rede, e mais dados de treino são necessários para evitar overfitting
    // usurá mais processamento, mas pode aprender padrões mais complexos
    
    // A ReLU é uma função de ativação que ajuda a rede a aprender padrões não lineares, 
    // o que é importante para capturar relações complexas entre as características de entrada 
    // e as categorias de saída.
    // é como se ela deixasse somente os dados interessantes seguierem adiante na rede
    // Se a informação chegou nesse neuronio é positiva, passa pra frente!
    // se for negativa, zera a informação, ou seja, não passa pra frente, o que ajuda a rede a
    // focar nas informações mais relevantes e evitar ruídos.
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));

    // Saída: 3 neurônios (premium, medium, basic) com função de ativação softmax
    // o softmax é uma função de ativação que transforma os outputs da camada em probabilidades,
    // ou seja, cada neurônio na camada de saída vai representar a probabilidade de uma pessoa pertencer
    // a uma das categorias (premium, medium, basic).
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    // compilando o modelo
    // optimizer Adam (Adaptive Moment Estimation) é um algoritmo de otimização que ajusta os pesos da rede neural
    // durante o treinamento para minimizar a função de perda.
    // loss categoricalCrossentropy é uma função de perda usada para problemas de classificação multiclasse,
    // onde as classes são mutuamente exclusivas. Ela mede a diferença entre as distribuições de probabilidade
    // previstas pelo modelo e as distribuições reais (labels).
    // é um treinador pessoal moderno para redes neurais:
    // ajusta os pesos de forma eficiente e inteligente
    // aprende com histórico de erros e acertos.

    // loss: categoricalCrossentropy
    // ele compara o que o modelo "acha" (os scores de cada categoria)
    // com o que é a verdade (labels one-hot encoded) e calcula o erro.

    // quanto mais distante da previsão do modelo da resposta correta
    // maior o erro (loss) e o modelo vai ajustar os pesos para tentar reduzir esse erro.
    // exemplo classico: classificação de imagens, recomendação de produtos, etc.
    // qq coisa em que a resposta é "apenas uma entre várias categorias possíveis"
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0, // não mostra o progresso do treinamento no console
            epochs: 100, // número de vezes que o modelo vai passar por todo o dataset de treino
            //batchSize: 3, // número de amostras que o modelo vai processar antes de atualizar os pesos
            shuffle: true, // embaralha os dados a cada época para melhorar o treinamento
            callbacks: {
                // onEpochEnd: (epoch, logs) => {
                //     console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
                // }
            }
        }
    );

    return model;
}

async function predict(model, pessoa) {
    // transformar o array js para o tensor (tfjs)
    const inputTensor = tf.tensor2d(pessoa);

    // faz a previsão usando o modelo treinado
    const pred = model.predict(inputTensor);
    const predArray = await pred.array();
    // console.log("Probabilidades de cada categoria (premium, medium, basic):", predArray[0]);
    return predArray[0].map((prob, index) => ({ categoria: labelsNomes[index], probabilidade: prob }));
}
// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// quanto mais dados melhor!
// assim o algoritmo tem mais exemplos para aprender padrões e generalizar melhor para novos casos.
const model = await trainModel(inputXs, outputYs)

const pessoa = { nome: "Felipe", idade: 28, cor: "verde", localizacao: "Curitiba" }
// Normalizando a idade da nova pessoa usando o mesmo padrão do treino
// exemplo: idade_min = 25, idade_max = 40
// idade_normalizada = (idade - idade_min) / (idade_max - idade_min)
// (28 - 25) / (40 - 25) = 0.2
const novaPessoa = [
    [
        0.2, // idade normalizada
        1,   // azul
        0,   // vermelho
        0,   // verde
        1,   // São Paulo
        0,   // Rio
        0    // Curitiba
    ]
]

const predictions = await predict(model, novaPessoa)
const results = predictions
    .sort((a, b) => b.probabilidade - a.probabilidade)
    .map(pred => `${pred.categoria}: ${(pred.probabilidade * 100).toFixed(2)}%`)
    console.log(`Previsões para ${pessoa.nome}:`, results.join(', '))