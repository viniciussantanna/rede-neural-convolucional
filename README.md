# Reconhecimento de Cães e Gatos com PyTorch

Este repositório contém um modelo de rede neural convolucional (CNN) baseado no ResNet-18 para classificação de imagens de cães e gatos utilizando PyTorch.

## Requisitos
Antes de iniciar, certifique-se de ter as seguintes dependências instaladas:

```bash
pip install torch torchvision matplotlib
```

## Estrutura do Conjunto de Dados
O conjunto de dados deve estar organizado da seguinte forma:

```
dataset/
    train/
        dogs/
        cats/
    val/
        dogs/
        cats/
```

Cada subpasta deve conter imagens correspondentes a cada classe.

## Como Usar

### 1. Treinamento do Modelo
Execute o script para treinar o modelo:

```bash
python train.py
```

Durante o treinamento, o script irá exibir a perda (loss) e a precisão (accuracy) para cada época.

### 2. Validação e Avaliação
O modelo será avaliado no conjunto de validação após cada época.

### 3. Salvando o Modelo
O modelo treinado será salvo como `model_cats_and_dogs.pth`.

## Estrutura do Código

- **`train.py`**: Script principal para carregar os dados, treinar e validar o modelo.
- **`model_cats_and_dogs.pth`**: Arquivo salvo com os pesos do modelo treinado.

## Visualização das Previsões
Após o treinamento, o script exibe algumas imagens de validação com suas previsões.

## Melhorias Futuras
- Implementar aumento de dados (data augmentation)
- Testar outras arquiteturas como VGG16 e EfficientNet
- Criar um script de inferência para classificação de novas imagens

## Licença
Este projeto está sob a licença LIVRE. Sinta-se à vontade para usar e modificar!

