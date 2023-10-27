# Language-independent-Entity-Linking
A implementation of languague-independent Entity Linking using XGboost Classifier

## Introduction
The state-of-the-art VIE, KIE models as examples LayoutXLM, LayoutLMv3, VI-LayoutXLM, ... that they would make misstakes in certain relatively simple scenarios, where the geometric relations between entities are not complecated. They seems to link two entities dependending more in the semantics than geometric layout. To futher verify my conjecture, I conduct an experiment by creating the linkings between entities using features: distance, direction, angle, coordinates based on bounding boxes of text.

## Environment

The dependencies are listed in `requirements.txt`. Please install follow the command as below:

```bash
pip install -r requirements.txt
```

## Dataset and model checkpoints
You can download raw dataset from [here](), and put the file xfund&funsd under the root folder.

You can download pre-processed dataset and model checkpoints from [here]() include `features`, `weights` and put the folders under src/ directory.

## Training

## Evaluation

## Experiments

