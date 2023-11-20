# Language-independent-Entity-Linking
A implementation of languague-independent Entity Linking using ML Classifier

## Introduction
The state-of-the-art VIE, KIE models as examples LayoutXLM, LayoutLMv3, VI-LayoutXLM, ... that they would make misstakes in certain relatively simple scenarios, where the geometric relations between entities are not complicated. They seems to link two entities dependending more in the semantics than geometric layout. To futher verify my conjecture, I conduct an experiment by creating the linkings between entities using features: distance, direction, angle, coordinates based on bounding boxes of text.

## Environment

The dependencies are listed in `requirements.txt`. Please install follow the command as below:

```bash
pip install -r requirements.txt
```

## Dataset and model checkpoints
You can download raw dataset from [here](), and put the file xfund&funsd under the root folder.

You can download pre-processed dataset and model checkpoints from [here]() include `features`, `weights` and put the folders under src/ directory.

## Training
Before training, please modify configuration in `src/config/cfg.py` file and getting started training

```bash
cd src/
python train --lang <language-specific>
```

## Evaluation
To evaluate for languague specific and one-shot task (training and evaluate on language X).
```bash
cd src
python evaluate.py --lang <languague-specific>
```

To evaluate for zero-shot task (training on English language and evaluate on language X).
```bash
cd src
python evaluate.py --task zero-shot
```

## Experiments
F1-score relation extraction on tasks

### Languague-specific task

||EN(FUNSD)|ZH|JA|ES|FR|IT|DE|PT|Avg|
|--|--|--|--|--|--|--|--|--|--|
|$LayoutXLM_{Large}$|64.1|78.9|72.5|76.7|71.1|76.9|68.4|67.9|72.1|
|Our approach|**92.5**|72.5|**78.6**|75.8|**82.2**|**84.5**|**76.1**|**71.6**|**79.2**|

### Zero-shot task

||EN(FUNSD)|ZH|JA|ES|FR|IT|DE|PT|Avg|
|--|--|--|--|--|--|--|--|--|--|
|$LayoutXLM_{Large}$|64.1|55.3|56.9|57.8|56.1|51.8|48.9|47.9|54.8|
|Our approach|**92.5**|**68.1**|**72.5**|**72.8**|**83.5**|**84.4**|**73.3**|**68.1**|**74.6**|