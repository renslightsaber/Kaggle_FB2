# [[Kaggle] Feedback Prize - Predicting Effective Arguments](https://www.kaggle.com/competitions/feedback-prize-effectiveness)
 Competition URL is closed, and we don't know why.

<img src="/imgs/스크린샷 2023-03-13 오후 2.48.04.png" width="99%"></img>

## Competition Info
 - Period: 2022.05.25 - 2022.08.24
 - Joined as: `TEAM`
 - TEAM_NAME : `Rotation of Salmon Planet` with [A.RYANG](https://github.com/nomaday)
 - TASK: `Text Classification`
 - Evaluation Metric: `Multi-Class Logarithmic Loss`
 - Environment: Colab & Kaggle-notebook

## Result 
 - PUBLIC  : 902nd
 - PRIVATE : 899th
 - Final: 899th / 1558 teams 
 
## Purpose : Experience 🤗Huggingface, Pytorch Lightning⚡ and wandb.
- Train and inference not only with pytorch🔥 but also with 🤗Huggingface.
  - During the periods of Compeition, we just concentrated on `score`, which means that we couldn't consider about these.
  - We just wanted Experience of 'wandb' logging.
  - This is why score from codes (on this github repository) is not so good.
- Also, train and inference with Pytorch Lightning⚡.
  - Could write the code briefly because of Pytorch Lightning⚡'s convenience.
  - Also Pytorch Lightning⚡ allows logging in wandb easily.  
- In practices with 🤗Huggingface library, there're two models
  - [AutoModel](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModel)  
    : This could be different  
    - [SequenceClassifierOutput](https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput)  
    - [Model outputs](https://huggingface.co/docs/transformers/main_classes/output)
  - [AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSequenceClassification)
  - [Check official docs](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html)
 
## How to train or inference in CLI? 
- [pip install ... ](https://github.com/renslightsaber/Kaggle_FB2/blob/main/needtoinstall.md)
- [🔥pytorch - Practice in cli](https://github.com/renslightsaber/Kaggle_FB2/blob/main/pytorch) 
- [🤗huggingface - Practice in cli](https://github.com/renslightsaber/Kaggle_FB2/blob/main/huggingface) 
- [⚡Pytorch Lightning - Practice in cli](https://github.com/renslightsaber/Kaggle_FB2/blob/main/lightning) 
- Check [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/FB_TWO?workspace=user-wako)

## Refrences
- [Pytorch Lightning wandb_logger](https://docs.wandb.ai/guides/integrations/lightning) 
- [LightningModule from checkpoint](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#lightningmodule-from-checkpoint)
- [Trainer accelerater](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags)
- [GunwooHan/nunbody_segmentation](https://github.com/GunwooHan/nunbody_segmentation)
- [cherryb12/2022-AI-Challenge-QA-Answering](https://github.com/cherryb12/2022-AI-Challenge-QA-Answering)
- [[Pytorch] FeedBack DeBERTa-V3 Baseline](https://www.kaggle.com/code/debarshichanda/pytorch-feedback-deberta-v3-baseline)
- [FeedBack Inference](https://www.kaggle.com/code/debarshichanda/feedback-inference)
