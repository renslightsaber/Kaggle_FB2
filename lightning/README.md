# [공사 중][Pytorch Lightning] How to train or inference in CLI? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1M7UAYk6ONEjpL5QaOPYb8zssmDDFGNXY?usp=sharing) [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/FB_TWO/groups/pytorch_cli_practice-Baseline/workspace?workspace=user-wako)


#### Check Jupyter Notebook Version at [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IimJTkmvZ8bFCXUzQhXbYHGBaqk3Sfx4?usp=sharing) [![wandb](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/wako/FB_TWO/groups/Pytorch-Baseline/workspace?workspace=user-wako)

<img src="/imgs/스크린샷 2023-03-14 오후 10.08.45.png" width="85%"></img>

  
## Download Data Kaggle API Command 
```python
$ kaggle competitions download -c feedback-prize-effectiveness
$ unzip '*.zip'
```
 
## [wandb login in CLI interface](https://docs.wandb.ai/ref/cli/wandb-login)
```python
$ wandb login --relogin '######### your API token ###########'                  
``` 

## Train 
```bash
$ python train.py --base_path '/content/Kaggle_FB2/lightning/' \
                  --model_save '/content/drive/MyDrive/깃헙/Projects/Kaggle FB2/pl/cli/' \
                  --sub_path '/content/drive/MyDrive/깃헙/Projects/Kaggle FB2/pl/cli' \
                  --model "microsoft/deberta-v3-base" \
                  --hash "Lightning_cli" \
                  --grad_clipping True\
                  --n_folds 3 \
                  --n_epochs 3 \
                  --device 'cuda' \
                  --max_length 256 \
                  --train_bs 8
```
- `base_path` : Data가 저장된 경로 (Default: `./data/`)
- `sub_path`  : `submission.csv` 제출하는 경로
- `model_save`: 학습된 모델이 저장되는 경로
- `model`: Huggingface Pratrained Model (Default: `"microsoft/deberta-v3-base"`)
- `hash`: Name for WANDB Monitoring
- `n_folds`  : Fold 수
- `n_epochs` : Epoch
- `seed` : Random Seed (Default: 2022)
- `train_bs` : Batch Size (Default: 8)
- `max_length` : Max Length (Default: 512) for HuggingFace Tokenizer when fine-tuning
- `grad_clipping`: [Gradient Clipping](https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem)
- `ratio` : 데이터를 Split하여 `train`(학습) 과 `valid`(성능 평가)를 만드는 비율을 의미. 정확히는 `train`의 Size 결정
- `device`: GPU를 통한 학습이 가능하다면, `cuda` 로 설정할 수 있다.
- `learning_rate`, `weight_decay`, `min_lr`, `T_max` 등은 생략 
- [`train.py`](https://github.com/renslightsaber/Kaggle_FB2/blob/main/lightning/train.py) 참고!   


## 주의
 - CLI 환경에서 train 시킬 때, `tqdm`의 Progress Bar가 엄청 많이 생성된다. 아직 원인과 해결을 못 찾은 상태이다.
 - Colab과 Jupyter Notebook에서는 정상적으로 Progress Bar가 나타난다.



## Inference 
```bash
$ python inference.py --base_path '/content/Kaggle_FB2/lightning/' \
                      --model_save '/content/drive/MyDrive/깃헙/Projects/Kaggle FB2/pl/cli/' \
                      --sub_path '/content/drive/MyDrive/깃헙/Projects/Kaggle FB2/pl/cli' \
                      --model "microsoft/deberta-v3-base" \
                      --hash "lightning_cli_practice" \
                      --n_folds 3 \
                      --n_epochs 3 \
                      --device 'cuda' \
                      --max_length 256 \
                      --valid_bs 16
```
- `base_path` : Data가 저장된 경로 (Default: `./data/`)
- `sub_path`  : `submission.csv` 제출하는 경로
- `model_save`: 학습된 모델이 저장되는 경로
- `model`: train했을 때의 Huggingface Pratrained Model (Default: `"microsoft/deberta-v3-base"`)
- `n_folds`  : `train.py`에서 진행항 KFold 수
- `n_epochs` : train 했을 때의 Epoch 수 (submission 파일명에 사용)  
- `seed` : Random Seed (Default: 2022)
- `valid_bs` : Batch Size for Inference (Default: 16) 
- `max_length` : Max Length (Default: 512) for HuggingFace Tokenizer
- `device`: GPU를 통한 학습이 가능하다면, `cuda` 로 설정할 수 있다.
- [`inference.py`](https://github.com/renslightsaber/Kaggle_FB2/blob/main/lightning/inference.py) 참고!   






