# RoCOCO
Robust Benchmark MS-COCO to Stress-test Robustness of Image-Text Matching Models

## Text-to-Image

### 1. Create additional images 
- Requirements:
```
torchvision
opencv-python 
```
- `mix_images.py`
	- `method`: str (default: `mix`): Method to mix images (choices: mix, patch).
	- `lam`: float (default: 0.9): Proportion of mixing images.
	- `data_path`: str (default: `None`): Directory where COCO images locate. New mixed images will be also stored in this directory with the directory name of `method + '_' + str(lam)` (e.g., mix_0.9). 
	- `img_list`: str (default: `data/coco_karpathy_test.json`): COCO Test image lists.
    - `seed` : int (default: 1): Random seed. In the paper, results are averaged from experiments with 3 different seeds, `1, 10, 100`.
- Usage: 
```bash
python mix_images.py --method mix --lam 0.9 --data_path /data/coco/images/ --seed 1
```

### 2. Evaluation
We use the image data created above to conduct image retrieval test.

Since the retrieval methods vary for each baseline, it is necessary to follow the specific baseline method when calculating similarity using text/image embeddings. Here, we provide examples of evaluation methods from [BLIP (ICML'22)](https://arxiv.org/abs/2201.12086).

#### BLIP example
- Requirements:
```
pyyaml
transformers==4.15.0
timm==0.4.12
fairscale==0.4.4
pycocoevalcap
```
- `eval_blip_t2i.py`
	- `miximage`: str (default: `mix_0.9`): Additional images created above to confuse the models.
	- `testfilename`: str (default: `coco_karpathy_test.json`): COCO Test dataset to list texts and images.
	- `config`: str (default: `blip/configs/retrieval_coco.yaml`): Config file for BLIP. Config file has :
	    - image_root: COCO Image directory (e.g., `data/coco/images/`)
		- ann_root: Annotation directory (e.g., `annotation/`)
- Usage: 
```bash
python -m torch.distributed.run --nproc_per_node=4 eval_blip_t2i.py --miximage mix_0.9
```
 
 
 ## Image-to-Text
 ### 1. Data
 Data is contained in `annotations` directory. 
 Json file include the image path, ground-truth captions, and adversarial captions. 
 Note that we are more thoroughly examining for the public release.
 ```bash
 .
├── annotations                 
├   ├── bert_voca.json              
├   ├── danger.json              
├   ├── final_diff_concept.json             
├   ├── final_same_concept.json            
```

### 2. Evaluation

#### BLIP example
- `eval_blip_i2t.py`
	- `testfilename`: str (default: `final_diff_concept.json`): Adversarial Test dataset to list texts and images.
	- `config`: str (default: `blip/configs/retrieval_coco.yaml`): Config file for BLIP. Config file has :
	    - image_root: COCO Image directory (e.g., `data/coco/images/`)
		- ann_root: Annotation directory (e.g., `annotation/`)
- Usage: 
```bash
python -m torch.distributed.run --nproc_per_node=4 eval_blip_i2t.py --testfilename final_diff_concept.json
```

