# Multi-label Zero-shot Classification 

This is the official implementation of the paper ["Multi-label Zero-shot Classification by Learning to Transfer from External Knowledge"](https://arxiv.org/abs/2007.15610) (BMVC'20, oral).

## Rquirements

Please install the following packages via `pip` or `conda`:

- Python 3
- Pytorch > 1.0
- torchvision > 0.3
- pycocotools
- mlflow
- numpy
- pickle
- tqdm


### Data Preparation
Remember to change the data paths in corresponding yaml.

#### MS-COCO
1. Download mscoco 2014 data from: 
    - [Training images](http://images.cocodataset.org/zips/train2014.zip),
    - [Validation images](http://images.cocodataset.org/zips/val2014.zip),
    - [Annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)
2. Extract mscoco data and make sure files are in the following structure: 
```
    mscoco/
          |-- annoataions/
                        |-- instances_train2014.json
                        |-- instances_val2014.json
          |-- train2014/
                      |-- xxxx.jpg
          |-- val2014/
                     |-- xxxx.jpg
```                     

#### NUS-WIDE

1. Download data from the official [NUS-WIDE](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) website:
    - [Image URLs](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE-urls.rar)
    - [Image lists](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/ImageList.zip)
    - [Concept list](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/ConceptsList.zip)
    - [Tags](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS_WID_Tags.zip)
    
2. Extract the downloaded files into the following structure, 
and put `data/nus_wide/crawl_nuswide_image.py` under the data root:
```
    nus_wide/
          |-- annoataions/
                        |-- Concepts81.txt
                        |-- ImageList/
                        |-- NUS_WID_Tags/
          |-- images/
          |-- crawl_nuswide_image.py
          |-- NUS-WIDE-urls.txt
```  
    
Then download images by running `python crawl_nuswide_image.py`.


#### Visual Genome (Optional)
1. Download annoataions from [Here](https://visualgenome.org/static/data/dataset/objects.json.zip), and extract it under `./data/visual_genome`
2. Download images from [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) and [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip), and extract them into a single directory (e.g., `./data/visual_genome/VG_100K`)
3. Change yamls, remember that `vg_root` points to the root of visual_genome, not VG_100K



## Usage

### Example Usage
1. Run `sh start_mlflow.sh`
2. Run `./scripts/run_mlzsl_posVAE_coco.sh` for COCO dataset, 
3. Open a browser and go to `localhost:5000` to see results
- All runnalbe scripts are under `./scripts`, 
and you can change the hyper-parameters by looking into which yaml file the script uses.
- For fast0tag, simply run `python main_fast0tag_nus81.py`, remember to change the gpu to use in the python file.

#### Run baseline:

##### Fast0Tag (CVPR'16):
- modify `scripts/fast0tag.yaml`, especially set `loss: rank`.
- run `./scripts/run_fast0tag.sh 0`, where `0` is the GPU to use.

##### Logistic Regression
- modify `scripts/fast0tag.yaml`, especially set `loss: bce`.
- run `./scripts/run_fast0tag.sh 0`, where `0` is the GPU to use.

##### SKG (CVPR'18)
- Multi-label zero-shot learning with structured knowledge graphs (CVPR’2018)
- modify `scripts/skg.yaml`, then run `./scripts/run_skg.sh 0`, where `0` is the GPU to use.

#### Run our model:
Can try different `loss` functions, either `bce` or `rank`, where `rank` is a contrasting loss used in Fast0Tag.
- `./scripts/run_gcn_posVAE_coco.sh`
- `./scripts/run_gcn_posVAE_nus.sh`
- `./scripts/run_gcn_posVAE_vg.sh`


## Reference
1. [Fast Zero-Shot Image Tagging](https://arxiv.org/abs/1605.09759)
2. [Multi-Label Zero-Shot Learning with Structured Knowledge Graphs](http://openaccess.thecvf.com/content_cvpr_2018/papers/Lee_Multi-Label_Zero-Shot_Learning_CVPR_2018_paper.pdf)


```
@article{huang2020multi,
  title={Multi-label Zero-shot Classification by Learning to Transfer from External Knowledge},
  author={Huang, He and Chen, Yuanwei and Tang, Wei and Zheng, Wenhao and Chen, Qing-Guo and Hu, Yao and Yu, Philip},
  journal={BMVC},
  year={2020}
}
```
 
