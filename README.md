Building-GAN
======

Code and instructions for our paper:

[Building-GAN: Graph-Conditioned Architectural Volumetric Design Generation](https://arxiv.org/abs/2104.13316), ICCV 2021.

<img src="https://github.com/AutodeskAILab/Building-GAN/blob/master/images/Building-GAN.png">

Volumetric Design Process
-----
<img src="https://github.com/AutodeskAILab/Building-GAN/blob/master/images/Volumetric_Design.png">

Data
------
- Download the dataset [here](https://d271velnk8wqmg.cloudfront.net/6types-raw_data.zip). 
- Put the subfolders and files in `raw-data` under the folder `6types-raw_data`.
- Run `Data/process_data.py` to process the raw data.

For the detail about how the raw data are processed, please refer the `Data/process_data.py`.  

In the dataset, each volumetric design comprises three json files:
- Global Graph: contains the FAR, program ratios, and the associated rooms for each program type. 
- Local Graph: contains the bubble diagram--the type and size of each room and the connectivity between rooms
- Voxel: contains the voxel graph

Running pretrained models
------

For running a pre-trained model, please follow the steps below:
- The pre-trained model is located at `runs/iccv2021/checkpoints/`
- Run ```python inference.py```
- Check out the results in the `inference/{model}/{epch_current_time}/output` folder.
- Check out the variation results from the same program graph in the `inference/{model}/{epch_current_time}/var_output*` folders.

Training models
------

For training a model from scratch, please follow the steps below:
- Follow the steps in Data section.
- run ```python train.py ```. Customized arguments can be set according to ```train_args.py```. 
- Check out ```output``` and ```checkpoints``` folders for intermediate outputs and checkpoints, respectively. They are under the ```runs/run_id/``` where run_id is the serial number of the
 experiment. 

Requirements
------
- PyTorch >= 1.7.0
- PyTorch Geometric 1.6.2

Citation
------
```
@article{chang2021building,
  title={Building-GAN: Graph-Conditioned Architectural Volumetric Design Generation},
  author={Chang, Kai-Hung and Cheng, Chin-Yi and Luo, Jieliang and Murata, Shingo and Nourbakhsh, Mehdi and Tsuji, Yoshito},
  booktitle={International Conference on Computer Vision},
  year={2021}
}
```

Contact
------
Unfortunately this repo is no longer actively maintained. 
If you have any question, feel free to contact Chin-Yi Cheng @chinyich or Kai-Hung Chang @kaihungc1993


## License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
