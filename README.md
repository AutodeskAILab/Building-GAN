Building-GAN
======

Code and instructions for our paper:

[Building-GAN: Graph-Conditioned Architectural Volumetric Design Generation](https://arxiv.org/abs/2104.13316), ICCV 2021.

Data
------
- Download the dataset [here](). 
- Put the `raw-data` and `processed data` under the folder `6types-raw_data/sum` and `6types-processed_data`.

For detail about how the raw data are processed, please refer the `Data/process_data.py`.  

Running pretrained models
------
```
TODO inference_test??
```

""
For running a pretrained model check out the following steps:
- Download the trained model [here](). 
- Place them anywhere and rename the dataset to train_data.npy.
- Set the path in variation_bbs_with_target_graph_segments_suppl.py to the path of the folder containing train_data.npy and to the pretrained model.
- Run ***python variation_bbs_with_target_graph_segments_suppl.py***.
- Check out the results in output folder.

Training models
------

For training a model from scratch check out the following steps:
- Follow the steps in Data section.
- run ```python main.py ```. Customized arguments can be set according to ```train_args.py```. 
- Check out ```output``` and ```checkpoints``` folders for intermediate outputs and checkpoints, respectively. They are under the ```runs/run_id/``` where run_id is the serial number of the
 experiment. 

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
If you have any question, feel free to contact Chin-Yi Cheng at <chin-yi.cheng@autodesk.com> or Kai-Hung Chang at <kai-hung.chang@autodesk.com>
.

