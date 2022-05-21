# Runtime  Detection  of  Executional  Errors  in  Robot-Assisted  Surgery

This is the code of Runtime  Detection  of  Executional  Errors  in  Robot-Assisted  Surgery\
Presented at 2022 International Conference on Robotics and Automation (ICRA 2022) \
The repo includes the models (LSTM, CNN, Siamese-LSTM and Siamses-CNN) and the experimental setups(GSTS,GST*,G\*TS,G\*T*). 

A video describing this work is available [here](https://www.youtube.com/watch?v=70h_hcaIXpc)


# Install
conda install --file requirements.txt

# Dataset
The error labels for the Suring task from the [JIGSAWS](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release) dataset can be found [here](https://github.com/UVA-DSA/ExecProc_Error_Analysis/tree/main/Error_Labels/Consensus_error_labels_suturing). We preprocessed the data with downsampling and normalization. The preprocessed data can be found in this repo.  

# How to run
We have 4 main scripts. The 'type' variable can be changed to 'double' or 'single' for performance evaluation on the Siamese network or the LSTM,CNN.  
* GSTS.py : training with gesture specific task specific setting
* GST*.py : training with gesture specific task non-specific setting
* G\*TS.py : training with gesture non-specific task specific setting
* G\*T* : training with gesture non-specific task non-specific setting

The util.py contains utility functions including data loading and parameter tuning.  

# Contact
Please let us know if you have any questions. You can send an email to 
Zongyu Li (zl7qw@virginia.edu)

# Citation

Bibtex 

@article{li2022runtime,
      title={Runtime Detection of Executional Errors in Robot-Assisted Surgery}, 
      author={Zongyu Li and Kay Hutchinson and Homa Alemzadeh},
      year={2022},
      booktitle={2022 IEEE International Conference on Robotics and Automation (ICRA)}}
