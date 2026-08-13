# DeepCDR in TensorFlow

> A TensorFlow port of DeepCDR for cancer drug-response prediction.

---

Built for the Argonne National Laboratory precision-medicine collaboration so that DeepCDR could
serve as one arm of the MEnKF stacker alongside models written in the same framework. Framework parity
matters here: the stacker needs to extract intermediate representations from every arm through one interface.

## Notebooks

**8 notebooks**, committed with their outputs intact so every figure and result table renders on GitHub without re-running anything.

- `Get_Joint_Learner_GDSC_1_Preds.ipynb`
- `Get_Joint_Learner_GDSC_2_Preds.ipynb`
- `Transfer_Learning_on_GDSC_2.ipynb`
- `get_sample_data.ipynb`
- `train_deepcdr_gcn.ipynb`
- `train_deepcdr_gcn_dropout_in_train.ipynb`
- `train_deepcdr_gcn_dropout_in_train_ccle_0.ipynb`
- `train_deepcdr_gcn_dropout_in_train_ctrpv2_0.ipynb`

## About this repository

Research code from my doctoral work at the University of Nebraska–Lincoln. Trained model checkpoints and bulk datasets are excluded from version control; the notebooks regenerate them. Previously hosted at `github.com/Ved-Piyush/DeepCDR_TF`.

---

**Ved Piyush, PhD** · Statistics, University of Nebraska–Lincoln  
[vedpiyush93@gmail.com](mailto:vedpiyush93@gmail.com) · [Google Scholar](https://scholar.google.com/citations?hl=en&user=657rVYAAAAAJ) · [LinkedIn](https://www.linkedin.com/in/ved-piyush)