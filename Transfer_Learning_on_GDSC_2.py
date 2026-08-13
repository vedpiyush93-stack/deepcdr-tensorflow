#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
import improve_utils as iu


# In[2]:


cancer_gen_expr_model = tf.keras.models.load_model("Models//cancer_gen_expr_model")
cancer_gen_mut_model = tf.keras.models.load_model("Models//cancer_gen_mut_model")
cancer_dna_methy_model = tf.keras.models.load_model("Models//cancer_dna_methy_model")
cancer_gen_cnv_model = tf.keras.models.load_model("Models//cancer_gen_cnv_model")


# In[3]:


import pickle


# In[4]:


import numpy as np


# In[5]:


with open("csa_data//drug_features.pickle", "rb") as f:
    dict_features = pickle.load(f)


# In[6]:


with open("csa_data//norm_adj_mat.pickle", "rb") as f:
    dict_adj_mat = pickle.load(f)


# In[7]:


with open("csa_data//combo_gen_expr_cnv_norm.pickle", "rb") as f:
    combo_gen_expr_cnv_norm = pickle.load( f)


# In[8]:


common_achs = list(combo_gen_expr_cnv_norm.keys())


# In[9]:


from tensorflow.keras import backend as K


# In[10]:


check = tf.keras.models.load_model("Models/combo_cdr_gcn_on_ctrpv2_0_drop_on_train_and_pred_sample_035")
check_full = tf.keras.models.load_model("Models/combo_cdr_gcn_on_ctrpv2_0_drop_on_train_and_pred")


# In[11]:


def get_features(train_keep1):
    train_gcn_feats = []
    train_adj_list = []
    for drug_id in train_keep1["Drug_ID"].values:
        train_gcn_feats.append(dict_features[drug_id])
        train_adj_list.append(dict_adj_mat[drug_id])
    train_gcn_feats = np.array(train_gcn_feats).astype("float32")
    train_adj_list = np.array(train_adj_list).astype("float32")
    train_gcn_feats_omics = []
    for cnc_id in train_keep1["Cell_Line"].values:
        train_gcn_feats_omics.append(combo_gen_expr_cnv_norm[cnc_id])
    train_gcn_feats_omics = np.array(train_gcn_feats_omics).astype("float32")
    l2, l1 = train_keep1["Cell_Line"].values.reshape(-1,1), train_keep1["Cell_Line"].values.reshape(-1,1)
    y_train = train_keep1["AUC"].values.reshape(-1,1)
    return [[train_gcn_feats, train_adj_list, train_gcn_feats_omics, l2, l1], y_train]


# In[12]:


gdsc_v2_train = iu.load_single_drug_response_data_v2(source = 'GDSCv2_all', split_file_name='GDSCv2_split_0_train.txt', y_col_name='AUC')[["improve_sample_id", "improve_chem_id", "auc"]]
gdsc_v2_train.columns = ["Cell_Line", "Drug_ID", "AUC"]
# gdsc_v1_all = gdsc_v1_all.sample(frac = 0.15)


# In[13]:


features_gdsc_v2_train, target_gdsc_v2_train = get_features(gdsc_v2_train)


# In[14]:


gdsc_v2_valid = iu.load_single_drug_response_data_v2(source = 'GDSCv2_all', split_file_name='GDSCv2_split_0_val.txt', y_col_name='AUC')[["improve_sample_id", "improve_chem_id", "auc"]]
gdsc_v2_valid.columns = ["Cell_Line", "Drug_ID", "AUC"]
# gdsc_v1_all = gdsc_v1_all.sample(frac = 0.15)


# In[15]:


features_gdsc_v2_valid, target_gdsc_v2_valid = get_features(gdsc_v2_valid)


# In[16]:


gdsc_v2_test = iu.load_single_drug_response_data_v2(source = 'GDSCv2_all', split_file_name='GDSCv2_split_0_test.txt', y_col_name='AUC')[["improve_sample_id", "improve_chem_id", "auc"]]
gdsc_v2_test.columns = ["Cell_Line", "Drug_ID", "AUC"]
# gdsc_v1_all = gdsc_v1_all.sample(frac = 0.15)


# In[17]:


features_gdsc_v2_test, target_gdsc_v2_test = get_features(gdsc_v2_test)


# In[ ]:


check.fit([features_gdsc_v2_train[0], features_gdsc_v2_train[1], features_gdsc_v2_train[2], features_gdsc_v2_train[3], features_gdsc_v2_train[4]],
          target_gdsc_v2_train,
         validation_data = ([features_gdsc_v2_valid[0], features_gdsc_v2_valid[1], features_gdsc_v2_valid[2], features_gdsc_v2_valid[3], features_gdsc_v2_valid[4]],
          target_gdsc_v2_valid), 
         batch_size = 64, epochs = 10000, 
         callbacks = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 10, restore_best_weights=True, 
                                                     mode = "min"), shuffle = True, verbose = 1, 
         validation_batch_size = 64)


# In[ ]:


check.save("Models/combo_cdr_gcn_on_ctrpv2_0_drop_on_train_and_pred_fine_tune_gdsc_v2")


# In[ ]:


check_full = tf.keras.models.load_model("Models/combo_cdr_gcn_on_ctrpv2_0_drop_on_train_and_pred_fine_tune_gdsc_v2")


# In[ ]:


# list_df = [gdsc_v2_all[i:i+25000] for i in range(0,gdsc_v2_all.shape[0],25000)]


# In[ ]:


# preds = []
# for i in range(0, 100): 
#     preds.append(get_one_preds())
#     K.clear_session()
#     gc.collect()


# In[ ]:


# import gc


# In[ ]:


# actual = []
# preds = []
# for batch in list_df: 
#     features_gdsc_v2_all, target_gdsc_v2_all = get_features(batch)
#     with tf.device('/gpu:0'):
#         preds_gdsc_v2_all = check_full.predict(features_gdsc_v2_all, verbose = 1, batch_size = 64)
#     actual.append(target_gdsc_v2_all)
#     preds.append(preds_gdsc_v2_all)
#     K.clear_session()
#     gc.collect()


# In[ ]:


# actual = [inner for item in actual for inner in item]


# In[ ]:


# preds = [inner for item in preds for inner in item]


# In[ ]:


# _, tar = get_features(list_df[0])


# In[ ]:


# list_df[0]


# In[ ]:


# features_gdsc_v1_all, target_gdsc_v1_all = get_features(gdsc_v1_all)


# In[ ]:


features_gdsc_v2_test, target_gdsc_v2_test


# In[ ]:


preds_gdsc_v1_all = check_full.predict(features_gdsc_v2_test, verbose = 1, batch_size = 64)


# In[ ]:


from scipy.stats import pearsonr


# In[ ]:


r = pearsonr(preds_gdsc_v1_all[:,0], target_gdsc_v2_test[:,0])


# In[ ]:


R_sq = r[0]**2


# In[ ]:


print(r, flush = True)


# In[ ]:


print(R_sq, flush = True)

