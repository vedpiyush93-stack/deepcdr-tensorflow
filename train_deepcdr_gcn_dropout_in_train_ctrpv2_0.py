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


cancer_gen_expr_model.trainable = False
cancer_gen_mut_model.trainable = False
cancer_dna_methy_model.trainable = False
cancer_gen_cnv_model.trainable = False


# In[4]:


import pickle


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


# combo_gen_expr_cnv_norm


# In[10]:


import pandas as pd


# In[11]:


pubchem_to_drugs_df = pd.read_csv('csa_data/1.Drug_listMon Jun 24 09_00_55 2019.csv')


# In[12]:


# pubchem_to_drugs_df


# In[13]:


# pubchem_to_drugs_df


# In[14]:


pubchem_to_drugs_df = pubchem_to_drugs_df[["drug_id", "PubCHEM"]]


# In[15]:


import numpy as np


# In[16]:


pubchem_to_drugs_df["PubCHEM"] = [int(val) if str(val).isdigit() else np.nan for val in pubchem_to_drugs_df["PubCHEM"] ]


# In[17]:


pubchem_to_drugs_df = pubchem_to_drugs_df.dropna()


# In[18]:


pubchem_to_drugs_df["Drug_ID"] = pubchem_to_drugs_df["drug_id"].astype(str)


# In[19]:


pubchem_to_drugs_df["Drug_ID"] = ["Drug_" + str(i) for i in pubchem_to_drugs_df["drug_id"].values]


# In[20]:


pubchem_to_drugs_df.head()


# In[21]:


train_keep = iu.load_single_drug_response_data_v2(source = 'CTRPv2', split_file_name='CTRPv2_split_0_train.txt', y_col_name='auc')[['improve_sample_id', 'improve_chem_id', 'auc']]


# In[22]:


# train_keep = train_keep.sample(frac = 0.25)


# In[23]:


test_keep = iu.load_single_drug_response_data_v2(source = 'CTRPv2', split_file_name='CTRPv2_split_0_val.txt', y_col_name='auc')[['improve_sample_id', 'improve_chem_id', 'auc']]


# In[24]:


train_keep.head()


# In[25]:


train_keep.columns = ["Cell_Line", "Drug_ID", "AUC"]


# In[26]:


test_keep.columns = ["Cell_Line", "Drug_ID", "AUC"]


# In[27]:


samp_drug = test_keep["Drug_ID"].unique()[-1]


# In[28]:


samp_ach = np.array(test_keep["Cell_Line"].unique()[-1])


# In[29]:


valid_keep = iu.load_single_drug_response_data_v2(source = 'CTRPv2', split_file_name='CTRPv2_split_0_test.txt', y_col_name='auc')[['improve_sample_id', 'improve_chem_id', 'auc']]


# In[30]:


valid_keep.columns = ["Cell_Line", "Drug_ID", "AUC"]


# In[31]:


# train_keep.to_csv("csa_data/dualgcndeepcdr_combo_train.csv", index = False)


# In[32]:


# test_keep.to_csv("csa_data/dualgcndeepcdr_combo_test.csv", index = False)


# In[33]:


# train_methy = cancer_dna_methy_model.predict(train_keep1["Cell_Line"].values, batch_size = 512)


# In[34]:


# train_methy.shape


# In[35]:


train_gcn_feats = []
train_adj_list = []
for drug_id in train_keep["Drug_ID"].values:
    train_gcn_feats.append(dict_features[drug_id])
    train_adj_list.append(dict_adj_mat[drug_id])


# In[36]:


valid_gcn_feats = []
valid_adj_list = []
for drug_id in test_keep["Drug_ID"].values:
    valid_gcn_feats.append(dict_features[drug_id])
    valid_adj_list.append(dict_adj_mat[drug_id])


# In[37]:


train_gcn_feats = np.array(train_gcn_feats).astype("float32")
valid_gcn_feats = np.array(valid_gcn_feats).astype("float32")


# In[38]:


valid_gcn_feats.shape


# In[39]:


train_adj_list = np.array(train_adj_list).astype("float32")
valid_adj_list = np.array(valid_adj_list).astype("float32")


# In[40]:


valid_adj_list.shape


# In[41]:


# combo_gen_expr_cnv_norm


# In[42]:


# valid_gcn_feats_omics


# In[43]:


train_gcn_feats_omics = []
for cnc_id in train_keep["Cell_Line"].values:
    train_gcn_feats_omics.append(combo_gen_expr_cnv_norm[cnc_id])


# In[44]:


train_gcn_feats_omics = np.array(train_gcn_feats_omics).astype("float32")


# In[45]:


train_gcn_feats_omics.shape


# In[46]:


valid_gcn_feats_omics = []
for cnc_id in test_keep["Cell_Line"].values:
    valid_gcn_feats_omics.append(combo_gen_expr_cnv_norm[cnc_id])


# In[47]:


valid_gcn_feats_omics = np.array(valid_gcn_feats_omics).astype("float32")


# In[48]:


valid_gcn_feats_omics.shape


# In[49]:


# train_methy = cancer_dna_methy_model(train_keep1["Cell_Line"].values)


# In[50]:


# cancer_gen_expr_model(samp_ach).numpy().shape[0]


# In[51]:


from tensorflow.keras import backend as K


# In[52]:


# cancer_gen_expr_model(train_keep1["Cell_Line"].values[:5])


# In[53]:


training = False
dropout1 = 0.20
dropout2 = 0.30


# In[54]:


## get the model architecture
def deepcdrgcn(training = training, dropout1 = dropout1, dropout2 = dropout2):
    
    input_gcn_features = tf.keras.layers.Input(shape = (dict_features[samp_drug].shape[0], 75))
    input_norm_adj_mat = tf.keras.layers.Input(shape = (dict_adj_mat[samp_drug].shape[0], dict_adj_mat[samp_drug].shape[0]))
    mult_1 = tf.keras.layers.Dot(1)([input_norm_adj_mat, input_gcn_features])
    dense_layer_gcn = tf.keras.layers.Dense(256, activation = "relu")
    dense_out = dense_layer_gcn(mult_1)
    dense_out = tf.keras.layers.BatchNormalization()(dense_out)
    dense_out = tf.keras.layers.Dropout(dropout1)(dense_out, training = training)
    mult_2 = tf.keras.layers.Dot(1)([input_norm_adj_mat, dense_out])
    dense_layer_gcn = tf.keras.layers.Dense(256, activation = "relu")
    dense_out = dense_layer_gcn(mult_2)
    dense_out = tf.keras.layers.BatchNormalization()(dense_out)
    dense_out = tf.keras.layers.Dropout(dropout1)(dense_out, training = training)

    dense_layer_gcn = tf.keras.layers.Dense(100, activation = "relu")
    mult_3 = tf.keras.layers.Dot(1)([input_norm_adj_mat, dense_out])
    dense_out = dense_layer_gcn(mult_3)
    dense_out = tf.keras.layers.BatchNormalization()(dense_out)
    dense_out = tf.keras.layers.Dropout(dropout1)(dense_out, training = training)

    dense_out = tf.keras.layers.GlobalAvgPool1D()(dense_out)
    
    input_gen_expr = tf.keras.layers.Input(shape = (cancer_gen_expr_model(samp_ach).numpy().shape[0],2))
    
    l1 = tf.keras.layers.Dense(32)(input_gen_expr)
    l1 = tf.keras.layers.Dropout(dropout1)(l1, training = training)
    l2 = tf.keras.layers.Dense(128)(l1)
    l2 = tf.keras.layers.Dropout(dropout1)(l2, training = training)
    
    dense_layer_gcn1 = tf.keras.layers.Dense(256, activation = "relu")
    dense_out1 = dense_layer_gcn1(l2)
    dense_out1 = tf.keras.layers.BatchNormalization()(dense_out1)
    dense_out1 = tf.keras.layers.Dropout(dropout1)(dense_out1, training = training)
# mult_21 = tf.keras.layers.Dot(1)([const_input, dense_out1])
    dense_layer_gcn1 = tf.keras.layers.Dense(256, activation = "relu")
    dense_out1 = dense_layer_gcn1(dense_out1)
    dense_out1 = tf.keras.layers.BatchNormalization()(dense_out1)
    dense_out1 = tf.keras.layers.Dropout(dropout1)(dense_out1, training = training)
    dense_layer_gcn1 = tf.keras.layers.Dense(256, activation = "relu")
    dense_out1 = dense_layer_gcn1(dense_out1)
    dense_out1 = tf.keras.layers.BatchNormalization()(dense_out1)
    dense_out1 = tf.keras.layers.Dropout(dropout1)(dense_out1, training = training)
    dense_layer_gcn1 = tf.keras.layers.Dense(256, activation = "relu")
    dense_out1 = dense_layer_gcn1(dense_out1)
    dense_out1 = tf.keras.layers.BatchNormalization()(dense_out1)
    dense_out1 = tf.keras.layers.Dropout(dropout1)(dense_out1, training = training)
    dense_out1 = tf.keras.layers.GlobalAvgPool1D()(dense_out1)
    
    
    input_gen_methy1 = tf.keras.layers.Input(shape = (1,), dtype = tf.string)
    # cancer_dna_methy_model.trainable
    input_gen_methy = cancer_dna_methy_model(input_gen_methy1)
    input_gen_methy.trainable = False
    gen_methy_layer = tf.keras.layers.Dense(64, activation = "tanh")
    
    gen_methy_emb = gen_methy_layer(input_gen_methy)
    gen_methy_emb = tf.keras.layers.BatchNormalization()(gen_methy_emb)
    gen_methy_emb = tf.keras.layers.Dropout(dropout1)(gen_methy_emb, training = training)
    gen_methy_layer = tf.keras.layers.Dense(64, activation = "relu")
    gen_methy_emb = gen_methy_layer(gen_methy_emb)
    
    
    input_gen_mut1 = tf.keras.layers.Input(shape = (1,), dtype = tf.string)
    input_gen_mut = cancer_gen_mut_model(input_gen_mut1)
    input_gen_mut.trainable = False
    
    reshape_gen_mut = tf.keras.layers.Reshape((1, cancer_gen_mut_model(samp_ach).numpy().shape[0], 1))
    reshape_gen_mut = reshape_gen_mut(input_gen_mut)
    gen_mut_layer = tf.keras.layers.Conv2D(50, (1, 700), strides=5, activation = "tanh")
    gen_mut_emb = gen_mut_layer(reshape_gen_mut)
    pool_layer = tf.keras.layers.MaxPooling2D((1,5))
    pool_out = pool_layer(gen_mut_emb)
    gen_mut_layer = tf.keras.layers.Conv2D(30, (1, 5), strides=2, activation = "relu")
    gen_mut_emb = gen_mut_layer(pool_out)
    pool_layer = tf.keras.layers.MaxPooling2D((1,10))
    pool_out = pool_layer(gen_mut_emb)
    flatten_layer = tf.keras.layers.Flatten()
    flatten_out = flatten_layer(pool_out)
    all_omics = tf.keras.layers.Concatenate()([gen_methy_emb, flatten_out, dense_out1, dense_out])
    x = tf.keras.layers.Dense(300,activation = 'tanh')(all_omics)
    x = tf.keras.layers.Dropout(dropout1)(x, training = training)
    x = tf.keras.layers.Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
    x = tf.keras.layers.Lambda(lambda x: K.expand_dims(x,axis=1))(x)
    x = tf.keras.layers.Conv2D(filters=30, kernel_size=(1,150),strides=(1, 1), activation = 'relu',padding='valid')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1,2))(x)
    x = tf.keras.layers.Conv2D(filters=10, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1,3))(x)
    x = tf.keras.layers.Conv2D(filters=5, kernel_size=(1,5),strides=(1, 1), activation = 'relu',padding='valid')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(1,3))(x)
    x = tf.keras.layers.Dropout(dropout1)(x, training = training)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(dropout2)(x, training = training)
    final_out_layer = tf.keras.layers.Dense(1, activation = "sigmoid")
    final_out = final_out_layer(x)
    simplecdr = tf.keras.models.Model([input_gcn_features, input_norm_adj_mat, input_gen_expr,
                                   input_gen_methy1, input_gen_mut1], final_out)
    simplecdr.compile(loss = tf.keras.losses.MeanSquaredError(), 
                      # optimizer = tf.keras.optimizers.Adam(lr=1e-3),
                    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), 
                    metrics = [tf.keras.metrics.RootMeanSquaredError()])
    
    return simplecdr


# In[55]:


check = deepcdrgcn(training = training, dropout1 = dropout1, dropout2 = dropout2)


# In[56]:


check.summary()


# In[57]:


# [layer.name for layer in check.layers]


# In[58]:


plt.hist( train_keep["AUC"].values.reshape(-1,1))
plt.show()


# In[ ]:


check.fit([train_gcn_feats, train_adj_list, train_gcn_feats_omics, train_keep["Cell_Line"].values.reshape(-1,1), train_keep["Cell_Line"].values.reshape(-1,1)],
          train_keep["AUC"].values.reshape(-1,1),
         validation_data = ([valid_gcn_feats, valid_adj_list, valid_gcn_feats_omics, test_keep["Cell_Line"].values.reshape(-1,1), test_keep["Cell_Line"].values.reshape(-1,1)],
          test_keep["AUC"].values.reshape(-1,1)), 
         batch_size = 256, epochs = 10000, 
         callbacks = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 20, restore_best_weights=True, 
                                                     mode = "min"), shuffle = True, verbose = 1, 
         validation_batch_size = 256)


# In[ ]:


check.save("Models/combo_cdr_gcn_on_ctrpv2_0_drop_on_train_and_pred")


# In[ ]:


check = tf.keras.models.load_model("Models/combo_cdr_gcn_on_ctrpv2_0_drop_on_train_and_pred")


# In[ ]:


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


# In[ ]:


# ccle_all


# In[ ]:


# ccle_all = iu.load_single_drug_response_data_v2(source = 'CCLE_all', split_file_name='CCLE_all.txt', y_col_name='AUC')[["improve_sample_id", "improve_chem_id", "auc"]]


# In[ ]:


# ccle_all = ccle_all.dropna()


# In[ ]:


valid_keep["AUC"].describe()


# In[ ]:


# ccle_all.shape


# In[ ]:


# ccle_all_cutoff = np.percentile(ccle_all["ic50"].values, q= (2.5))


# In[ ]:


# ccle_all = ccle_all[ccle_all["ic50"] >= ccle_all_cutoff]


# In[ ]:


# ccle_all.shape


# In[ ]:


# plt.hist(ccle_all["ic50"].values)


# In[ ]:


# CTRPv2_all = iu.load_single_drug_response_data_v2(source = 'CTRPv2_all', split_file_name='CTRPv2_all.txt', y_col_name='IC50')[["improve_sample_id", "improve_chem_id", "ic50"]]


# In[ ]:


# CTRPv2_all = CTRPv2_all.dropna()


# In[ ]:


# CTRPv2_all_cutoff = np.percentile(CTRPv2_all["ic50"].values, q= (2.5))


# In[ ]:


# CTRPv2_all = CTRPv2_all[CTRPv2_all["ic50"] >= CTRPv2_all_cutoff]


# In[ ]:


# CTRPv2_all.shape


# In[ ]:


# plt.hist(CTRPv2_all["ic50"].values)
# plt.show()


# In[ ]:


# CTRPv2_all = CTRPv2_all.sample(frac = 0.2)


# In[ ]:


# CTRPv2_all.shape


# In[ ]:


# gsci_all = iu.load_single_drug_response_data_v2(source = 'gsci_all', split_file_name='gCSI_all.txt', y_col_name='IC50')[["improve_sample_id", "improve_chem_id", "ic50"]]


# In[ ]:


# gsci_all = gsci_all.dropna()


# In[ ]:


# gdsc_v1_all = iu.load_single_drug_response_data_v2(source = 'GDSCv1_all', split_file_name='GDSCv1_all.txt', y_col_name='IC50')[["improve_sample_id", "improve_chem_id", "ic50"]]


# In[ ]:


# gdsc_v1_all = gdsc_v1_all.dropna()


# In[ ]:


# gdsc_v2_all = iu.load_single_drug_response_data_v2(source = 'GDSCv2_all', split_file_name='GDSCv2_all.txt', y_col_name='IC50')[["improve_sample_id", "improve_chem_id", "ic50"]]


# In[ ]:


# gdsc_v2_all = gdsc_v2_all.dropna()


# In[ ]:


# combo_keep1 = pd.concat([train_keep, test_keep], ignore_index = True)


# In[ ]:


# ccle_all.columns


# In[ ]:


# CTRPv2_all.columns = combo_keep1.columns


# In[ ]:


# CTRPv2_all["IC_50"] = CTRPv2_all["IC_50"].astype("float32")


# In[ ]:


# CTRPv2_all.head()


# In[ ]:


# ccle_all_right = pd.merge(combo_keep1,CTRPv2_all, how='outer', indicator=True)
# to_pred_ccle = ccle_all_right[ccle_all_right['_merge'] == 'right_only']


# In[ ]:


valid_keep.head()


# In[ ]:


plt.hist(train_keep['AUC'].values)


# In[ ]:


plt.hist(valid_keep['AUC'].values)


# In[ ]:


features, target = get_features(valid_keep)


# In[ ]:


# features, target = features.astype("float32"), target.astype("float32")


# In[ ]:


target.shape


# In[ ]:


features[0].shape


# In[ ]:


preds = check.predict(features, verbose = 1, batch_size = 64)


# In[ ]:


plt.hist(preds)


# In[ ]:


# check.evaluate(features,target, verbose = 1, batch_size = 64)


# In[ ]:


# features[0].shape


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


# target


# In[ ]:


preds.shape


# In[ ]:


np.sqrt(mean_squared_error(preds, target))


# In[ ]:


from scipy.stats import pearsonr


# In[ ]:


pearsonr(preds[:,0], target[:,0])


# In[ ]:


print(pearsonr(preds[:,0], target[:,0]), flush = True)

