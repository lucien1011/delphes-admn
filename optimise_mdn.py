import os,pickle,numpy,importlib,sys,shutil

import tensorflow as tf
import pandas as pd

from KerasSimpleTrainer.MiniBatchTrainer import MiniBatchTrainer
from PyUtilities.mkdir_p import mkdir_p

from model.MDN import MDN

tf.get_logger().setLevel('ERROR')

# __________________________________________________________________________________________ ||
if sys.argv[1].endswith(".py"):
    spec = importlib.util.spec_from_file_location("preprocess_data_cfg", sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.cfg
elif sys.argv[1].endswith(".pkl"):
    cfg = pickle.load(open(sys.argv[1],"rb"))

# __________________________________________________________________________________________ ||
if cfg.use_gpu_verbose:
    tf.debugging.set_log_device_placement(True)

# __________________________________________________________________________________________ ||
mkdir_p(cfg.output_path)
df_x = pd.read_csv(cfg.preprocess_df_path.replace(".csv","_x.csv"),header=None,index_col=None,nrows=cfg.n_event,)
df_condition = pd.read_csv(cfg.preprocess_df_path.replace(".csv","_condition.csv"),header=None,index_col=None,nrows=cfg.n_event,)
try:
    shutil.copy(sys.argv[1],cfg.output_path)
except shutil.SameFileError:
    print("cfg file exists already in target folder, stop copying")

df_x.drop(cfg.drop_column_list,axis=1,inplace=True)
df_condition.drop(cfg.drop_condition_column_list,axis=1,inplace=True)
df_x.drop([0,],inplace=True)
df_condition.drop([0,],inplace=True)

# __________________________________________________________________________________________ ||
model = MDN(df_condition.shape[1],cfg.ndf,df_x.shape[1])

# __________________________________________________________________________________________ ||
batch_trainer = MiniBatchTrainer()
#optimizer = tf.keras.optimizers.Adam(cfg.learning_rate,cfg.beta_1,cfg.beta_2,cfg.epsilon,)
optimizer = tf.keras.optimizers.Adam(cfg.learning_rate,cfg.beta_1,cfg.beta_2,cfg.epsilon,)
for i_epoch in range(cfg.n_epoch): 
    idx_train = numpy.random.randint(0, df_x.shape[0], cfg.batch_size)
    x_train = df_x.iloc[idx_train].to_numpy()
    x_train = numpy.asarray(x_train).astype('float32')
    condition_train = df_condition.iloc[idx_train].to_numpy()
    condition_train = numpy.asarray(condition_train).astype('float32')

    with tf.GradientTape() as tape:
        inputs = model(condition_train)
        
        ll = tf.math.abs(tf.math.log(model.calculate_loss(inputs,x_train)))
        ll = tf.reduce_mean(ll)
    grad = tape.gradient(ll,model.trainable_weights)
    optimizer.apply_gradients(zip(grad,model.trainable_weights))
    batch_trainer.add_loss("ll",ll)
    batch_trainer.add_epoch()
    batch_trainer.print_loss(cfg.print_per_point)
    batch_trainer.make_history_plot(os.path.join(cfg.output_path,"loss.png"),log_scale=True,)

batch_trainer.save_history(os.path.join(cfg.output_path,"history.p"),)
model.save(cfg.output_path+cfg.saved_model_name)
