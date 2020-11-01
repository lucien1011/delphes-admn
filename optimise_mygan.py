import os,pickle,numpy,importlib,sys,shutil

import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from KerasSimpleTrainer.MiniBatchTrainer import MiniBatchTrainer
from PyUtilities.mkdir_p import mkdir_p

from mygan_builder import build_model

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
discriminator,generator,x_input_layer,condition_input_layer = build_model(cfg)

# __________________________________________________________________________________________ ||
batch_trainer = MiniBatchTrainer()
bce = tf.keras.losses.BinaryCrossentropy()
disc_optimizer = tf.keras.optimizers.Adam(cfg.disc_learning_rate,cfg.disc_beta_1,cfg.disc_beta_2,cfg.disc_epsilon,)
gen_optimizer = tf.keras.optimizers.Adam(cfg.gen_learning_rate,cfg.gen_beta_1,cfg.gen_beta_2,cfg.gen_epsilon,)
for i_epoch in range(cfg.n_epoch): 
    
    y_real_train = numpy.ones((cfg.batch_size,1))
    y_fake_train = numpy.zeros((cfg.batch_size,1))

    for _ in range(cfg.n_real_disc_epoch):
        idx_train = numpy.random.randint(0, df_x.shape[0], cfg.batch_size)
        x_real_train = df_x.iloc[idx_train].to_numpy()
        condition_real_train = df_condition.iloc[idx_train].to_numpy()

        with tf.GradientTape() as real_disc_tape:
            real_disc_output = discriminator([x_real_train,condition_real_train,],)
            real_loss_disc = bce(y_real_train, real_disc_output)
        real_grads = real_disc_tape.gradient(real_loss_disc, discriminator.trainable_weights)
        disc_optimizer.apply_gradients(zip(real_grads, discriminator.trainable_weights))
   
    for _ in range(cfg.n_fake_disc_epoch):
        condition_fake_train = condition_real_train

        with tf.GradientTape() as fake_disc_tape:
            fake_disc_output = discriminator([generator(condition_fake_train),condition_fake_train,],)
            fake_loss_disc = bce(y_fake_train, fake_disc_output)
        fake_grads = fake_disc_tape.gradient(fake_loss_disc, discriminator.trainable_weights)
        disc_optimizer.apply_gradients(zip(fake_grads, discriminator.trainable_weights))

    batch_trainer.save_grad(real_grads,os.path.join(cfg.output_path,"disc_real_grad.pkl"),n_per_point=cfg.save_per_point,) 
    batch_trainer.save_grad(fake_grads,os.path.join(cfg.output_path,"disc_fake_grad.pkl"),n_per_point=cfg.save_per_point,) 
    batch_trainer.add_loss("real disc loss",real_loss_disc)
    batch_trainer.add_loss("fake disc loss",fake_loss_disc)
    
    with tf.GradientTape() as gen_tape:
        gen_output = generator(condition_fake_train)
        logits = discriminator([gen_output,condition_fake_train])
        loss_gen = bce(y_real_train,logits)
    grads = gen_tape.gradient(loss_gen, generator.trainable_weights)
    gen_optimizer.apply_gradients(zip(grads, generator.trainable_weights))
    batch_trainer.add_loss("gen loss",loss_gen)
    batch_trainer.save_grad(grads,os.path.join(cfg.output_path,"gen_grad.pkl"),n_per_point=cfg.save_per_point,)

    batch_trainer.add_epoch()
    batch_trainer.print_loss(cfg.print_per_point)
    
    batch_trainer.make_history_plot(
            os.path.join(cfg.output_path,"loss.png"),
            log_scale=True,
            )

    batch_trainer.save_history(os.path.join(cfg.output_path,"history.p"),)
   
    _ = generator.predict(condition_fake_train)
    #batch_trainer.save(generator,os.path.join(cfg.output_path,cfg.gen_model_name),n_per_point=cfg.save_per_point,)
    batch_trainer.save_weights(generator,os.path.join(cfg.output_path,cfg.gen_model_name),n_per_point=cfg.save_per_point,)

    if i_epoch % cfg.plot_per_point == 0 and i_epoch != 0:
        idx_train = numpy.random.randint(0, df_x.shape[0], cfg.plot_size)
        condition_real_plot = df_condition.iloc[idx_train].to_numpy()
        gen_vars = generator.predict(condition_real_plot)
        plot_dir = os.path.join(cfg.output_path,cfg.plot_folder_name+"_"+str(i_epoch)+"/")
        mkdir_p(plot_dir)
        for var in cfg.var_list:
            plt.clf()
            idx_train = numpy.random.randint(0, df_x.shape[0],cfg.plot_size)
            df_plot = numpy.float64(df_x.iloc[idx_train,var.index])
            plt.hist(gen_vars[:,var.index],bins=cfg.plot_bins,alpha=0.5,density=1.,label="gan",range=var.range,)
            plt.hist(df_plot,bins=cfg.plot_bins,alpha=0.5,density=1.,label="mc",range=var.range,)
            plt.legend(loc="best")
            plt.savefig(os.path.join(plot_dir,"plot_"+var.name+".png"))  

batch_trainer.save_history(os.path.join(cfg.output_path,"history.p"),)
