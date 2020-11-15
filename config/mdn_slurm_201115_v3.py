from PyUtilities.Config import Config

cfg = Config("mygan",
        # data detail
        job_name = "mdn_hpg_20115_v3",
        input_csv_path = "/cmsuf/data/store/user/t2/users/klo/Delphes/ALP_HToZaTo2l2g_M1/2020-06-02/MuonTreeProducer/csv/HToZaTo2l2g_M1/input.csv",
        preprocess_df_path = "/cmsuf/data/store/user/t2/users/klo/HEP-ML-Tools/Delphes/bigan_slurm_20200805_v1/preprocess_df.csv",
        output_path = "/cmsuf/data/store/user/t2/users/klo/delphes-admn/mdn_slurm_201115_v3/",
        log_output_path = "/cmsuf/data/store/user/t2/users/klo/delphes-admn/mdn_slurm_201115_v3/",
        drop_column_list = [4,],
        drop_condition_column_list = [4,5,6,],
        
        # training detail
        n_event = 1000000,
        n_epoch = 10000,
        batch_size = 512,
        save_per_point = 100,
        print_per_point = 100,
        saved_model_name = "saved_model",
        use_gpu_verbose = False,
       
        ndf = 1,
        nparam = 9,
        learning_rate=1E-4, 
        beta_1=0.9, 
        beta_2=0.99, 
        epsilon = 1E-9,
        clip_value_min = -0.1,
        clip_value_max = 0.1,

        saved_model_path = "output/mdn_macpro_20115_v2/saved_model",
        )
