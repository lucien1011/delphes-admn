from PyUtilities.Config import Config

cfg = Config("mygan",
        # data detail
        job_name = "mdn_macpro_20114",
        folder_name = "Delphes",
        input_csv_path = "/Users/lucien/Downloads/input_new.csv",
        preprocess_df_path = "/Users/lucien/Downloads/input_new_cleaned.csv",
        output_path = "output/mdn_macpro_20114/",
        drop_column_list = [4,],
        drop_condition_column_list = [4,5,6,],
        
        # training detail
        n_event = 1000000,
        n_epoch = 5000,
        batch_size = 512,
        save_per_point = 100,
        print_per_point = 100,
        saved_model_name = "saved_model",
        use_gpu_verbose = False,
       
        ndf = 9,
        latent_dim = 8,
        learning_rate=1E-3, 
        beta_1=0.5, 
        beta_2=0.99, 
        epsilon = 1E-9,

        saved_model_path = "output/mdn_macpro_20114/saved_model",
        )
