dataset_config = {
    'split': 0.9, # training: testing = 9: 1 
    'num_workers': 8, # threads for dataloader
    'batch_size': 1, # batch size (it should be noted that an entry in a batch contains the trajectory of an entire game)
}