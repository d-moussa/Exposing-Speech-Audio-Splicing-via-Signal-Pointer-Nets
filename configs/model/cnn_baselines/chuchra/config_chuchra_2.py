num_epochs = 100  # Number of training epochs
lr = 0.001  # learning rate (Chuchra et al)
batch_size = 64  # batch size (Chuchra et al)

save_model_dir = "models/chuchra/"  # Output dir of model
early_stopping_wait = 20  # Wait for N epochs for improvement
early_stopping_delta = 0.  # Minimal accepted improvement

load_checkpoint_path = "models/chuchra/chuchra_2.pth"  # Path to pretrained weights or None

model_name = "chuchra_2"  # Name of model (train: write to, predict: load from)
