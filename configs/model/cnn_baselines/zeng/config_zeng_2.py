num_epochs = 100  # Number of training epochs
lr = 0.001  # learning rate (Zeng et al)
batch_size = 128  # batch size (Zeng et al)

save_model_dir = "models/zeng/"  # Output dir of model
early_stopping_wait = 20  # Wait for N epochs for improvement
early_stopping_delta = 0.  # Minimal accepted improvement

load_checkpoint_path = "models/zeng/zeng_2.pth"  # Path to pretrained weights or None

model_name = "zeng_2"  # Name of model (train: write to, predict: load from)
