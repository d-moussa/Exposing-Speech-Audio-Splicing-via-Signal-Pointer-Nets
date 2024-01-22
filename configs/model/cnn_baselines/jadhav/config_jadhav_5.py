num_epochs = 100  # Number of training epochs
lr = 0.001  # learning rate
batch_size = 128  # batch size

save_model_dir = "models/jadhav/"  # Output dir of model
early_stopping_wait = 20  # Wait for N epochs for improvement
early_stopping_delta = 0.  # Minimal accepted improvement

load_checkpoint_path = "models/jadhav/jadhav_5.pth"  # Path to pretrained weights or None

model_name = "jadhav_5"  # Name of model (train: write to, predict: load from)
