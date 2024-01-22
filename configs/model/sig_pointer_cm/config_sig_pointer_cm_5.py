num_encoder_layers = 5  # Number of encoder layers
num_decoder_layers = 5  # Number of decoder layers
emb_size = 279  # Embedding size of the encoder
nhead = 9  # Number of attention heads
ffn_hid_dim = 512  # Sublayer feedforward dimension
batch_size = 350  # Batch size
num_epochs = 100  # Number of training epochs
lr = 0.0001  # Learning rate
betas = (0.9, 0.98)  # Adam optimizer beta parameters
eps = 1e-9  # Adam optimizer eps parameter
dropout = 0.2  # Dropout

save_model_dir = "models/sig_pointer_cm/"  # Output dir of model
early_stopping_wait = 100  # Wait for N epochs for improvement
early_stopping_delta = 0.  # Minimal accepted improvement

load_checkpoint_path = "models/sig_pointer_cm/sig_pointer_cm_5.pth"

model_name = "sig_pointer_cm_5"  # Name of model (train: write to, predict: load from)
encoder_memory_transformation = "concatenate"  # Encoder strategy: projection, concatenate, None
