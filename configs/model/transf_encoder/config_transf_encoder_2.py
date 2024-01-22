num_encoder_layers = 12  # Number of encoder layers
num_decoder_layers = 0  # Number of decoder layers
emb_size = 279  # Embedding size of the encoder
nhead = 9  # Number of attention heads
ffn_hid_dim = 2048  # Sublayer feedforward dimension
batch_size = 64  # Batch size
num_epochs = 100  # Number of training epochs
lr = 0.0001  # Learning rate
betas = (0.9, 0.98)  # Adam optimizer beta parameters
eps = 1e-9  # Adam optimizer eps parameter
dropout = 0.1  # Dropout

save_model_dir = "models/transformer_encoder"  # Output dir of model
early_stopping_wait = 20  # Wait for N epochs for improvement
early_stopping_delta = 0.  # Minimal accepted improvement

load_checkpoint_path = "models/transformer_encoder/transformer_encoder_2.pth"  # Path to pretrained weights or None

model_name = "transformer_encoder_2"  # Name of model (train: write to, predict: load from)
encoder_memory_transformation = "concatenate"  # Encoder strategy: projection, concatenate, None
