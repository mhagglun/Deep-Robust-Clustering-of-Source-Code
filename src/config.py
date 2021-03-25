import torch

DATASET = 'dataset'

LAMBDA = 0.1 # Regularization parameter
TEMPERATURE = 0.5

EMBEDDING_DIM = 128
TARGET_DIM = 5
CODE_VECTOR_DIM = TARGET_DIM
DROPOUT = 0.25
BATCH_SIZE = 64
MAX_LENGTH = 200
EPOCHS = 30
SAVE_EVERY = 3000
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')