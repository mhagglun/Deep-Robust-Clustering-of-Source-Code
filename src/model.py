import config
import torch
import torch.nn as nn
import torch.nn.functional as F

class Code2VecEncoder(nn.Module):
    
    def __init__(self, word_vocab_size, path_vocab_size, embedding_dim, code_vector_dim, dropout):
        super(Code2VecEncoder, self).__init__()
        self.word_vocab_size = word_vocab_size
        self.path_vocab_size = path_vocab_size
        self.embedding_dim = embedding_dim
        self.code_vector_dim = code_vector_dim
        self.word_embedding = nn.Embedding(word_vocab_size, embedding_dim)
        self.path_embedding = nn.Embedding(path_vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(3 * embedding_dim, code_vector_dim, bias=False)
        attention_weights = torch.nn.init.uniform_(torch.empty(code_vector_dim, 1, dtype=torch.float32, requires_grad=True))
        self.attention_weights = nn.parameter.Parameter(attention_weights, requires_grad=True)

    def extract_features(self, x_s, path, x_t):
        #Shape of x_s, path, x_t is [batch size, max length]
        batch_size = x_s.shape[0]

        # Look up path and token embeddings
        embedded_x_s = self.word_embedding(x_s)
        embedded_path = self.path_embedding(path)
        embedded_x_t = self.word_embedding(x_t)

        # Concatenate embeddings
        context_vector = torch.cat((embedded_x_s, embedded_path, embedded_x_t), dim=2) # [batch size, max length, embedding dim * 3 dim]

        # Pass the path context to fully connected layer
        combined_context_vector = self.fc(self.dropout(context_vector)) # [batch size, max length, code vector dim]
        combined_context_vector = torch.tanh(combined_context_vector)

        # Apply attention weights
        attention_weights = self.attention_weights.repeat(batch_size, 1, 1)
        attention_weights = torch.bmm(combined_context_vector, attention_weights)  # [batch size, max length, 1]
        
        # Create padding mask
        mask = torch.zeros((batch_size, config.MAX_LENGTH, 1)).to(config.DEVICE)
        attention_weights += mask
        attention_weights = F.softmax(attention_weights, dim=1)  # Apply activation function

        # Obtain code vector by elementwise multiplication with the combined context vectors and the attention weights, then reduce to single code vector
        combined_vectors = torch.mul(combined_context_vector, attention_weights.expand_as(combined_context_vector))  # [batch size, max length, code vector dim]
        code_vector = torch.sum(combined_vectors, dim=1)

        return code_vector, attention_weights

    def forward(self, x1, x2):
        # Extract features for sample
        x_s1, path1, x_t1 = x1
        cv1, aw1 = self.extract_features(x_s1, path1, x_t1)

        # Extract features for augmented sample
        x_s2, path2, x_t2 = x2
        cv2, aw2 = self.extract_features(x_s2, path2, x_t2)
        return cv1, cv2

class Code2Vec(nn.Module):
    
    def __init__(self, encoder, target_dim):
        super(Code2Vec, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.code_vector_dim, target_dim)

    def forward(self, x1, x2):
        code_vector_1, code_vector_2 = self.encoder(x1, x2)
        assignment_prob_1 = self.classifier(code_vector_1)  # [batch size, target_dim]
        assignment_prob_2 = self.classifier(code_vector_2)
        return assignment_prob_1, code_vector_1, assignment_prob_2, code_vector_2

    def extract_features(self, x_s, path, x_t):
        code_vector, _ = self.encoder.extract_features(x_s, path, x_t)
        assignment_probability = self.classifier(code_vector)
        return assignment_probability, code_vector
