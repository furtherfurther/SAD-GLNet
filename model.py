import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model_gcn import GCN

iemocap_similarity_matrix = torch.tensor([
    [1.0, 0.3, 0.4, 0.2, 0.6, 0.3],  # happy
    [0.3, 1.0, 0.4, 0.7, 0.4, 0.6],  # sad
    [0.2, 0.5, 1.0, 0.4, 0.5, 0.2],  # neutral
    [0.2, 0.7, 0.4, 1.0, 0.3, 0.5],  # angry
    [0.6, 0.4, 0.5, 0.3, 1.0, 0.4],  # excited
    [0.3, 0.8, 0.3, 0.5, 0.4, 1.0]   # frustrated
], dtype=torch.float32)


class SimilarityAwareFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, size_average=True):
        super(SimilarityAwareFocalLoss, self).__init__()
        self.gamma = gamma
        # self.alpha = alpha
        self.alpha = torch.tensor([0.1807,0.1707,0.1585,0.1667,0.1725,0.1506]).cuda()  # Set according to class frequency
        self.size_average = size_average

    def forward(self, logits, labels, similarity_matrix=iemocap_similarity_matrix):
        """
        :param logits: Model predictions
        :param labels: Ground truth labels
        :param similarity_matrix: Similarity matrix between classes
        """
        similarity_matrix = similarity_matrix.to(labels.device)
        labels = labels.view(-1)
        label_onehot = torch.zeros(logits.size(0), logits.size(1)).to(logits.device).scatter_(1, labels.unsqueeze(1), 1)

        log_p = F.log_softmax(logits, dim=-1)
        pt = torch.exp(log_p)
        sub_pt = 1 - pt

        # Class similarity weighting
        similarity_weights = similarity_matrix[labels]  # Get weighting for each class from similarity matrix

        # Calculate Focal Loss and weight by similarity
        focal_loss = -self.alpha * (sub_pt ** self.gamma) * log_p * similarity_weights.unsqueeze(1)

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class SimilarityAwareFocalLoss_2(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, size_average=True):
        super(SimilarityAwareFocalLoss_2, self).__init__()
        self.gamma = gamma
        # self.alpha = alpha
        self.alpha = torch.tensor([0.1, 0.25, 0.25, 0.3, 0.25, 0.3])  # Set according to class frequency
        self.size_average = size_average

    def forward(self, log_pred, target, similarity_matrix=iemocap_similarity_matrix):
        """
        :param log_pred: Model predictions, log probabilities
        :param target: Target probability distribution
        :param similarity_matrix: Similarity matrix between classes
        """
        similarity_matrix = similarity_matrix.to(target.device)
        target = target.view(-1)
        # label_onehot = torch.zeros(log_pred.size(0), log_pred.size(1)).to(log_pred.device).scatter_(1, target.unsqueeze(1), 1)

        pt = torch.exp(log_pred)  # Convert log probabilities to probabilities
        sub_pt = 1 - pt

        # Class similarity weighting
        similarity_weights = similarity_matrix[target.long()]  # Ensure target is long type
        # similarity_weights = similarity_matrix[target]  # Get weighting for each class from similarity matrix

        # Calculate Focal Loss and weight by similarity
        focal_loss = -self.alpha * (sub_pt ** self.gamma) * log_pred * similarity_weights.unsqueeze(1)

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class MaskedKLDivLoss(nn.Module):
    """Calculate KL divergence loss considering a mask to ensure loss is only computed for masked (non-zero) parts"""
    def __init__(self):
        super(MaskedKLDivLoss, self).__init__()
        # reduction='sum' means sum all sample losses when calculating loss
        self.loss = nn.KLDivLoss(reduction='sum')

    def forward(self, log_pred, target, mask):
        """
        :param log_pred: Model output log probabilities
        :param target: Target probability distribution
        :param mask: Boolean mask indicating which data points should be considered in loss calculation
        """
        # Reshape mask into column vector for element-wise multiplication with log_pred and target
        mask_ = mask.view(-1, 1)
        # Calculate loss, but only for positions where mask is True. This is achieved by multiplying log_pred and target with mask_
        # Normalize by dividing total loss by number of True elements in mask
        loss = self.loss(log_pred * mask_, target * mask_) / torch.sum(mask)   
        return loss


class KLDivLoss(nn.Module):
    """Calculate KL divergence loss considering a mask to ensure loss is only computed for masked (non-zero) parts"""
    def __init__(self):
        super(KLDivLoss, self).__init__()
        # reduction='sum' means sum all sample losses when calculating loss
        self.loss = nn.KLDivLoss(reduction='sum')

    def forward(self, log_pred, target):
        """
        :param log_pred: Model output log probabilities
        :param target: Target probability distribution
        :param mask: Boolean mask indicating which data points should be considered in loss calculation
        """
        # Reshape mask into column vector for element-wise multiplication with log_pred and target

        # Calculate loss, but only for positions where mask is True. This is achieved by multiplying log_pred and target with mask_
        # Normalize by dividing total loss by number of True elements in mask
        # print("log_pred.size(0)", log_pred.size(0))
        loss = self.loss(log_pred, target) / log_pred.size(0)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.5, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001

    def forward(self, logits, labels):
        """
        calculates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        labels_length = logits.size(1)
        seq_length = logits.size(0)

        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([seq_length, labels_length]).cuda().scatter_(1, new_label, 1)

        log_p = F.log_softmax(logits, -1)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt) ** self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()

class MaskedNLLLoss(nn.Module):
    """Calculate Negative Log Likelihood Loss considering a mask to ensure loss is only computed for masked (non-zero) parts"""
    def __init__(self, weight=None):
        # Receives optional weight parameter for assigning different weights to different classes
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        # If weight parameter is provided, it will be used to weight the loss
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        """
        :param pred: Model output log probabilities
        :param target: Target class indices
        :param mask: Boolean mask indicating which data points should be considered in loss calculation
        """
        mask_ = mask.view(-1, 1)
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            # Normalize by dividing total loss by sum of weight and mask product
            # In Python, backslash \ is used as line continuation character
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())
        return loss

def gelu(x):
    """
    Implements Gaussian Error Linear Unit (GELU) activation function
    Can directly call nn.GELU()
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    """
    Implements Feed-Forward Network (FFN) in Transformer model, also known as Positionwise Feed-Forward Network
    Residual connections and layer normalization help avoid gradient vanishing during training
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        :param d_model: Model feature dimension
        :param d_ff: Intermediate layer dimension in feed-forward network
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        # Create linear layer to transform input from d_model dimension to d_ff dimension
        self.w_1 = nn.Linear(d_model, d_ff)
        # Create another linear layer to transform intermediate output from d_ff dimension back to d_model dimension
        self.w_2 = nn.Linear(d_ff, d_model)
        # Create layer normalization layer for normalizing input features
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # Assign GELU activation function to self.actv member variable
        self.actv = gelu
        # Create two Dropout layers to prevent overfitting during training
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        # First apply layer normalization to input x, then pass through first linear layer w_1, then apply GELU activation, finally through Dropout layer
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        # Pass intermediate result inter through second linear layer w_2, then apply second Dropout layer
        output = self.dropout_2(self.w_2(inter))
        # Add Dropout output with original input x for residual connection, then return result
        return output + x


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention mechanism, a key component in Transformer model
    """
    def __init__(self, head_count, model_dim, dropout=0.1):
        """
        :param head_count: Number of attention heads
        :param model_dim: Model feature dimension
        """
        # Ensure model dimension is divisible by number of heads, requirement for multi-head attention
        assert model_dim % head_count == 0
        # Calculate dimension per head
        self.dim_per_head = model_dim // head_count
        # Save number of heads
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        # Create three linear layers for transforming key, value, and query
        # model_dim = head_count * self.dim_per_head
        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        # Softmax layer for calculating attention weights
        self.softmax = nn.Softmax(dim=-1)
        # Dropout layer for preventing overfitting
        self.dropout = nn.Dropout(dropout)
        # For merging multi-head output back to model dimension
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        # Get batch size
        batch_size = key.size(0)
        # Get dimension per head and number of heads
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        # Two helper functions for adjusting tensor shape for multi-head attention calculation
        def shape(x):
            """ projection """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ compute context """
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)

        # Apply linear transformation to key, value, and query, adjust shape to separate different heads
        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        # Scale query, common operation in multi-head attention for gradient stability
        query = query / math.sqrt(dim_per_head)
        # Calculate dot product of query and key to get attention scores
        scores = torch.matmul(query, key.transpose(2, 3))

        # If mask provided, apply to attention scores to mask irrelevant information
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e10)

        # Calculate attention weights through Softmax layer
        attn = self.softmax(scores)
        # Apply Dropout to attention weights
        drop_attn = self.dropout(attn)
        # Calculate weighted value to get context vector
        context = torch.matmul(drop_attn, value).transpose(1, 2).\
                    contiguous().view(batch_size, -1, head_count * dim_per_head)
        # Pass context vector through linear layer, output final multi-head attention result
        output = self.linear(context)
        # Return multi-head attention output
        return output


class PositionalEncoding(nn.Module):
    """Positional Encoding, a component of Transformer that provides position information to model by adding sine and cosine function values to input sequence"""
    def __init__(self, dim, max_len=512):
        """
        :param dim: Positional encoding dimension
        :param max_len: Maximum length of positional encoding
        """
        super(PositionalEncoding, self).__init__()
        # For storing positional encoding
        pe = torch.zeros(max_len, dim)
        # Create sequence from 0 to max_len and convert to column vector
        position = torch.arange(0, max_len).unsqueeze(1)
        # Calculate divisor term for subsequent sine and cosine functions
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        # Calculate sine and cosine values for odd and even dimensions of positional encoding
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        # Add dimension before positional encoding tensor, shape becomes (1, max_len, dim) for broadcasting with batch sequences
        pe = pe.unsqueeze(0)
        # Register positional encoding as module buffer so it's not treated as model parameter, won't update during training
        self.register_buffer('pe', pe)

    def forward(self, x, speaker_emb):
        # Get sequence length of input x
        L = x.size(1)
        # Take part of positional encoding matching input sequence length
        pos_emb = self.pe[:, :L]
        # Add positional encoding and speaker embedding (if any) to input x to provide position information to model
        x = x + pos_emb + speaker_emb
        # Return output with added positional encoding
        return x


class TransformerEncoderLayer(nn.Module):
    """Implements one encoder layer in Transformer model, consisting of self-attention mechanism and feed-forward network, including layer normalization and residual connections"""
    def __init__(self, d_model, heads, d_ff, dropout):
        """
        :param d_model: Model feature dimension
        :param heads: Number of heads in multi-head attention mechanism
        :param d_ff: Intermediate layer dimension in feed-forward network
        :param dropout:
        """
        super(TransformerEncoderLayer, self).__init__()
        # For implementing self-attention mechanism
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        # For implementing feed-forward network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        # For normalizing input features
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask):
        # Check if inputs_a and inputs_b are equal
        if inputs_a.equal(inputs_b):
            # If not first iteration (usually for incremental learning during training), apply layer normalization to inputs_b
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            # Adjust mask shape to match self-attention mechanism input requirements
            mask = mask.unsqueeze(1)
            # Apply self-attention mechanism to calculate context information
            context = self.self_attn(inputs_b, inputs_b, inputs_b, mask=mask)
        else:
            # If inputs_a and inputs_b are not equal, meaning they are different inputs, usually for encoder part of Transformer
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_a, inputs_a, inputs_b, mask=mask)

        # Apply Dropout and add residual connection
        out = self.dropout(context) + inputs_b
        # Pass residual connection output to feed-forward network, return final encoder layer output
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    """Implements encoder part of Transformer model, consisting of multiple encoder layers, including positional encoding and Dropout layers"""
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1):
        """
        :param d_model: Model feature dimension
        :param d_ff: Intermediate layer dimension in feed-forward network
        :param heads: Number of heads in multi-head attention mechanism
        :param layers: Number of layers in encoder
        :param dropout:
        """
        super(TransformerEncoder, self).__init__()
        # Save model feature dimension, number of layers in encoder
        self.d_model = d_model
        self.layers = layers
        # For adding positional encoding to input sequence
        self.pos_emb = PositionalEncoding(d_model)
        # Create module list containing layers instances of TransformerEncoderLayer
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a, x_b, mask, speaker_emb):
        # Check if x_a and x_b are equal
        if x_a.equal(x_b):
            # Add positional encoding and speaker embedding (if any) to x_b
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            # Iterate through all encoder layers
            for i in range(self.layers):
                # For each encoder layer, pass x_b as input and use mask (mask.eq(0)) to mask irrelevant information
                x_b = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
        # If x_a and x_b are not equal, meaning they are different inputs, usually for encoder part of Transformer
        else:
            # Add positional encoding and speaker embedding to x_a and x_b respectively
            x_a = self.pos_emb(x_a, speaker_emb)
            x_a = self.dropout(x_a)
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                # For each encoder layer, pass x_a and x_b as input and use mask (mask.eq(0)) to mask irrelevant information
                x_b = self.transformer_inter[i](i, x_a, x_b, mask.eq(0))
        return x_b


class Unimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size, dataset):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        if dataset == 'MELD':
            # Initialize weights as identity matrix
            self.fc.weight.data.copy_(torch.eye(hidden_size, hidden_size))
            # Freeze weights so they don't update during training
            self.fc.weight.requires_grad = False

    def forward(self, x):
        gate = self.gate(x)
        out = gate * x
        return out

class Multimodal_GatedFusion(nn.Module):
    """Implements gated fusion for multimodal data, allowing model to combine features from different modalities and automatically adjust contribution based on importance of each modality"""
    def __init__(self, hidden_size):
        """
        :param hidden_size: Feature dimension
        """
        super(Multimodal_GatedFusion, self).__init__()
        # 1. For transforming features
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        # 2. Gating mechanism: Create Softmax layer for normalizing gating coefficients in second-to-last dimension
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b, c):
        # 3. Feature processing and concatenation
        # Receive input features from three different modalities a, b, and c
        a_new = a.unsqueeze(-2)
        b_new = b.unsqueeze(-2)
        c_new = c.unsqueeze(-2)
        # 4. Calculate gating coefficients
        # Concatenate features from three modalities to form new tensor
        utters = torch.cat([a_new, b_new, c_new], dim=-2)
        # Process each modality's features through linear layer and concatenate them
        utters_fc = torch.cat([self.fc(a).unsqueeze(-2), self.fc(b).unsqueeze(-2), self.fc(c).unsqueeze(-2)], dim=-2)
        # Calculate Softmax of gating coefficients, assigning weight to each modality's features indicating importance in fusion process
        utters_softmax = self.softmax(utters_fc)

        # 5. Weighted fusion
        # Multiply gating coefficients with concatenated features to get weighted feature representation
        utters_three_model = utters_softmax * utters
        # Sum weighted features in specified dimension to get final fused representation
        final_rep = torch.sum(utters_three_model, dim=-2, keepdim=False)
        # Return fused feature representation
        return final_rep


def simple_batch_graphify(features, lengths, no_cuda):
    node_features = []
    batch_size = features.size(1)
    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])

    node_features = torch.cat(node_features, dim=0)

    if not no_cuda:
        node_features = node_features.to("cuda:0")
    return node_features


class Transformer_Based_Model(nn.Module):
    """Implements Transformer-based multimodal emotion classification model that processes text, visual, and audio data and predicts emotion classes"""
    def __init__(self, dataset, temp, D_text, D_visual, D_audio, n_head,
                 n_classes, hidden_dim, n_speakers, dropout, D_g=1024, graph_hidden_size=1024, num_L=3, num_K=4, modals='avl'):
        """
        :param dataset: Name or type of dataset used
        :param temp: Temperature parameter for adjusting softmax smoothness
        :param D_text: Dimension of input text features
        :param D_visual: Dimension of input visual features
        :param D_audio: Dimension of input audio features
        :param n_classes: Number of output emotion classes
        :param hidden_dim: Model hidden layer dimension
        :param n_speakers: Number of speakers in conversation
        """
        super(Transformer_Based_Model, self).__init__()
        # Save temperature parameter for adjusting softmax smoothness
        self.temp = temp
        self.n_classes = n_classes
        self.n_speakers = n_speakers
        self.dropout = dropout
        
        if self.n_speakers == 2:
            padding_idx = 2
        if self.n_speakers == 9:
            padding_idx = 9
        # Create embedding layer for converting speaker indices to embedding vectors
        self.speaker_embeddings = nn.Embedding(n_speakers+1, hidden_dim, padding_idx)
        
        # Temporal convolutional layers
        # Create three 1D convolutional layers for converting input features from different modalities to required hidden dimension
        self.textf_input = nn.Conv1d(D_text, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.acouf_input = nn.Conv1d(D_audio, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.visuf_input = nn.Conv1d(D_visual, hidden_dim, kernel_size=1, padding=0, bias=False)
        
        # Intra- and Inter-modal Transformers
        # Create multiple TransformerEncoder instances for processing intra- and inter-modal features
        self.a_t = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.v_t = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t_t = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        self.a_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.v_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        self.v_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.a_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        
        # Unimodal-level Gated Fusion
        # Create multiple Unimodal_GatedFusion instances for unimodal-level gated fusion
        self.t_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.a_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.v_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.a_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.v_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.v_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.a_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        # Create three linear layers for dimensionality reduction after unimodal fusion/concatenation
        self.features_reduce_t = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_a = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_v = nn.Linear(3 * hidden_dim, hidden_dim)

        # Multimodal-level Gated Fusion
        self.last_gate = Multimodal_GatedFusion(hidden_dim)

        # Emotion Classifier
        # Create three sequential models, one for each modality, for emotion classification
        self.t_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.a_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.v_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )

        self.graph_model = GCN(n_dim=D_g, nhidden=graph_hidden_size,
                               dropout=self.dropout, lamda=0.5, alpha=0.1, variant=True,
                               return_feature=True, use_residue=False, n_speakers=n_speakers,
                               modals='avl', use_speaker=True, use_modal=False, num_L=num_L,
                               num_K=num_K)
        self.multi_modal = True
        self.att_type = 'concat_DHT'
        self.modals = self.modals = [x for x in modals]  # a, v, l
        self.use_residue = False

        if self.multi_modal:
            self.dropout_ = nn.Dropout(self.dropout)
            self.hidfc = nn.Linear(graph_hidden_size, n_classes)
            if self.att_type == 'concat_subsequently':
                if self.use_residue:
                    self.smax_fc = nn.Linear((D_g+graph_hidden_size)*len(self.modals), n_classes)
                else:
                    self.smax_fc = nn.Linear((graph_hidden_size)*len(self.modals), n_classes)
            elif self.att_type == 'concat_DHT':
                if self.use_residue:
                    self.smax_fc = nn.Linear((D_g+graph_hidden_size*2)*len(self.modals), n_classes)
                else:
                    self.smax_fc = nn.Linear((graph_hidden_size*2)*len(self.modals), n_classes)

            elif self.att_type == 'gated':
                if len(self.modals) == 3:
                    print("len(self.modals)", len(self.modals))
                    print("graph_hidden_size", len(graph_hidden_size))
                    self.smax_fc = nn.Linear(100*len(self.modals), graph_hidden_size)
                else:
                    self.smax_fc = nn.Linear(100, graph_hidden_size)
            else:
                self.smax_fc = nn.Linear(D_g+graph_hidden_size*len(self.modals), graph_hidden_size)

        self.all_output_layer = nn.Linear(hidden_dim, n_classes)

    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len, epoch=None):
        """
        :param textf: Input features for text modality
        :param visuf: Input features for visual modality
        :param acouf: Input features for audio modality
        :param u_mask: Usually for identifying valid or invalid modal data regions
        :param qmask: For indicating speaker changes or speaker indices at each time step
        :param dia_len: Dialogue length
        :return:
        """
        # Find speaker index for each time step through qmask
        spk_idx = torch.argmax(qmask, -1)
        spk_idx = torch.argmax(qmask.permute(1, 0, 2), -1)
        origin_spk_idx = spk_idx
        
        # Use n_speakers for adjusting speaker indices after dialogue length
        if self.n_speakers == 2:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (2*torch.ones(origin_spk_idx[i].size(0)-x)).int().cuda()
        if self.n_speakers == 9:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (9*torch.ones(origin_spk_idx[i].size(0)-x)).int().cuda()
        # Convert speaker indices to embedding vectors
        spk_embeddings = self.speaker_embeddings(spk_idx)

        # Temporal convolutional layers
        # Process input features through 1D convolutional layers and convert feature dimensions to required model dimensions
        textf = self.textf_input(textf.permute(1, 2, 0)).transpose(1, 2)
        acouf = self.acouf_input(acouf.permute(1, 2, 0)).transpose(1, 2)
        visuf = self.visuf_input(visuf.permute(1, 2, 0)).transpose(1, 2)

        # Intra- and Inter-modal Transformers
        t_t_transformer_out = self.t_t(textf, textf, u_mask, spk_embeddings)
        a_t_transformer_out = self.a_t(acouf, textf, u_mask, spk_embeddings)
        v_t_transformer_out = self.v_t(visuf, textf, u_mask, spk_embeddings)

        a_a_transformer_out = self.a_a(acouf, acouf, u_mask, spk_embeddings)
        t_a_transformer_out = self.t_a(textf, acouf, u_mask, spk_embeddings)
        v_a_transformer_out = self.v_a(visuf, acouf, u_mask, spk_embeddings)

        v_v_transformer_out = self.v_v(visuf, visuf, u_mask, spk_embeddings)
        t_v_transformer_out = self.t_v(textf, visuf, u_mask, spk_embeddings)
        a_v_transformer_out = self.a_v(acouf, visuf, u_mask, spk_embeddings)

        # Unimodal-level Gated Fusion
        t_t_transformer_out = self.t_t_gate(t_t_transformer_out)
        a_t_transformer_out = self.a_t_gate(a_t_transformer_out)
        v_t_transformer_out = self.v_t_gate(v_t_transformer_out)
        features_l = torch.cat([t_t_transformer_out, a_t_transformer_out, v_t_transformer_out])

        a_a_transformer_out = self.a_a_gate(a_a_transformer_out)
        t_a_transformer_out = self.t_a_gate(t_a_transformer_out)
        v_a_transformer_out = self.v_a_gate(v_a_transformer_out)
        features_a = torch.cat([a_a_transformer_out, t_a_transformer_out, v_a_transformer_out])

        v_v_transformer_out = self.v_v_gate(v_v_transformer_out)
        t_v_transformer_out = self.t_v_gate(t_v_transformer_out)
        a_v_transformer_out = self.a_v_gate(a_v_transformer_out)
        features_v = torch.cat([v_v_transformer_out, t_v_transformer_out, a_v_transformer_out])

        # Gated fusion for different modality Transformer encoder outputs
        t_transformer_out = self.features_reduce_t(torch.cat([t_t_transformer_out, a_t_transformer_out, v_t_transformer_out], dim=-1))
        a_transformer_out = self.features_reduce_a(torch.cat([a_a_transformer_out, t_a_transformer_out, v_a_transformer_out], dim=-1))
        v_transformer_out = self.features_reduce_v(torch.cat([v_v_transformer_out, t_v_transformer_out, a_v_transformer_out], dim=-1))

        # GCN
        features_a = simple_batch_graphify(a_transformer_out.permute(1, 0, 2), dia_len, False)
        features_v = simple_batch_graphify(v_transformer_out.permute(1, 0, 2), dia_len, False)
        features_l = simple_batch_graphify(t_transformer_out.permute(1, 0, 2), dia_len, False)

        emotions_feat = self.graph_model(features_a, features_v, features_l, dia_len, qmask, epoch)
        emotions_feat = self.dropout_(emotions_feat)
        emotions_feat = nn.ReLU()(emotions_feat)
        log_prob = F.log_softmax(self.smax_fc(emotions_feat), 1)
        prob = F.softmax(self.smax_fc(emotions_feat), 1)

        # Multimodal-level Gated Fusion
        all_transformer_out = self.last_gate(t_transformer_out, a_transformer_out, v_transformer_out)

        # Emotion Classifier
        # Process features through emotion classifier to get emotion predictions for each modality and multimodal fusion
        t_final_out = self.t_output_layer(t_transformer_out)
        a_final_out = self.a_output_layer(a_transformer_out)
        v_final_out = self.v_output_layer(v_transformer_out)
        all_final_out = self.all_output_layer(all_transformer_out)
        t_final_out_1 = self.t_output_layer(features_l)
        a_final_out_1 = self.a_output_layer(features_a)
        v_final_out_1 = self.v_output_layer(features_v)

        # Calculate softmax and log_softmax for subsequent loss calculation and probability interpretation
        t_log_prob = F.log_softmax(t_final_out, 2)
        a_log_prob = F.log_softmax(a_final_out, 2)
        v_log_prob = F.log_softmax(v_final_out, 2)
        t_log_prob_1 = F.log_softmax(t_final_out_1, 1)
        a_log_prob_1 = F.log_softmax(a_final_out_1, 1)
        v_log_prob_1 = F.log_softmax(v_final_out_1, 1)

        all_log_prob = F.log_softmax(all_final_out, 2)
        all_prob = F.softmax(all_final_out, 2)

        # Log probabilities and probabilities adjusted using temperature parameter self.temp
        kl_t_log_prob = F.log_softmax(t_final_out /self.temp, 2)
        kl_a_log_prob = F.log_softmax(a_final_out /self.temp, 2)
        kl_v_log_prob = F.log_softmax(v_final_out /self.temp, 2)
        kl_t_log_prob_1 = F.log_softmax(t_final_out_1 / self.temp, 1)
        kl_a_log_prob_1 = F.log_softmax(a_final_out_1 / self.temp, 1)
        kl_v_log_prob_1 = F.log_softmax(v_final_out_1 / self.temp, 1)

        kl_all_prob = F.softmax(all_final_out /self.temp, 2)
        kl_all_prob_1 = F.softmax(self.smax_fc(emotions_feat) /self.temp, 1)

        return t_log_prob, a_log_prob, v_log_prob, log_prob, prob, \
               kl_t_log_prob_1, kl_a_log_prob_1, kl_v_log_prob_1, kl_all_prob_1
