import math
import torch
import torchvision
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Image encoder.
    """

    def __init__(self, encoded_image_size=14, emb_dim=300, encoder_dim=2048):
        """
        Initialize.
        
        :param encoded_image_size: the size of the encoded image
        :param emb_dim: target embedding dimension
        :param encoder_dim: dimension size of the encoder
        """
        super(Encoder, self).__init__()
        self.emb_dim = emb_dim
        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101
        # Remove linear and pool layers (since we're not doing classification).
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # Resize image to the fixed size to allow input images of variable size.
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )
        self.conv1 = nn.Conv2d(encoder_dim, emb_dim, 1)
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images to encode
        :return: encoded images
        """
        out = self.resnet(images)
        out = self.adaptive_pool(out)
        batch_size, _, _, _ = out.shape
        out = self.conv1(out)
        out = out.view(batch_size, self.emb_dim, -1)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Enable/disable computation of gradients for the encoder.

        :param fine_tune: whether or not to enable fine-tuning
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4.
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class EntityEncoder(nn.Module):
    """
    Encodes entities in the entity context.

    The current implementation considers geographic entities and 
    encodes them based on their geographic features.
    """

    def __init__(self, emb_dim, type_embedding):
        """
        Initialize.
        
        :param emb_dim: the embedding dimension for a single entity
        :param type_embedding: the embedding layer for entity types
        """
        super(EntityEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.type_embedding = type_embedding
        
    def forward(self, entities):
        """
        Forward propagation.

        :param entities: entities to encode
        :return: encoded entities
        """
        entities_encoded = torch.zeros(entities.shape[0], entities.shape[1], self.emb_dim)
        # Encode based on the entity features.
        # Distance.
        entities_encoded[:, :, 0] = entities[:, :, 1]
        # Normalized azimuth.
        original_azimuth = entities[:, :, 2].clone()
        entities_encoded[:, :, 1] = original_azimuth.apply_(self.get_dist_to_north)
        original_azimuth = entities[:, :, 2].clone()
        entities_encoded[:, :, 2] = original_azimuth.apply_(self.get_dist_to_east)
        # Size.
        entities_encoded[:, :, 3] = entities[:, :, 3]
        # Store an embedding of the entity type.
        original_type = entities[:, :, 4].clone().long().to(device)
        entities_encoded[:, :, 4:] = self.type_embedding(original_type)
        entities_encoded = entities_encoded.to(device)
        return entities_encoded
        
    @staticmethod
    def get_dist_to_east(val):
        """
        Get the 'distance' between the passed value and the absolute east (90 degrees).
        """
        if val >= -90:
            res = abs(90 - val)
        else:
            res = 90 + abs(val + 180)
        return res / 180

    @staticmethod
    def get_dist_to_north(val):
        """
        Get the 'distance' between the passed value and the absolute north (0 degrees).
        """
        return abs(val) / 180
        
       
class CaptionEmbedder(nn.Module):
    """
    Transforms caption token indices into embeddings.
    
    CaptionEmbedder uses the precomputed mask that shows which of the tokens in the caption
    are regular vocabulary words and which are entities. These two types of tokens
    are embedded in different ways.
    """

    def __init__(self, vocab_size):
        """
        Initialize.
        
        :param vocab_size: the size of the wordmap
        """
        super(CaptionEmbedder, self).__init__()
        self.vocab_size = vocab_size

    def forward(self, caption_indices, entities_encoded, word_embedding, pad_token, caption_masks):
        """
        Forward propagation.

        :param caption_indices: caption token indices
        :param entities_encoded: encoded entities from the entity context
        :param word_embedding: an embedding layer for the regular vocabulary
        :param pad_token: the index of the <pad> token in the vocabulary
        :param caption_masks: the mask indicating the types of tokens in the captions
        :return: captions with every token embedded as a regular vocab word or an entity
        """
        caption_embeddings_lst = []
        for batch_num, batch in enumerate(caption_indices):
            #
            # Get indices that would belong to entities.
            batch_ents = batch - self.vocab_size
            # If an index does not belong to an entity, map it to <unk_ent> (always last in the entity context).
            batch_ents[(batch_ents < 0) | (batch_ents >= entities_encoded.shape[1])] = entities_encoded.shape[1] - 1
            #
            # Get regular vocab word indices.
            batch_words = batch.clone()
            # If an index does not belong to a regular vocab word, map it to the pad token.
            batch_words[batch_words >= self.vocab_size] = pad_token
            #
            # Embed the different token types.
            embeddings_words = word_embedding(batch_words)
            try:
                embedded_entities_batch = entities_encoded[batch_num]
            except:
                embedded_entities_batch = entities_encoded[0]
            embeddings_ents = torch.index_select(embedded_entities_batch, 0, batch_ents)
            #
            # Select the correct embeddings for the caption tokens according to the provided mask.
            embeddings = torch.where(
                caption_masks[batch_num] == 1, embeddings_ents, embeddings_words
            )
            caption_embeddings_lst.append(embeddings)
        caption_embeddings = torch.stack(caption_embeddings_lst)
        return caption_embeddings
        
        
class PositionEncoder(nn.Module):
    """
    Encodes the position of the tokens in the caption.
    """
    
    def __init__(self, emb_dim, dropout, max_len=5000):
        """
        Initialize.
        
        :param emb_dim: the dimensionality of the token embeddings
        :param dropout: the dropout value
        :param max_len: the maximum length of a caption
        """
        super(PositionEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
        
  
class DecoderTransformer(nn.Module):
    """
    Generates the caption.
    """
    
    def __init__(
        self, 
        word_map, emb_dim, decoder_dim, encoder_dim, num_heads, num_layers, dropout_dec=0.5, dropout_enc=0.5, dropout_pos=0.1
    ):
        """
        Initialize.
        
        :param word_map: wordmap (vocabulary)
        :param emb_dim: dimension of the token embeddings
        :param decoder_dim: dimension of the feedforward network in the Transformer decoder 
        :param encoder_dim: dimension of the feedforward network in the Transformer encoder
        :param num_heads: number of heads in a Transformer layer
        :param num_layers: number of Transformer layers
        """
        super(DecoderTransformer, self).__init__()
        #
        self.word_map = word_map
        vocab_size = len(word_map)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.softmax = nn.Softmax(dim=-1)
        #
        self.lookahead_mask = None
        self.pos_encoder = PositionEncoder(emb_dim, dropout_pos)
        decoder_layers = nn.TransformerDecoderLayer(emb_dim, num_heads, decoder_dim, dropout_dec)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        encoder_layers_entities = nn.TransformerEncoderLayer(emb_dim, num_heads, encoder_dim, dropout_enc)
        self.transformer_encoder_entities = nn.TransformerEncoder(encoder_layers_entities, num_layers)
        #
        self.word_embedding = nn.Embedding(vocab_size, emb_dim)
        type_embedding = nn.Embedding(1000, emb_dim - 4)
        self.entity_encoder = EntityEncoder(emb_dim, type_embedding)
        self.caption_embedder = CaptionEmbedder(vocab_size)
        #
        self.fc_vocab = nn.Linear(emb_dim, vocab_size)
        self.fc_entity = nn.Linear(emb_dim, 1)
        #
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        """
        Generate a mask for Transformer decoding.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask
    
    def init_weights(self):
        """
        Initialize the linear layer weights from a uniform distribution.
        """
        initrange = 0.1
        self.fc_vocab.bias.data.zero_()
        self.fc_vocab.weight.data.uniform_(-initrange, initrange)
        self.fc_entity.bias.data.zero_()
        self.fc_entity.weight.data.uniform_(-initrange, initrange)
        
    def load_pretrained_embeddings(self, embeddings):
        """
        Initialize the embedding layer with pre-trained word embeddings.

        :param embeddings: pre-trained word embeddings
        """
        self.word_embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Fine-tune the embedding layer.

        :param fine_tune: whether to fine-tune the vocabulary word embeddings
        """
        for p in self.word_embedding.parameters():
            p.requires_grad = fine_tune
                        
    def get_scores(self, h, entities_encoded):
        """
        Get generation probability scores over the vocabulary and entity context.
        
        :param h: the hidden state of the decoder
        :param entities_encoded: the encoded entity context
        :return: the scores over the vocabulary and entity context
        """
        caption_length = h.shape[0]
        batch_size = h.shape[1]
        entity_context_size  = entities_encoded.shape[1]
        # Compute scores over the vocabulary.
        preds_vocab = self.fc_vocab(h)
        # Compute scores over the entity context.
        h_expanded = h.unsqueeze(2).expand(caption_length, batch_size, entity_context_size, self.emb_dim)
        entities_encoded_expanded = (
            entities_encoded.unsqueeze(0).expand(caption_length, batch_size, entity_context_size, self.emb_dim)
        )
        entity_input = h_expanded * entities_encoded_expanded
        preds_entities = self.fc_entity(entity_input).squeeze(3)
        # Combine the computed scores.
        scores = torch.cat([preds_vocab, preds_entities], dim=2)
        return scores

    def forward(self, captions, encoder_out, caption_masks, caption_lengths, entities):
        """
        Generate a caption (at training).
        
        :param captions: captions as token indices
        :param encoder_out: encoded image
        :param caption_masks: the mask for the captions indicating the type of each token (vocab word, entity)
        :param caption_lengths: the lengths of each caption
        :param entities: the entity context
        :return: 
            - the scores over the vocabulary and entity context for each position in the caption
            - captions sorted by decreasing length
            - caption decoding lengths
        """
        # Sort input data by decreasing length.
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        captions = captions[sort_ind]
        caption_masks = caption_masks[sort_ind].unsqueeze(2)
        entities = entities[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()
        # Encode the entity context.
        entities_encoded = self.entity_encoder(entities) 
        # Get embeddings for the caption tokens.
        embeddings = self.caption_embedder(
            captions, 
            entities_encoded, 
            self.word_embedding,
            self.word_map["<pad>"],
            caption_masks
        )
        # Get the full encoding of the context, including the image.
        encoder_out_permuted = encoder_out.permute(2, 0, 1)
        entity_context_encoded = self.transformer_encoder_entities(entities_encoded.permute(1, 0, 2))
        context_encoded = torch.cat([encoder_out_permuted, entity_context_encoded]) 
        # Create the lookahead mask.
        captions_permuted = captions.permute(1, 0)
        if self.lookahead_mask is None or self.lookahead_mask.size(0) != len(captions_permuted):
            self.lookahead_mask = self._generate_square_subsequent_mask(len(captions_permuted)).to(device)
        # Decode based on the embeddings of the previous text and the combined context.
        embeddings = embeddings.permute(1, 0, 2)
        embeddings = embeddings*math.sqrt(self.emb_dim)
        embeddings = self.pos_encoder(embeddings)
        h = self.transformer_decoder(embeddings, context_encoded, self.lookahead_mask)
        # Produce probability scores for the vocabulary and entity context.
        scores = self.get_scores(h, entities_encoded).permute(1, 0, 2)
        return scores, captions, decode_lengths
        
    def predict(self, encoder_out, max_pred_len, entities):
        """
        Generate a caption (at inference).
        
        :param encoder_out: encoded image
        :param max_pred_len: the maximum length of a caption to generate
        :param entities: the entity context
        :return: the generated caption (as token indices)
        """
        # Encode the entity context.
        entities_encoded = self.entity_encoder(entities) 
        entity_context_encoded = self.transformer_encoder_entities(entities_encoded.permute(1, 0, 2))
        # Get the full encoding of the context, including the image.
        encoder_out_permuted = encoder_out.permute(2, 0, 1)
        context_encoded = torch.cat([encoder_out_permuted, entity_context_encoded])
        #
        batch_size = context_encoded.shape[1]
        captions = (torch.ones((max_pred_len, batch_size), dtype=int) * self.word_map["<start>"]).to(device)
        caption_masks = torch.zeros((max_pred_len, batch_size), dtype=int).to(device)
        # Create the lookahead mask.
        if self.lookahead_mask is None or self.lookahead_mask.size(0) != len(captions):
            self.lookahead_mask = self._generate_square_subsequent_mask(len(captions)).to(device)
        #
        output = (torch.ones((max_pred_len, batch_size), dtype=int) * self.word_map["<pad>"]).to(device)
        prev_top_two = []
        # Iterate until the maximum caption length is reached.
        for i in range(max_pred_len):
            captions = captions.permute(1, 0)
            caption_masks = caption_masks.permute(1, 0)
            # Embed the already generated caption tokens.
            embeddings = self.caption_embedder(
                captions, 
                entities_encoded, 
                self.word_embedding,
                self.word_map["<pad>"],
                caption_masks.unsqueeze(2)
            )
            embeddings = embeddings.permute(1, 0, 2)
            captions = captions.permute(1, 0)
            caption_masks = caption_masks.permute(1, 0)
            # Decode based on the embeddings of the previous text and the combined context.
            embeddings = embeddings*math.sqrt(self.emb_dim)
            embeddings = self.pos_encoder(embeddings)
            h = self.transformer_decoder(embeddings, context_encoded, self.lookahead_mask)
            h = h[i].unsqueeze(0)
            # Produce probability scores for the vocabulary and entity context.
            scores = self.get_scores(h, entities_encoded).squeeze(0)
            scores = self.softmax(scores)
            # Store the highest scoring token.
            out = scores.argmax(dim=1)           
            output[i] = out
            if out == self.word_map["<end>"]:
                # The end of generation token has been produced.
                break
            # Clean up generation loops.
            _, top_indices = scores.topk(k=2, dim=1)
            top_two = top_indices[0][1].unsqueeze(0)
            prev_top_two.append(top_two)
            for dupl_idx in {0, 2, 4}:
                if i > dupl_idx:
                    dupl_strings = []
                    for dupl_idx_prev in range(dupl_idx+2):
                        dupl_strings.append(output[i-dupl_idx_prev])
                    dupl_string_one = dupl_strings[:int(len(dupl_strings)/2)]
                    dupl_string_two = dupl_strings[int(len(dupl_strings)/2):]
                    if dupl_string_one == dupl_string_two:
                        if dupl_idx == 0:
                            top_dupl_idx = 1
                        else:
                            top_dupl_idx = dupl_idx
                        for dupl_idx_rewrite in range(top_dupl_idx):
                            output[i-dupl_idx_rewrite] = prev_top_two[-(dupl_idx_rewrite+1)]
                        break
            out = output[i]
            # Record the caption and mask tokens for the next iteration.
            if i < max_pred_len-1:
                captions[i+1] = out
                if out >= len(self.word_map):
                    # Generated an entity token.
                    caption_masks[i+1] = 1
        return output
