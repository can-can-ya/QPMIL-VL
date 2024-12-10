import torch.nn as nn
import torch
import torch.nn.functional as F

from .conch import tokenize, get_tokenizer


def merge_parameter(cfg, param):
    if cfg['opt_name'] == 'adam':
        merged_param = torch.cat([param[i] for i in range(cfg['pool_size'])], dim=0)
    else:
        raise NotImplementedError('Invalid optimizer')

    return merged_param


class TunableVLearner(nn.Module):
    """
    Tunable Vector
    Adapted from TaskRes: https://github.com/geekyutao/TaskRes
    """
    def __init__(self, cfg, tunable_v):
        """
        :param cfg: the dict of configuration parameters.
        :param tunable_v: the trainable parameter, which is predefined at QPMIL-VL/manager/manager.py.
        """
        super().__init__()

        self.cfg = cfg
        self.alpha = cfg['alpha']
        self.tunable_v = tunable_v # Trainable parameters

    def forward(self, class_ensemble_feature):
        """
        :param class_ensemble_feature: Tensor(num_cls, C).
        :return: enhanced class feature: Tensor(num_cls, C).
        """
        return class_ensemble_feature + self.alpha * torch.cat([self.tunable_v[i] for i in range(self.cfg['task_num'])], dim=0) # [num_cls, C]


class CONCHPromptEncoder(nn.Module):
    """
    CONCH (CONCH custom API at github.com/mahmoodlab/CONCH)
    """
    def __init__(self, coca_model):
        super().__init__()
        # load text model and text projection
        coca_text_model = coca_model.text

        # Config of clip text model
        self.pad_id = coca_text_model.pad_id
        assert self.pad_id == 0, "Assume pad_id = 0 in CONCH to encode prompts as expected."
        self.heads = coca_text_model.heads

        # Embedding
        self.positional_embedding = coca_text_model.positional_embedding

        self.attn_mask = coca_text_model.attn_mask
        # Transformer
        self.transformer = coca_text_model.transformer
        self.ln_final = coca_text_model.ln_final

        self.cls_emb = coca_text_model.cls_emb

        # Text projection
        self.text_projection = coca_text_model.text_projection

        # Additional attributes used for downstream calls
        self.token_embedding = coca_text_model.token_embedding
        self.text_config = {
            'max_num_tokens': 127,
            'embedding_dim': self.token_embedding.embedding_dim,
            'embedding_dtype': self.token_embedding.weight.dtype
        }

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def _repeat(self, t, N: int):
        return t.reshape(1, 1, -1).repeat(N, 1, 1)

    def forward(self, prompts_embedding, prompts_pseudo_tokens):
        """
        Adapted from CONCH API:
            https://github.com/mahmoodlab/CONCH/blob/main/conch/open_clip_custom/transformer.py#L418

        input:
            prompts_embedding: the embedding of ranking prompts, with shape [n_batch, length_ctx, dim_embedding].
                Note that this input differs from the text_encoder input of HuggingFace CLIP.
            prompts_pseudo_tokens: the pseudo-tokens of ranking prompts, with shape [n_batch, length_ctx].
        """
        cast_dtype = self.transformer.get_cast_dtype()
        device = prompts_embedding.device
        seq_len = prompts_embedding.shape[1] # max_length - 1 = 128 - 1
        x = prompts_embedding.to(cast_dtype)  # [batch_size, n_ctx, d_model]

        prompts_pseudo_tokens = prompts_pseudo_tokens.to(device)
        attn_mask = self.attn_mask.to(device)
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, self._repeat(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(prompts_pseudo_tokens, cast_dtype)
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.cls_emb is not None:
            pooled, tokens = x[:, -1], x[:, :-1]
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled, tokens = x[torch.arange(x.shape[0]), prompts_pseudo_tokens.argmax(dim=-1)], x

        if self.text_projection is not None:
            pooled = pooled @ self.text_projection

        return pooled


class PromptLearner(nn.Module):
    """
    Learn the prompt by concatenating the prompt to the embedding and generate class ensemble feature.
    """
    def __init__(self, cfg, base_model, text_encoder, device, prompt, current_ensemble_classes):
        """
        :param cfg: the dict of configuration parameters.
        :param base_model: the basic pre-trained VL model, e.g., CONCH.
        :param text_encoder: the text encoder of basic pre-trained VL model.
        :param device: the used device (CPU or GPU).
        :param prompt: the trainable parameter, which is predefined at QPMIL-VL/manager/manager.py.
        :param current_ensemble_classes: {'count': list(), 'ensemble_classes': list()}, 'count' specifies the number of text descriptions for each subtype and 'ensemble_classes' contains all class text descriptions.
        """
        super().__init__()

        self.cfg = cfg
        if cfg['base_model_arch'] == 'CONCH':
            self.embedding_dim = base_model.text.ln_final.weight.shape[0] # D_e: embedding dimension of E_txt
        else:
            raise NotImplementedError("Please specify a valid architecture.")
        text_encoder.eval()

        # Trainable parameters
        self.prompt = prompt

        # prompt
        self.tokenized_prompts, embedding = self._get_embedding(base_model, device, cfg['prompt_length'])
        self.embedding_prefix = embedding[:, :1, :]
        self.embedding_suffix = embedding[:, 1 + cfg['prompt_length']:, :]

        # Class Ensemble: current_ensemble_classes
        count = current_ensemble_classes['count']
        tokenized_prompts, embedding = self._get_embedding(base_model, device, classes=current_ensemble_classes['ensemble_classes'])
        with torch.no_grad():
            class_feature_matrix = text_encoder(embedding, tokenized_prompts)
        self.class_ensemble_feature = self._get_ensemble_feature(class_feature_matrix, count) # [num_cls, C]

    def _get_ensemble_feature(self, class_feature_matrix, count):
        count_t = []
        t = 0
        for i in range(len(count)):
            t += count[i]
            count_t.append(t)

        index = 0
        feature_list = []
        for i in range(len(count)):
            feature = torch.mean(class_feature_matrix[index:count_t[i]], dim=0, keepdim=True)
            feature_list.append(feature)
            index = count_t[i]

        return torch.cat(feature_list, dim=0)

    def _get_embedding(self, base_model, device, num_placeholder=0, classes=None):
        prompt_prefix = ' '.join(['x'] * num_placeholder)
        if classes is None:
            prompts = [prompt_prefix + '.']
        elif prompt_prefix != '':
            prompts = [prompt_prefix + ' ' + name + '.' for name in classes]
        else:
            prompts = [name + '.' for name in classes]
        if self.cfg['base_model_arch'] == 'CONCH':
            tokenized_prompts = tokenize(get_tokenizer(), prompts)[:, :-1]
            with torch.no_grad():
                embedding = base_model.text.token_embedding(tokenized_prompts.to(device)).type(base_model.dtype)
        else:
            raise NotImplementedError("Please specify a valid architecture.")

        return tokenized_prompts, embedding

    def forward(self, indices, mini_batch):
        """
        :param indices: Tensor(MB, match_size), the matching indices.
        :param mini_batch: the mini-batch size.
        :return:
            embedding: Tensor(MB * match_size, X3, embedding_dim).
            tokenized prompts: Tensor(MB * match_size, X3).
        """
        merged_prompt = merge_parameter(self.cfg, self.prompt) # [pool_size, prompt_length, embedding_dim]
        embedding_core = merged_prompt[indices] # [MB, match_size, prompt_length, embedding_dim]

        embedding_prefix = self.embedding_prefix.unsqueeze(0).repeat(mini_batch, self.cfg['match_size'], 1, 1) # [MB, match_size, X1, embedding_dim]
        embedding_suffix = self.embedding_suffix.unsqueeze(0).repeat(mini_batch, self.cfg['match_size'], 1, 1) # [MB, match_size, X2, embedding_dim]
        embedding = torch.cat([embedding_prefix, embedding_core, embedding_suffix], dim=2).view(mini_batch * self.cfg['match_size'], -1, self.embedding_dim) # [MB * match_size, X3, embedding_dim]
        tokenized_prompts = self.tokenized_prompts.unsqueeze(0).repeat(mini_batch, self.cfg['match_size'], 1).view(mini_batch * self.cfg['match_size'], -1) # [MB * match_size, X3]

        return embedding, tokenized_prompts


class QPMIL_VL(nn.Module):
    """
    The first VL-based framework with Queryable Prototype MIL, QPMIL-VL, specially for incremental WSI classification.
    Paper: https://arxiv.org/abs/2410.10573
    """
    def __init__(self, cfg, base_model, device, key, prompt, tunable_v, current_ensemble_classes, train_key_frequency):
        """
        :param cfg: the dict of configuration parameters.
        :param base_model: the basic pre-trained VL model, e.g., CONCH.
        :param device: the used device (CPU or GPU).
        :param key, prompt, tunable_v: the trainable parameters, which are predefined at QPMIL-VL/manager/manager.py.
        :param current_ensemble_classes: {'count': list(), 'ensemble_classes': list()}, 'count' specifies the number of text descriptions for each subtype and 'ensemble_classes' contains all class text descriptions.
        :param train_key_frequency: the matching frequency of prototype key on training sets of different data sets.
        """
        super().__init__()

        self.cfg = cfg
        self.dtype = base_model.dtype
        self.logit_scale = base_model.logit_scale
        self.device = device

        """
        Trainable parameters:
            - Vision branch: key-prompt pairs in Prototype Pool (key, prompt)
            - Language branch: Tunable Vector (tunable_v)
        """
        self.key = key
        self.tunable_v_learner = TunableVLearner(cfg, tunable_v) # Tunable Vector
        if cfg['base_model_arch'] == 'CONCH':
            self.text_encoder = CONCHPromptEncoder(base_model)
        else:
            raise NotImplementedError("Please specify a valid architecture.")
        self.prompt_learner = PromptLearner(cfg, base_model, self.text_encoder, device, prompt, current_ensemble_classes)

        if cfg['task_num'] > 1:
            self.penalty_table = self._get_penalty_table(train_key_frequency)
        else:
            self.penalty_table = None

    def _get_penalty_table(self, train_key_frequency):
        penalty_table = None
        train_key_frequency_v_list = list(train_key_frequency.values())
        for i in range(self.cfg['task_num'] - 1):
            one_task_frequency = train_key_frequency_v_list[i][-1]
            total_count = torch.sum(one_task_frequency)
            frequency_table = one_task_frequency / total_count
            if penalty_table is None:
                penalty_table = frequency_table
            else:
                penalty_table += frequency_table
        penalty_table = penalty_table / (self.cfg['task_num'] - 1)
        penalty_table = penalty_table.type(self.dtype).to(self.device)

        return penalty_table

    def _query_prototype_pool(self, x_list, mini_batch, eval):
        """
        Prototype Pool
        Adapted from L2P: https://github.com/JH-LEE-KR/l2p-pytorch

        :param x_list: [Tensor(B=1, N, C), ], list of multiple WSI samples.
        :param mini_batch: the mini-batch size.
        :param eval:
            True: evaluation mode.
            False: training mode.
        :return:
            matching indices: Tensor(MB, match_size).
            matching loss.
        """
        q_vec_list = []
        for i in range(mini_batch):
            x_list[i] = x_list[i].type(self.dtype)
            if self.cfg['pooling'] == 'max':
                q_vec, _ = torch.max(x_list[i], dim=1)  # [B, N, C] -> [B, C]: B = 1
            elif self.cfg['pooling'] == 'mean':
                q_vec = torch.mean(x_list[i], dim=1)  # [B, N, C] -> [B, C]: B = 1
            else:
                raise NotImplementedError('invalid pooling method')
            q_vec_list.append(q_vec)
        q_vecs = torch.cat(q_vec_list, dim=0) # [MB, C]
        q_vecs = q_vecs / q_vecs.norm(dim=-1, keepdim=True)

        merged_key = merge_parameter(self.cfg, self.key) # [pool_size, C]
        merged_key = merged_key / merged_key.norm(dim=-1, keepdim=True)
        cos_sim = q_vecs @ merged_key.t()  # Cosine similarity [MB, pool_size]
        if not eval and self.cfg['task_num'] > 1:
            cos_dist = 1 - cos_sim
            punished_cos_dist = cos_dist * self.penalty_table
            _, indices = punished_cos_dist.topk(k=self.cfg['match_size'], dim=1, largest=False) # [MB, match_size]
        else:
            _, indices = cos_sim.topk(k=self.cfg['match_size'], dim=1, largest=True) # [MB, match_size]
        key_id, id_counts = torch.unique(indices, return_counts=True, sorted=True)
        _, major_idx = torch.topk(id_counts, k=self.cfg['match_size'])
        indices = key_id[major_idx]  # minibatch-wise indices [match_size, ]
        indices = indices.expand(mini_batch, -1) # [MB, match_size]

        # matching loss
        if eval:
            matching_loss = None
        else:
            matched_key = merged_key[indices]
            matched_key = matched_key / matched_key.norm(dim=-1, keepdim=True)
            q_vecs = q_vecs.unsqueeze(1).repeat(1, self.cfg['match_size'], 1)
            matching_loss = 1 - ((q_vecs * matched_key) / (mini_batch * self.cfg['match_size'])).sum()

        return indices, matching_loss

    def _get_bag_feature(self, x_list, indices, mini_batch):
        """
        Prototype-guided Aggregation
        Adapted from TOP: https://github.com/miccaiif/TOP

        :param x_list: [Tensor(B=1, N, C), ], list of multiple WSI samples.
        :param indices: Tensor(MB, match_size), the matching indices.
        :param mini_batch: the mini-batch size.
        :return: bag-level feature: Tensor(MB, 1, C).
        """
        embedding, tokenized_prompts = self.prompt_learner(indices, mini_batch)
        prototype_features = self.text_encoder(embedding, tokenized_prompts)
        prototype_features = prototype_features / prototype_features.norm(dim=-1, keepdim=True)
        prototype_features = prototype_features.view(mini_batch, self.cfg['match_size'], -1) # [MB, match_size, C]

        bag_feature_list = []
        for i, x in enumerate(x_list):
            x = x.squeeze()
            x_norm = x / x.norm(dim=-1, keepdim=True)
            scaled_cos_sim_matrix = self.cfg['csm_logit_scale'] * x_norm @ prototype_features[i].t() # [N, match_size]
            weighted_aggregation_matrix = torch.softmax(scaled_cos_sim_matrix, dim=0)
            bag_feature = torch.mean(weighted_aggregation_matrix.t() @ x, dim=0, keepdim=True) # [1, C]
            bag_feature_list.append(bag_feature)
        bag_feature = torch.cat(bag_feature_list, dim=0) # [MB, C]
        bag_feature = bag_feature / bag_feature.norm(dim=-1, keepdim=True)
        bag_feature = bag_feature.unsqueeze(1) # [MB, 1, C]

        return bag_feature

    def _compute_class_sim_loss(self, enhanced_class_feature): # class similarity loss
        features = enhanced_class_feature[0]
        n_cls = features.shape[0]
        cos_sim = features @ features.permute(1, 0)
        cos_sim += 1
        class_sim_loss = cos_sim[~torch.eye(n_cls, dtype=torch.bool, device=self.device)].mean()

        return class_sim_loss

    def forward(self, x_list, eval=False):
        """
        :param x_list: [Tensor(B=1, N, C), ], list of multiple WSI samples.
        :param eval:
            True: evaluation mode.
            False: training mode.
        :return:
            evaluate:
                classification logits: Tensor(MB, num_cls).
                matching indices: Tensor(MB, match_size).
            train:
                classification logits: Tensor(MB, num_cls).
                loss dict: the matching loss and class similarity loss.
                matching indices: Tensor(MB, match_size).
                feature dict: the image (bag-level) feature and text (enhanced class) feature.
        """
        mini_batch = len(x_list)

        # Vision Branch (QPMIL)
        indices, matching_loss = self._query_prototype_pool(x_list, mini_batch, eval) # Prototype Pool [MB, match_size]
        bag_feature = self._get_bag_feature(x_list, indices, mini_batch) # Prototype-guided Aggregation  [MB, 1, C]

        # Language Branch (CFE)
        enhanced_class_feature = self.tunable_v_learner(self.prompt_learner.class_ensemble_feature) # Tunable Vector [num_cls, C]
        enhanced_class_feature = enhanced_class_feature.unsqueeze(0).repeat(mini_batch, 1, 1) # [MB, num_cls, C]
        enhanced_class_feature = enhanced_class_feature / enhanced_class_feature.norm(dim=-1, keepdim=True)

        # compute logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * (bag_feature * enhanced_class_feature).sum(-1) # [MB, num_cls]

        if eval: # eval
            return logits.float(), indices
        else: # train
            # class similarity loss
            class_sim_loss = self._compute_class_sim_loss(enhanced_class_feature)
            loss_dict = {
                'matching_loss': matching_loss.float(),
                'class_sim_loss': class_sim_loss.float()
            }

            feature_dict = {
                'image': bag_feature.float().squeeze(1),
                'text': enhanced_class_feature.float()
            }

            return logits.float(), loss_dict, indices, feature_dict