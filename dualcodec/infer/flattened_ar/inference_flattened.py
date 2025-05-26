from cv2 import repeat
import torch
from einops import rearrange
from .flatten_patterns import offset_codes, deoffset_codes


class Inference:
    def __init__(
        self, model, ckpt_path, cfg, device="cuda", normalize=False, half=False, split_paragraph=True, **kwargs
    ) -> None:
        self.model = model
        import safetensors.torch

        self.model.to(device)
        self.model.eval()
        safetensors.torch.load_model(self.model, ckpt_path, device=device)
        self.cfg = cfg
        self.tokenizer = self.cfg.get_tokenizer()

        for key in self.cfg.semantic_model:
            if isinstance(self.cfg.semantic_model[key], torch.nn.Module) or isinstance(
                self.cfg.semantic_model[key], torch.Tensor
            ):
                self.cfg.semantic_model[key] = self.cfg.semantic_model[key].to(device)
        self.device = device
        self.normalize = normalize

        if (
            hasattr(self.cfg, "skip_semantic_normalize")
            and self.cfg.skip_semantic_normalize
        ):
            print("skip semantic normalize")

        self.model = self.model.half()
        
        self.split_paragraph = split_paragraph
    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def _extract_codes_dac(self, input_features, attention_mask, audio):
        vq_emb = self.cfg.semantic_model["model"](
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[self.cfg.semantic_model["output_idx"]]  # (B, T, C)
        if (
            hasattr(self.cfg, "skip_semantic_normalize")
            and self.cfg.skip_semantic_normalize
        ):
            pass
        else:
            feat = (feat - self.cfg.semantic_model["mean"]) / self.cfg.semantic_model[
                "std"
            ]
        feat = feat.transpose(1,2)
        feat = torch.nn.functional.avg_pool1d(feat, self.cfg.semantic_model['repcodec_model'].semantic_downsample_factor, self.cfg.semantic_model['repcodec_model'].semantic_downsample_factor)
        
        audio = audio.reshape(1,1,-1)
        semantic_codes, acoustic_codes = self.cfg.semantic_model["repcodec_model"].encode(audio, semantic_repr=feat)
        semantic_codes = rearrange(semantic_codes, 'b 1 t -> b t')
        acoustic_codes = rearrange(acoustic_codes, 'b q t -> b t q')
        return semantic_codes, acoustic_codes

    @torch.no_grad()
    def inference(
        self,
        speech_24k,
        prompt_speech,
        prompt_text,
        prompt_language,
        target_text,
        target_language,
        use_prompt_text=True,
        temp=1.0,
        top_k=1000,
        top_p=0.85,
        repeat_penalty=1.1,
    ):
        """
            Generate text given speech and text prompts.

        Args:
            prompt_speech (str or Tensor): Speech file path or a tensor with shape (n_samples,).
            prompt_text (str): Text prompt.
            prompt_language (str): Language of the prompt.
            target_text (str): Target text to be completed.
            target_language (str): Language of the target text.
            use_prompt_text (bool, optional): Whether to use the prompt text as input. Defaults to True.
            temp (float, optional): Temperature parameter for the distribution. Defaults to 1.0.
            top_k (int, optional): Number of tokens to keep before applying `top_p`. Defaults to 1000.
            top_p (float, optional): Probability threshold to use for filtering tokens. Defaults to 0.85.

        Returns:
            str: Completed text.
        """
        self.model.eval()
        prompt_text = prompt_text.strip()
        # prompt_text = prompt_text.replace('.',',')
        # prompt_text = prompt_text.replace('。','，')
        target_text = target_text.replace("\n", "")
        target_text = target_text.replace("\t", "")
        return_values_0 = []
        return_values_1 = []

        prompt_len_tmp = len(self.tokenizer.encode(prompt_text)) // 2

        if self.split_paragraph:
            if prompt_language == 'zh':
                from dualcodec.utils.frontend_utils import split_paragraph
                texts = split_paragraph(
                    target_text,
                    None,
                    "zh",
                    token_max_n=60 - prompt_len_tmp,
                    token_min_n=40 - prompt_len_tmp,
                    merge_len=20,
                    comma_split=False,
                )
            elif prompt_language == 'ja':
                from dualcodec.utils.frontend_utils import split_paragraph
                texts = split_paragraph(
                    target_text,
                    None,
                    "zh",
                    token_max_n=70,
                    token_min_n=60,
                    merge_len=20,
                    comma_split=False,
                )
            elif prompt_language == 'en':
                from dualcodec.utils.frontend_utils import split_paragraph
                texts = split_paragraph(
                    target_text,
                    self.tokenizer.encode,
                    "en",
                    token_max_n=70 - prompt_len_tmp,
                    token_min_n=60 - prompt_len_tmp,
                    merge_len=20,
                    comma_split=True,
                )
            else:
                texts = [target_text]
        if prompt_language == 'en':
            texts = [prompt_text + ' ' + t for t in texts]
        else:
            texts = [prompt_text + t for t in texts]
        print(texts)

        for text in texts:

            if self.normalize:
                from dualcodec.dataset.processor import normalize
                text = list(normalize([{
                    'language': prompt_language,
                    'text': text,
                }], en_punct=True, use_kana=False))[0]['text']
            print(text)


            prompt_text_tokens = torch.tensor(
                [
                    [self.tokenizer.to_language_token(prompt_language)]
                    + self.tokenizer.encode(text)
                ],
                dtype=torch.int32,
                device=self.device,
            )
            prompt_text_len = torch.tensor(
                [prompt_text_tokens.shape[-1]], device=self.device
            )

            # target_text_tokens = torch.tensor(
            #     [tokenizer.encode(target_text)], dtype=torch.int32
            # )
            # target_text_len = torch.tensor([target_text_tokens.shape[-1]])

            text_token = prompt_text_tokens

            feature_extractor = self.cfg.feature_extractor
            inputs = feature_extractor(
                prompt_speech, sampling_rate=16000, return_tensors="pt"
            )
            input_features = inputs["input_features"][0]
            attention_mask = inputs["attention_mask"][0]

            input_features = input_features.unsqueeze(0).to(self.device)
            attention_mask = attention_mask.unsqueeze(0).to(self.device)

            batch = {
                "input_features": input_features,
                "attention_mask": attention_mask,
                "text_token": text_token,
                "text_token_len": prompt_text_len,
                "top_k": top_k,
                "top_p": top_p,
                'repeat_penalty': repeat_penalty,
                'temperature': temp,
                'speech': torch.tensor(speech_24k, device=self.device),
            }
            result = self._inference_batch(batch)
            return_values_0.append(result[0])
            # print(result[0][-1])
            # if prompt_language == 'en':
            #     return_values_0.append(result[0][0,-1] * torch.ones_like(result[0][:, :10]))
            return_values_1.append(result[1])
        return torch.cat(return_values_0, dim=1), return_values_1[0], input_features, attention_mask

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def _extract_semantic_code(self, input_features, attention_mask):
        """
            从输入特征中提取语义编码。
        该函数不需要梯度，因此被标记为@torch.no_grad().

        Args:
            input_features (torch.Tensor, shape=(B, T, C)): 输入特征，其中B是batch size，T是时间维度，C是通道维度。
            attention_mask (torch.Tensor, shape=(B, T)): 注意力掩码，其中元素为0表示对应位置的特征无效，非0表示有效。

        Returns:
            tuple (torch.Tensor, shape=(B, T)): 返回一个元组，包含语义编码和对应的量化索引（可选）。
                - semantic_code (torch.Tensor, shape=(B, T)): 语义编码，其中B是batch size，T是时间维度。
                - rep_index (Optional, torch.Tensor, shape=(B, T)): 对于每个时间步骤，如果存在对应的量化索引，则返回这些索引；否则返回None。
        """
        vq_emb = self.cfg.semantic_model["model"](
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[self.cfg.semantic_model["output_idx"]]  # (B, T, C)

        if (
            hasattr(self.cfg, "skip_semantic_normalize")
            and self.cfg.skip_semantic_normalize
        ):
            pass
        else:
            feat = (feat - self.cfg.semantic_model["mean"]) / self.cfg.semantic_model[
                "std"
            ]

        if hasattr(self.cfg, 'use_our_codec'):
            # mean pool to 25hz
            feat = torch.nn.functional.avg_pool1d(feat.transpose(1,2), 2, 2)
            # if feat.shape[-1] % 2 != 0:
            #     feat = feat[..., 1:]

            semantic_code = self.cfg.semantic_model["repcodec_model"].semantic_quantize(feat)
        else:
            semantic_code, _ = self.cfg.semantic_model["repcodec_model"].quantize(
                feat
            )  # (B, T)
        return semantic_code, None

    @torch.no_grad()
    def _inference_batch(self, batch):
        """
        Infer a batch of data using the model.

        Args:
            batch (dict): A dictionary containing the input data, including "input_features" and "attention_mask".
                The keys are expected to be "input_features", "attention_mask", "text_token", "text_token_len",
                "prompt_text", "prompt_text_len", "prompt_speech_token", "prompt_speech_token_len", "embedding".
                All values should be tensors.

        Returns:
            tuple (dict, torch.Tensor):
                - dict: A dictionary containing the output of the model, including "logits", "output_lengths",
                    "generated_hypo", "generated_scores", "generated_ids", "generated_token_num", "generated_eos_num".
                    All values are tensors.
                - torch.Tensor: The semantic code generated by the model.
        """
        # limit the length of input features
        device = self.device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        input_features = batch["input_features"]
        attention_mask = batch["attention_mask"]

        # prompt semantic codes
        # semantic_code, _ = self._extract_semantic_code(input_features, attention_mask)
        semantic_codes, acoustic_codes = self._extract_codes_dac(input_features, attention_mask, batch['speech'])
        semantic_codes = rearrange(semantic_codes, 'b t -> b t 1')
        num_codec_layers = len(self.cfg.offset_sizes)
        semantic_code = torch.cat([semantic_codes, acoustic_codes], dim=-1)[..., :num_codec_layers]

        semantic_code = offset_codes(semantic_code, self.cfg.offset_sizes)
        semantic_code = rearrange(semantic_code, 'b t q -> b (t q)')

        ret_semantic_code = semantic_code.clone().detach()

        out = self.model.inference(
            text=batch["text_token"],
            text_len=batch["text_token_len"],
            prompt_text=None,
            prompt_text_len=None,
            prompt_speech_token=semantic_code,
            prompt_speech_token_len=torch.tensor([semantic_code.shape[-1]]),
            top_k=batch["top_k"],
            top_p=batch['top_p'],
            repeat_penalty=batch['repeat_penalty'],
            temperature=batch['temperature'],
        )
        out = deoffset_codes(out, self.cfg.offset_sizes)
        # out = self.cfg.semantic_model["repcodec_model"].decode_from_codes(out)
        return out, None
