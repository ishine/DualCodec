import torch

class ValleARInference:
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
        # torch._C._jit_set_fusion_strategy([('STATIC', 1)])
        # torch._C._jit_set_profiling_mode(False)
        # torch._C._jit_set_profiling_executor(False)
        
        # self.model.llm = torch.compile(self.model.llm.forward_chunk, dynamic=True)
        # self.model.text_encoder.forward = torch.compile(self.model.text_encoder.forward, dynamic=True)
        # self.model.fast_llm = torch.compile(self.model.fast_llm, dynamic=False)

        self.split_paragraph = split_paragraph

    @torch.inference_mode()
    def inference(
        self,
        text,
        prompt_speech,
        prompt_language,
        temp=1.0,
        top_k=1000,
        top_p=0.85,
        repeat_penalty=1.1,
        return_prompt=False
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

        # print(text)
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
        }
        result = self._inference_batch(batch,return_prompt=return_prompt)
        # return_values_0.append(result[0])
        # print(result[0][-1])
        # if prompt_language == 'en':
        #     return_values_0.append(result[0][0,-1] * torch.ones_like(result[0][:, :10]))
        # return_values_1.append(result[1])
        if return_prompt:
            return result[0], result[1], input_features, attention_mask
        else:
            return result, None, input_features, attention_mask

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def _extract_semantic_code(self, input_features, attention_mask):
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
            feat = torch.nn.functional.avg_pool1d(feat.transpose(1,2), self.cfg.semantic_model['repcodec_model'].semantic_downsample_factor, self.cfg.semantic_model['repcodec_model'].semantic_downsample_factor)
            # if feat.shape[-1] % 2 != 0:
            #     feat = feat[..., 1:]

            semantic_code = self.cfg.semantic_model["repcodec_model"].semantic_quantize(feat)
        else:
            semantic_code, _ = self.cfg.semantic_model["repcodec_model"].quantize(
                feat
            )  # (B, T)
        return semantic_code, None

    @torch.no_grad()
    def _inference_batch(self, batch, return_prompt=False):
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
        semantic_code, _ = self._extract_semantic_code(input_features, attention_mask)

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
        if return_prompt:
            return out, ret_semantic_code
        else:
            return out
