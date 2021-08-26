#!/usr/bin/env python
import argparse
import logging
import os
from pathlib import Path
import torch

import pytorch_lightning as pl
from typing import Dict

from utils import (
    Seq2SeqDataset,
    check_output_dir,
)

from transformers.models.bart.modeling_bart import shift_tokens_right
from finetune import SummarizationModule, main

logger = logging.getLogger(__name__)


class ConSumModule(SummarizationModule):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.loss_length_penalty = hparams.loss_length_penalty
        self.margin_value = hparams.margin_value
        self.a = hparams.a
        self.b = hparams.b

    # https://github.com/huggingface/transformers/blob/master/src/transformers/generation_beam_search.py#L361
    # score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
    def cal_loss(self, p_lm_logits, n_lm_logits, p_tgt_ids, n_tgt_ids):
        p_lodprobs = torch.nn.functional.log_softmax(p_lm_logits, dim=-1)
        n_lodprobs = torch.nn.functional.log_softmax(n_lm_logits, dim=-1)

        if p_tgt_ids.dim() == p_lodprobs.dim() - 1:
            p_tgt_ids = p_tgt_ids.unsqueeze(-1)
        if n_tgt_ids.dim() == n_lodprobs.dim() - 1:
            n_tgt_ids = n_tgt_ids.unsqueeze(-1)

        p_logprobs = p_lodprobs.gather(dim=-1, index=p_tgt_ids)
        n_logprobs = n_lodprobs.gather(dim=-1, index=n_tgt_ids)

        p_pad_mask = p_tgt_ids.eq(self.tokenizer.pad_token_id)
        n_pad_mask = n_tgt_ids.eq(self.tokenizer.pad_token_id)

        p_logprobs.masked_fill_(p_pad_mask, 0.0)
        n_logprobs.masked_fill_(n_pad_mask, 0.0)

        p_logprobs = p_logprobs.squeeze(-1)
        n_logprobs = n_logprobs.squeeze(-1)

        p_pad_mask = p_pad_mask.squeeze(-1)
        n_pad_mask = n_pad_mask.squeeze(-1)

        p_sum_logprobs = p_logprobs.sum(dim=-1)
        n_sum_logprobs = n_logprobs.sum(dim=-1)

        p_length = p_tgt_ids.size(1) - p_pad_mask.sum(dim=-1)
        n_length = n_tgt_ids.size(1) - n_pad_mask.sum(dim=-1)

        p_score = p_sum_logprobs / (p_length ** self.loss_length_penalty)
        n_score = n_sum_logprobs / (n_length ** self.loss_length_penalty)

        ce_loss = - p_sum_logprobs / p_length

        con_loss = torch.nn.functional.relu(n_score - p_score + self.margin_value)

        loss = con_loss * self.a + ce_loss * self.b

        loss = loss.mean()

        return loss, -p_score.mean(), -n_score.mean()

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        SummarizationModule.add_model_specific_args(parser, root_dir)
        add_contrastive_args(parser)
        return parser

    def calculate_lm_logits(self, src_ids, src_mask, tgt_ids):
        pad_token_id = self.tokenizer.pad_token_id
        decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id)
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        return outputs["logits"]

    def training_step(self, batch, batch_idx) -> Dict:
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]
        self.model.eval()
        silver_tgt_ids = self.model.generate(
            src_ids,
            attention_mask=src_mask,
            use_cache=True,
            decoder_start_token_id=self.decoder_start_token_id,
            num_beams=1,
            max_length=self.eval_max_length,
        )
        self.model.train()

        p_lm_logits = self.calculate_lm_logits(src_ids, src_mask, tgt_ids)
        n_lm_logits = self.calculate_lm_logits(src_ids, src_mask, silver_tgt_ids)

        loss, p_loss, n_loss = self.cal_loss(p_lm_logits, n_lm_logits, tgt_ids, silver_tgt_ids)

        loss_tensors = (loss,)

        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        logs["p_loss"] = p_loss
        logs["n_loss"] = n_loss
        return {"loss": loss_tensors[0], "log": logs}


def add_contrastive_args(parser):
    parser.add_argument("--loss_length_penalty", default=0.6, type=float, required=False)
    parser.add_argument("--margin_value", default=0.0, type=float, required=False)
    parser.add_argument("--a", default=1.0, type=float, required=False)
    parser.add_argument("--b", default=1.0, type=float, required=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ConSumModule.add_model_specific_args(parser, os.getcwd())

    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)
    check_output_dir(args, expected_items=3)

    model = ConSumModule(args)

    main(args, model=model)
