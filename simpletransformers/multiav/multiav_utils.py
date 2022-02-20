import logging
import os
import pickle
from multiprocessing import Pool
from functools import partial
from typing import Tuple, Dict, List, Union, Any

import pandas as pd
import torch
import transformers

from torch.utils.data import Dataset
from tqdm.auto import tqdm

from transformers.models.bart.modeling_bart import (
    shift_tokens_right as _shift_tokens_right,
)
from datasets import Features, Sequence, Value, load_dataset
from datasets import Dataset as HFDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if transformers.__version__ < "4.2.0":
    shift_tokens_right = (
        lambda input_ids, pad_token_id, decoder_start_token_id: _shift_tokens_right(
            input_ids, pad_token_id
        )
    )
else:
    shift_tokens_right = _shift_tokens_right


def preprocess_batch_for_hf_dataset(
    dataset, encoder_tokenizer, decoder_tokenizer, args
):
    if args.model_type == "bart":
        input_ids = encoder_tokenizer.batch_encode_plus(
            dataset["input_text"],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )

        target_ids = encoder_tokenizer.batch_encode_plus(
            dataset["target_text"],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )

        return {
            "source_ids": input_ids["input_ids"].squeeze(),
            "source_mask": input_ids["attention_mask"].squeeze(),
            "target_ids": target_ids["input_ids"].squeeze(),
        }
    elif args.model_type == "mbart":
        tokenized_example = encoder_tokenizer.prepare_seq2seq_batch(
            src_texts=dataset["input_text"],
            tgt_texts=dataset["target_text"],
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            max_length=args.max_seq_length,
            padding="max_length",  # pad_to_max_length=True won't work in this case
            return_tensors="np",
            truncation=True,
        )

        decoder_input_ids = tokenized_example["labels"].clone()
        decoder_input_ids = shift_tokens_right(
            decoder_input_ids,
            encoder_tokenizer.pad_token_id,
            encoder_tokenizer.lang_code_to_id[args.tgt_lang],
        )

        labels = tokenized_example["labels"]
        labels[labels == encoder_tokenizer.pad_token_id] = -100

        return {
            "input_ids": tokenized_example["input_ids"].squeeze(),
            "attention_mask": tokenized_example["attention_mask"].squeeze(),
            "decoder_input_ids": decoder_input_ids.squeeze(),
            "labels": labels.squeeze(),
        }

    elif args.model_type in ["rag-token", "rag-sequence"]:
        source_inputs = encoder_tokenizer(
            dataset["input_text"],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )
        try:
            target_inputs = encoder_tokenizer.generator(
                dataset["target_text"],
                max_length=args.max_seq_length,
                padding="max_length",
                return_tensors="np",
                truncation=True,
            )
        except (TypeError, ValueError) as e:
            logger.warn(e)
            logger.warn(
                """Error encountered while converting target_text.
            All target_text values have been manually cast to String as a workaround.
            This may have been caused by NaN values present in the data."""
            )
            dataset["target_text"] = [str(d) for d in dataset["target_text"]]
            target_inputs = encoder_tokenizer.generator(
                dataset["target_text"],
                max_length=args.max_seq_length,
                padding="max_length",
                return_tensors="np",
                truncation=True,
            )
        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }
    else:
        source_inputs = encoder_tokenizer(
            dataset["input_text"],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )

        target_inputs = decoder_tokenizer(
            dataset["target_text"],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="np",
            truncation=True,
        )
        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }


def load_hf_dataset(data, encoder_tokenizer, decoder_tokenizer, args):
    if isinstance(data, str):
        dataset = load_dataset(
            "csv",
            data_files=data,
            delimiter="\t",
            download_mode="force_redownload"
            if args.reprocess_input_data
            else "reuse_dataset_if_exists",
            cache_dir=args.dataset_cache_dir,
        )
    else:
        dataset = HFDataset.from_pandas(data)

    dataset = dataset.map(
        lambda x: preprocess_batch_for_hf_dataset(
            x,
            encoder_tokenizer=encoder_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            args=args,
        ),
        batched=True,
    )

    if args.model_type == "bart":
        column_names = [
            "source_ids",
            "source_mask",
            "target_ids",
        ]
    elif args.model_type == "mbart":
        column_names = [
            "input_ids",
            "attention_mask",
            "decoder_input_ids",
            "labels",
        ]
    else:
        column_names = [
            "input_ids",
            "attention_mask",
            "decoder_input_ids",
        ]

    dataset.set_format(type="pt", columns=column_names)

    if isinstance(data, str):
        # This is not necessarily a train dataset. The datasets library insists on calling it train.
        return dataset["train"]
    else:
        return dataset


def preprocess_data_bart(
    data: Tuple[str, List[Dict[str, str]], Any, Dict], mode: str, tf: bool
) -> Dict[str, Union[torch.tensor, List[torch.tensor]]]:
    """
    [summary]

    Parameters
    ----------
    data : [type]
        [description]
    mode : [str]
        "train" or "dev" (eval). if train, we use the Seq2SeqTF approach to
        split the data across multiple input output pairs.

    Returns
    -------
    Dict[str, Union[torch.tensor, List[torch.tensor]]]
        dict with keys "source_ids", "source_mask", "target_ids".
        source_ids and source_mask are torch.tensor.
        target_ids is a list of torch.tensor, each containing the ids for
        one of the labels.
    """
    input_text, target_texts, tokenizer, args = data

    def teacher_force_data(
        input_text: str, target_dicts: List[Dict[str, str]]
    ) -> Tuple[List[str], List[str]]:
        # TODO: check that all rows have the same attributes?
        attributes = target_dicts[0].keys()
        # print(attributes)
        input_texts = []
        target_texts = []

        for target_dict in target_dicts:
            text = input_text
            # try:
            for attribute in attributes:
                text = text + f"; {attribute}:"
                response = target_dict[attribute]
                input_texts.append(text)
                target_texts.append(response)
                # print(text[0:10] + "..." + text[-50:], "--", response)
                text += " " + response
            # except Exception as e:
            # logging.error(e)

        return input_texts, target_texts

    if mode == "train":
        # if tf:
        input_texts, target_texts = teacher_force_data(input_text, target_texts)
        # print(input_texts, target_texts)
        input_ids = tokenizer.batch_encode_plus(
            input_texts,
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        target_ids = tokenizer.batch_encode_plus(
            target_texts,
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
    out = {
        "source_ids": input_ids["input_ids"].squeeze(),
        "source_mask": input_ids["attention_mask"].squeeze(),
        "target_ids": target_ids["input_ids"].squeeze(),
    }
    out_final = []
    for i in range(len(target_ids["input_ids"])):
        out_indv = {
            "source_ids": out["source_ids"][i, :],
            "source_mask": out["source_mask"][i, :],
            "target_ids": out["target_ids"][i, :],
        }
        out_final.append(out_indv)
    return out_final


class MultiAVDataset(Dataset):
    def __init__(
        self, tokenizer, args, data: List[Tuple[str, List[Dict[str, str]]]], mode: str
    ):
        self.tokenizer = tokenizer
        self.mode = mode

        cached_features_file = os.path.join(
            args.cache_dir,
            args.model_name + "_cached_" + str(args.max_seq_length) + str(len(data)),
        )

        if os.path.exists(cached_features_file) and (
            (not args.reprocess_input_data and not args.no_cache)
            or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s", args.cache_dir)

            data = [
                (input_text, target_texts, tokenizer, args)
                for input_text, target_texts in data
            ]

            if (mode == "train" and args.use_multiprocessing) or (
                mode == "dev" and args.use_multiprocessing_for_evaluation
            ):
                raise NotImplementedError()
                # if args.multiprocessing_chunksize == -1:
                #     chunksize = max(len(data) // (args.process_count * 2), 500)
                # else:
                #     chunksize = args.multiprocessing_chunksize

                # with Pool(args.process_count) as p:
                #     self.examples = list(
                #         tqdm(
                #             p.imap(preprocess_data_bart, data, chunksize=chunksize),
                #             total=len(data),
                #             disable=args.silent,
                #         )
                #     )
            else:
                exs = []
                for data_tuple in tqdm(data, disable=args.silent):
                    exs += preprocess_data_bart(data_tuple, mode=mode, tf=args.tf)
                self.examples = (
                    exs  # List[Dict[str, Union[torch.tensor, List[torch.tensor]]]]
                )
                # print(exs)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> Dict[str, Union[torch.tensor, List[torch.tensor]]]:
        return self.examples[index]
