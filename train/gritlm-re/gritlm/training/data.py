import logging
import math
import random
from dataclasses import dataclass
from typing import Iterator, List, Tuple, Union

import torch
from transformers import BatchEncoding, DataCollatorWithPadding, PreTrainedTokenizer

import datasets

from .arguments import DataArguments

logger = logging.getLogger(__name__)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        args: DataArguments,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 2048,
    ):
        # Only embedding mode is supported
        self.ds_embedding = dataset
        self.total_len = self.len_embedding = len(self.ds_embedding)
        self.args = args
        self.tokenizer = tokenizer

        # Too long items will be stuck in communication so cut them on the fly
        self.max_char_len = max_seq_len * 10

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        """
        Returns:
            query: Query text (str or list)
            passages: List of passage texts [positive, neg1, neg2, ...]
        """
        query = self.ds_embedding[item]["query"]

        if isinstance(query, str):
            query = query[: self.max_char_len]
        elif isinstance(query, list):
            query = [x[: self.max_char_len] for x in query]

        passages = []
        pos = random.choice(self.ds_embedding[item]["pos"])

        if isinstance(pos, str):
            pos = pos[: self.max_char_len]
        elif isinstance(pos, list):
            pos = [x[: self.max_char_len] for x in pos]
        else:
            raise ValueError(f"Unexpected type for pos: {type(pos)}")
        passages.append(pos)

        if len(self.ds_embedding[item]["neg"]) == 0:  # @lucaswychan add checking of no negs since I may only use in-batch negs
            negs = []
        else:
            if len(self.ds_embedding[item]["neg"]) < self.args.train_group_size - 1:
                num = math.ceil((self.args.train_group_size - 1) / len(self.ds_embedding[item]["neg"]))
                negs = random.sample(self.ds_embedding[item]["neg"] * num, self.args.train_group_size - 1)
            else:
                negs = random.sample(self.ds_embedding[item]["neg"], self.args.train_group_size - 1)

            for i, neg in enumerate(negs):
                if isinstance(neg, str):
                    negs[i] = neg[: self.max_char_len]
                elif isinstance(neg, list):
                    negs[i] = [x[: self.max_char_len] for x in neg]
                else:
                    raise ValueError(f"Unexpected type for neg: {type(neg)}")
        passages.extend(negs)

        return query, passages


@dataclass
class CustomCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    query_max_len: int = 32
    passage_max_len: int = 128

    base_bos: str = ""

    user_bos: str = ""
    user_eos: str = ""

    embed_bos: str = ""
    # Am embed eos is useless as there is no generative loss on it so it won't be learned
    # & it does not add anything new; It only makes sense for lasttoken pooling
    embed_eos: str = ""

    def __post_init__(self):
        # Cache for tokenization patterns to avoid repeated string operations
        self._base_embed_prefix = self.base_bos + self.embed_bos.lstrip()
        self._user_embed_prefix = self.base_bos + self.user_bos
        self._embed_suffix = self.user_eos + self.embed_bos

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]

        # Flatten if list of lists
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        features = {}

        # If each sample is a tuple it is of format (instruction, text)
        # @ lucaswychan remove .strip("\t\n :")
        q_instruction_lens = None
        if isinstance(query[0], (tuple, list)):
            # Pre-build all query strings first for batch tokenization
            query_strs = []
            instr_strs = []
            for f in query:
                if f[0]:
                    instr_str = self._user_embed_prefix + f[0] + self._embed_suffix
                    query_str = instr_str + f[1] + self.embed_eos
                else:
                    instr_str = self._base_embed_prefix
                    query_str = instr_str + f[1] + self.embed_eos
                instr_strs.append(instr_str)
                query_strs.append(query_str)

            # Batch tokenize instruction strings for better performance
            q_instruction_lens = [len(self.tokenizer.tokenize(s)) for s in instr_strs]
            query = query_strs

            # @lucaswychan add checking of passage type, since original approach will assume there is instruction in the passage
            # if the query has instruction, then the passage will also have instruction
            # and will not work for the case where there is no instruction in the passage
            # and in our case we have no instruction in the passage
            if isinstance(passage[0], (tuple, list)):
                passage_strs = []
                passage_instr_strs = []
                for f in passage:
                    if f[0]:
                        instr_str = self._user_embed_prefix + f[0] + self._embed_suffix
                        passage_str = instr_str + f[1] + self.embed_eos
                    else:
                        instr_str = self._base_embed_prefix
                        passage_str = instr_str + f[1] + self.embed_eos
                    passage_instr_strs.append(instr_str)
                    passage_strs.append(passage_str)

                d_instruction_lens = [len(self.tokenizer.tokenize(s)) for s in passage_instr_strs]
                passage = passage_strs
            else:
                d_instruction_lens = []
                passage = [self._base_embed_prefix + f + self.embed_eos for f in passage]

        features["query"] = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
            add_special_tokens=False,  # BOS / EOS is already in the prompt
        )
        features["passage"] = self.tokenizer(
            passage,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
            add_special_tokens=False,  # BOS / EOS is already in the prompt
        )

        if q_instruction_lens:
            # Check that there is no mistake
            for i, l in enumerate(q_instruction_lens):
                assert features["query"]["input_ids"][i, l] != self.tokenizer.pad_token, f"No text to embed: {query[i]}"
            for i, l in enumerate(d_instruction_lens):
                assert features["passage"]["input_ids"][i, l] != self.tokenizer.pad_token, f"No text to embed: {passage[i]}"
            # Need to be masked out later
            features["query"]["instruction_lens"] = torch.tensor(q_instruction_lens)
            # @lucaswychan: if there is no instruction in the passage, we don't add instruction_lens
            if d_instruction_lens:
                features["passage"]["instruction_lens"] = torch.tensor(d_instruction_lens)

        return features


@dataclass
class CustomRandomSampler(torch.utils.data.sampler.RandomSampler):
    """
    Sampler used when training on multiple datasets to ensure each
    batch only contains samples from one dataset for the majority of cases.
    """

    total_batch_size: int = 8
    ds_lens: List[int] = None
    _num_samples: int = None
    data_source: CustomDataset = None
    replacement: bool = False

    @torch.no_grad()
    def __iter__(self) -> Iterator[int]:

        if not hasattr(self, "generator") or self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # We have multiple datasets each with a different number of samples
        # e.g. [100, 150, 50]
        # We would like to sample from them such that as much as possible each batch
        # only has samples from the same dataset.
        # For example if our batch size is 4 then
        # indices might be [0,1,2,3,100,101,102,103,150,151,152,153,50,51,52,53]
        # To do so:
        # 1. Shuffle the indices of each dataset separately
        # 2. Create batches with only samples from one dataset
        # 3. Keep the remaining samples which do not fit into a batch separate
        # 4. Then create mixed batches from the remaining samples
        # 5. Then yield randomly from all the batches
        # Testing:
        # ds_lens = [100, 150, 50]
        # batch_size = 8
        # Create random indices for each dataset
        ds_indices = [torch.randperm(n, generator=generator).tolist() for n in self.ds_lens]
        # Increase the indices to be indices of the concatenated dataset
        ds_indices = [[i + sum(self.ds_lens[:j]) for i in ds_indices[j]] for j in range(len(self.ds_lens))]
        # Create batches with only samples from one dataset
        ds_batches = [list(torch.split(torch.tensor(ds_indices[j]), self.total_batch_size)) for j in range(len(self.ds_lens))]
        # Create separate batches from the remaining samples
        incomplete_indices = []
        for b in ds_batches:
            if len(b[-1]) < self.total_batch_size:
                incomplete_indices.append(b.pop())

        if incomplete_indices:
            # Randomly permute the incomplete indices
            order = torch.randperm(len(incomplete_indices), generator=generator).tolist()
            incomplete_indices = torch.cat([incomplete_indices[i] for i in order])
            # Then split again into groups of four & drop the last one if it is incomplete
            mixed_batches = list(torch.split(incomplete_indices, self.total_batch_size))
            if len(mixed_batches[-1]) < self.total_batch_size:
                mixed_batches.pop()
            # Merge all batches to look like [...tensor([259, 273, 284, 289]), tensor([262, 280, 295, 258]), ...]
            ds_batches = sum(ds_batches, []) + mixed_batches
            logger.info(f"Using global batch size {self.total_batch_size} created {len(ds_batches) - len(mixed_batches)} single-dataset batches & {len(mixed_batches)} mixed dataset batches.")
        else:
            ds_batches = sum(ds_batches, [])
            logger.info(f"Using global batch size {self.total_batch_size} created {len(ds_batches)} single-dataset batches.")

        # Randomly permute the order of all batches, then merge them to look like tensor([...259, 273, 284, 289, 262, 280, 295, 258...])
        order = torch.randperm(len(ds_batches), generator=generator).tolist()
        ds_batches = [int(i) for i in torch.cat([ds_batches[i] for i in order]).tolist()]
        # Yield the indices
        yield from ds_batches
