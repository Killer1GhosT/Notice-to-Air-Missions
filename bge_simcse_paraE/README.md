---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:283
- loss:MultipleNegativesRankingLoss
base_model: BAAI/bge-large-en-v1.5
widget:
- source_sentence: RWY27 EDGE Lighting Unserviceable
  sentences:
  - 'FOUR LETTER LOCATION INDICATOR VADL IS ASSIGNED TO DHOLERA INTERNATIONAL AIRPORT
    IN MUMBAI FIR WITH GEOGRAPHICAL Coordinates OF Aerodrome Reference Point AS 222143.82628N
    0721803.92172E. INSERT IN GEN2.4 OF E-AIP INDIA. CREATED: 16 Apr 2025 08:55:00  SOURCE:
    VIDPYNYX'
  - RWY27 EDGE Lighting Unserviceable
  - NAGPUR FLYING CLUB WILL BE CONDUCTING TRAINING FLIGHT     Within 10NM RADIUS OF
    CENTRE Coordinates:195544.26N0791331E
- source_sentence: 'HF MAIN RX  RDARA 8861/8948KHZ  NOT Available  WKG ON S/BY RX
    ON 8861KHZ  ONLY. CREATED: 02 Sep 2010 03:27:00  SOURCE: VECCYNYX'
  sentences:
  - 'RWY16R/34L Closed,Aircraft SHALL CROSS RWY16R/34L VIA Taxiway  R2,R3,R4,R5. CREATED:
    18 Jun 2025 04:44:00  SOURCE: ZBBBYNYX'
  - 'HF MAIN RX  RDARA 8861/8948KHZ  NOT Available  WKG ON S/BY RX ON 8861KHZ  ONLY.
    CREATED: 02 Sep 2010 03:27:00  SOURCE: VECCYNYX'
  - 'Runway End Safety Area DIMENSIONS CHANGED AS Follow(s): RWY01:LENGTH 220M WIDTH
    120M CHANGED TO LENGTH 240M WIDTH 120M. CREATED: 06 Jun 2025 08:14:00  SOURCE:
    ZBBBYNYX'
- source_sentence: 'RWY18L/36R Closed.Aircraft ARE FORBIDDEN TO CROSS RWY18L/36R VIA
    Taxiway  A0,A1,A8,A9. CREATED: 13 Jun 2025 06:21:00  SOURCE: ZBBBYNYX'
  sentences:
  - 'ADVANCED VISUAL Docking GUIDANCE SYSTEM(AVDGS) FOR STAND 36 NOT Available. PILOTS
    TO Follow(s) MARSHALLER FOR Docking CREATED: 30 May 2025 11:30:00  SOURCE: VOMMYNYX'
  - 'RWY17R/35L Closed,Aircraft SHALL CROSS RWY17R/35L VIA Taxiway P1,P4. CREATED:
    18 Jun 2025 04:46:00  SOURCE: ZBBBYNYX'
  - 'RWY18L/36R Closed.Aircraft ARE FORBIDDEN TO CROSS RWY18L/36R VIA Taxiway  A0,A1,A8,A9.
    CREATED: 13 Jun 2025 06:21:00  SOURCE: ZBBBYNYX'
- source_sentence: 'Taxiway L10,L11,L12,L12A Closed. CREATED: 16 Jun 2025 03:56:00  SOURCE:
    ZBBBYNYX'
  sentences:
  - 'Work In Progress ON THE BASIC STRIP OF RWY07/25 Between  Taxiway D AND Taxiway
    M WITH THE Follow(s) DETAILS:  100M Distance From Centre Line OF RWY07/25 AND
    60M Distance From THE Centre Line OF Taxiway B.PILOTS TO Exercise Caution CREATED:
    30 Apr 2025 04:44:00  SOURCE: VOMMYNYX'
  - 'TXL N1, TXL N3, TXL N4 AND Parking STANDS 201-218 NOT Available FOR Operations
    DUE Surface REPAIR WORKS. CREATED: 16 Apr 2025 07:50:00  SOURCE: VIDPYNYX'
  - 'Taxiway L10,L11,L12,L12A Closed. CREATED: 16 Jun 2025 03:56:00  SOURCE: ZBBBYNYX'
- source_sentence: 'Transmitter NOT Available DUE POWER FAILURE AT Transmitter STATION.    Operations
    ON LOW POWER Transmitter, SELCAL NOT Available FOR    Frequency 8948KHZ. CREATED:
    07 Apr 2023 12:16:00  SOURCE: VABBYNYX'
  sentences:
  - 'Transmitter NOT Available DUE POWER FAILURE AT Transmitter STATION.    Operations
    ON LOW POWER Transmitter, SELCAL NOT Available FOR    Frequency 8948KHZ. CREATED:
    07 Apr 2023 12:16:00  SOURCE: VABBYNYX'
  - 'Refer To AIP PAGE Aerodrome 2 VECC 2-201 REGARDING VOR Runway 01L Procedure OF
    KOLKATA AIRPORT. CIRCLING Oceanic Control Area (H) FOR CAT C/D Aircraft IS AMD
    AS 880 (860) FT. CREATED: 25 Jul 2024 07:24:00  SOURCE: VIDPYNYX'
  - 'ACTIVITIES OF BIRD FLOCKS TAKE PLACE AROUND Aerodrome:  HEIGHT: 100M-200M. AERODROME
    AUTHORITY RESORTS TO DISPERSAL METHODS TO REDUCE BIRD  ACTIVITIES. EXERCISE CAUTION
    WHILE LANDING AND TAKE-OFF. CREATED: 12 Jun 2025 10:55:00  SOURCE: ZBBBYNYX'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on BAAI/bge-large-en-v1.5

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5). It maps sentences & paragraphs to a 1024-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) <!-- at revision d4aa6901d3a41ba39fb536a557fa166f842b0e09 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 1024 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': True}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Transmitter NOT Available DUE POWER FAILURE AT Transmitter STATION.    Operations ON LOW POWER Transmitter, SELCAL NOT Available FOR    Frequency 8948KHZ. CREATED: 07 Apr 2023 12:16:00  SOURCE: VABBYNYX',
    'Transmitter NOT Available DUE POWER FAILURE AT Transmitter STATION.    Operations ON LOW POWER Transmitter, SELCAL NOT Available FOR    Frequency 8948KHZ. CREATED: 07 Apr 2023 12:16:00  SOURCE: VABBYNYX',
    'Refer To AIP PAGE Aerodrome 2 VECC 2-201 REGARDING VOR Runway 01L Procedure OF KOLKATA AIRPORT. CIRCLING Oceanic Control Area (H) FOR CAT C/D Aircraft IS AMD AS 880 (860) FT. CREATED: 25 Jul 2024 07:24:00  SOURCE: VIDPYNYX',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 283 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 283 samples:
  |         | sentence_0                                                                          | sentence_1                                                                          |
  |:--------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                              |
  | details | <ul><li>min: 10 tokens</li><li>mean: 98.55 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 10 tokens</li><li>mean: 98.55 tokens</li><li>max: 512 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                  | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                  |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Refer To EAIP Aerodrome 2 VABB 3-905 31 December 2020 REGARDING Required Navigation Performance Y Runway 14 Procedure -MUMBAI: IN PROFILE VIEW TRANSITION Altitude IS AMENDED AS 6000FT. AMEND ACCORDINGLY.   CNS123W SEQUENCE CHECK FOR VIDPYNYX G --    EXPECTED G0182/25 -- RECEIVED G0181/25 CNN038W DUPLICATE NOTAM TRANSACTION ROLLED BACK, RETCODE = 14 CREATED: 12 Feb 2025 06:59:00  SOURCE: VIDPYNYX</code> | <code>Refer To EAIP Aerodrome 2 VABB 3-905 31 December 2020 REGARDING Required Navigation Performance Y Runway 14 Procedure -MUMBAI: IN PROFILE VIEW TRANSITION Altitude IS AMENDED AS 6000FT. AMEND ACCORDINGLY.   CNS123W SEQUENCE CHECK FOR VIDPYNYX G --    EXPECTED G0182/25 -- RECEIVED G0181/25 CNN038W DUPLICATE NOTAM TRANSACTION ROLLED BACK, RETCODE = 14 CREATED: 12 Feb 2025 06:59:00  SOURCE: VIDPYNYX</code> |
  | <code>Refer To EAIP Aerodrome 2 VABB 2-203 25 March 2021 REGARDING VOR Runway 27 Procedure -MUMBAI: IN PROFILE VIEW TRANSITION Altitude IS AMENDED AS 6000FT. AMEND ACCORDINGLY.   CNS123W SEQUENCE CHECK FOR VIDPYNYX G --    EXPECTED G0179/25 -- RECEIVED G0178/25 CNN038W DUPLICATE NOTAM TRANSACTION ROLLED BACK, RETCODE = 14 CREATED: 12 Feb 2025 06:54:00  SOURCE: VIDPYNYX</code>                                  | <code>Refer To EAIP Aerodrome 2 VABB 2-203 25 March 2021 REGARDING VOR Runway 27 Procedure -MUMBAI: IN PROFILE VIEW TRANSITION Altitude IS AMENDED AS 6000FT. AMEND ACCORDINGLY.   CNS123W SEQUENCE CHECK FOR VIDPYNYX G --    EXPECTED G0179/25 -- RECEIVED G0178/25 CNN038W DUPLICATE NOTAM TRANSACTION ROLLED BACK, RETCODE = 14 CREATED: 12 Feb 2025 06:54:00  SOURCE: VIDPYNYX</code>                                  |
  | <code>Refer To AIP PAGE Aerodrome 2 VECC 2-205 REGARDING VOR Runway 19L Procedure OF KOLKATA AIRPORT. CIRCLING Oceanic Control Area (H) FOR CAT C/D Aircraft IS AMD AS 880 (860) FT. CREATED: 25 Jul 2024 07:16:00  SOURCE: VIDPYNYX</code>                                                                                                                                                                                 | <code>Refer To AIP PAGE Aerodrome 2 VECC 2-205 REGARDING VOR Runway 19L Procedure OF KOLKATA AIRPORT. CIRCLING Oceanic Control Area (H) FOR CAT C/D Aircraft IS AMD AS 880 (860) FT. CREATED: 25 Jul 2024 07:16:00  SOURCE: VIDPYNYX</code>                                                                                                                                                                                 |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 4
- `per_device_eval_batch_size`: 4
- `num_train_epochs`: 2
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 4
- `per_device_eval_batch_size`: 4
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Framework Versions
- Python: 3.12.4
- Sentence Transformers: 3.1.1
- Transformers: 4.52.4
- PyTorch: 2.5.1+cu121
- Accelerate: 1.8.1
- Datasets: 3.6.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->