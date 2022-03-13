# ubiquant-market-prediction

## Installation

```bash
# Create virtual environment and activate it (optional)
$ python -m venv env && . env/bin/activate

# Install dependencies
$ python -m pip install -r requirements.txt
```

## Usage

```
$ python main.py --help
usage: main.py [-h] [--logger [LOGGER]] [--checkpoint_callback [CHECKPOINT_CALLBACK]] [--enable_checkpointing [ENABLE_CHECKPOINTING]] [--default_root_dir DEFAULT_ROOT_DIR]
               [--gradient_clip_val GRADIENT_CLIP_VAL] [--gradient_clip_algorithm GRADIENT_CLIP_ALGORITHM] [--process_position PROCESS_POSITION] [--num_nodes NUM_NODES]
               [--num_processes NUM_PROCESSES] [--devices DEVICES] [--gpus GPUS] [--auto_select_gpus [AUTO_SELECT_GPUS]] [--tpu_cores TPU_CORES] [--ipus IPUS]
               [--log_gpu_memory LOG_GPU_MEMORY] [--progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE] [--enable_progress_bar [ENABLE_PROGRESS_BAR]]
               [--overfit_batches OVERFIT_BATCHES] [--track_grad_norm TRACK_GRAD_NORM] [--check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH] [--fast_dev_run [FAST_DEV_RUN]]
               [--accumulate_grad_batches ACCUMULATE_GRAD_BATCHES] [--max_epochs MAX_EPOCHS] [--min_epochs MIN_EPOCHS] [--max_steps MAX_STEPS] [--min_steps MIN_STEPS]
               [--max_time MAX_TIME] [--limit_train_batches LIMIT_TRAIN_BATCHES] [--limit_val_batches LIMIT_VAL_BATCHES] [--limit_test_batches LIMIT_TEST_BATCHES]
               [--limit_predict_batches LIMIT_PREDICT_BATCHES] [--val_check_interval VAL_CHECK_INTERVAL] [--flush_logs_every_n_steps FLUSH_LOGS_EVERY_N_STEPS]
               [--log_every_n_steps LOG_EVERY_N_STEPS] [--accelerator ACCELERATOR] [--strategy STRATEGY] [--sync_batchnorm [SYNC_BATCHNORM]] [--precision PRECISION]
               [--enable_model_summary [ENABLE_MODEL_SUMMARY]] [--weights_summary WEIGHTS_SUMMARY] [--weights_save_path WEIGHTS_SAVE_PATH]
               [--num_sanity_val_steps NUM_SANITY_VAL_STEPS] [--resume_from_checkpoint RESUME_FROM_CHECKPOINT] [--profiler PROFILER] [--benchmark [BENCHMARK]]
               [--deterministic [DETERMINISTIC]] [--reload_dataloaders_every_n_epochs RELOAD_DATALOADERS_EVERY_N_EPOCHS]
               [--reload_dataloaders_every_epoch [RELOAD_DATALOADERS_EVERY_EPOCH]] [--auto_lr_find [AUTO_LR_FIND]] [--replace_sampler_ddp [REPLACE_SAMPLER_DDP]]
               [--detect_anomaly [DETECT_ANOMALY]] [--auto_scale_batch_size [AUTO_SCALE_BATCH_SIZE]] [--prepare_data_per_node [PREPARE_DATA_PER_NODE]] [--plugins PLUGINS]
               [--amp_backend AMP_BACKEND] [--amp_level AMP_LEVEL] [--move_metrics_to_cpu [MOVE_METRICS_TO_CPU]] [--multiple_trainloader_mode MULTIPLE_TRAINLOADER_MODE]
               [--stochastic_weight_avg [STOCHASTIC_WEIGHT_AVG]] [--terminate_on_nan [TERMINATE_ON_NAN]] [--workers WORKERS] [--input INPUT] [--batch_size BATCH_SIZE] [--lr LR]
               [--weight_decay WEIGHT_DECAY] [--seed SEED] [--optimizer {adam,adamw}] [--lr_scheduler LR_SCHEDULER] [--loss {mse,pcc}] [--emb_dim EMB_DIM] [--n_fold N_FOLD]
               [--split_ratios SPLIT_RATIOS [SPLIT_RATIOS ...]] [--early_stop] [--swa] [--n_emb N_EMB] [--szs SZS [SZS ...]] [--mhas MHAS [MHAS ...]] [--dropout DROPOUT]
               [--test] [--checkpoint CHECKPOINT]

optional arguments:
  -h, --help            show this help message and exit
  --workers WORKERS
  --input INPUT         path to train data (default: ../input/ubiquant-parquet/train_low_mem.parquet)
  --batch_size BATCH_SIZE
  --lr LR
  --weight_decay WEIGHT_DECAY
  --seed SEED
  --optimizer {adam,adamw}
  --lr_scheduler LR_SCHEDULER
  --loss {mse,pcc}
  --emb_dim EMB_DIM
  --n_fold N_FOLD
  --split_ratios SPLIT_RATIOS [SPLIT_RATIOS ...]
                        train, val, and test set (optional) split ratio (default: [0.7, 0.15, 0.15])
  --early_stop
  --swa                 whether to perform Stochastic Weight Averaging (default: False)
  --n_emb N_EMB
  --szs SZS [SZS ...]
  --mhas MHAS [MHAS ...]
                        Insert MHA layer (BertLayer) at the i-th layer (start from 1). Every element should be <= len(szs) (default: [])
  --dropout DROPOUT     dropout rate, set to 0.0 to disable (default: 0.0)
  --test
  --checkpoint CHECKPOINT
                        path to checkpoints (for test) (default: None)

pl.Trainer:
  --logger [LOGGER]     Logger (or iterable collection of loggers) for experiment tracking. A ``True`` value uses the default ``TensorBoardLogger``. ``False`` will disable
                        logging. If multiple loggers are provided and the `save_dir` property of that logger is not set, local files (checkpoints, profiler traces, etc.) are
                        saved in ``default_root_dir`` rather than in the ``log_dir`` of any of the individual loggers. (default: True)
  --checkpoint_callback [CHECKPOINT_CALLBACK]
                        If ``True``, enable checkpointing. .. deprecated:: v1.5 ``checkpoint_callback`` has been deprecated in v1.5 and will be removed in v1.7. Please consider
                        using ``enable_checkpointing`` instead. (default: None)
  --enable_checkpointing [ENABLE_CHECKPOINTING]
                        If ``True``, enable checkpointing. It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
                        :paramref:`~pytorch_lightning.trainer.trainer.Trainer.callbacks`. (default: True)
  --default_root_dir DEFAULT_ROOT_DIR
                        Default path for logs and weights when no logger/ckpt_callback passed. Default: ``os.getcwd()``. Can be remote file paths such as `s3://mybucket/path`
                        or 'hdfs://path/' (default: None)
  --gradient_clip_val GRADIENT_CLIP_VAL
                        The value at which to clip gradients. Passing ``gradient_clip_val=None`` disables gradient clipping. If using Automatic Mixed Precision (AMP), the
                        gradients will be unscaled before. (default: None)
  --gradient_clip_algorithm GRADIENT_CLIP_ALGORITHM
                        The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"`` to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by
                        norm. By default it will be set to ``"norm"``. (default: None)
  --process_position PROCESS_POSITION
                        Orders the progress bar when running multiple models on same machine. .. deprecated:: v1.5 ``process_position`` has been deprecated in v1.5 and will be
                        removed in v1.7. Please pass :class:`~pytorch_lightning.callbacks.progress.TQDMProgressBar` with ``process_position`` directly to the Trainer's
                        ``callbacks`` argument instead. (default: 0)
  --num_nodes NUM_NODES
                        Number of GPU nodes for distributed training. (default: 1)
  --num_processes NUM_PROCESSES
                        Number of processes for distributed training with ``accelerator="cpu"``. (default: 1)
  --devices DEVICES     Will be mapped to either `gpus`, `tpu_cores`, `num_processes` or `ipus`, based on the accelerator type. (default: None)
  --gpus GPUS           Number of GPUs to train on (int) or which GPUs to train on (list or str) applied per node (default: None)
  --auto_select_gpus [AUTO_SELECT_GPUS]
                        If enabled and ``gpus`` is an integer, pick available gpus automatically. This is especially useful when GPUs are configured to be in "exclusive mode",
                        such that only one process at a time can access them. (default: False)
  --tpu_cores TPU_CORES
                        How many TPU cores to train on (1 or 8) / Single TPU to train on [1] (default: None)
  --ipus IPUS           How many IPUs to train on. (default: None)
  --log_gpu_memory LOG_GPU_MEMORY
                        None, 'min_max', 'all'. Might slow performance. .. deprecated:: v1.5 Deprecated in v1.5.0 and will be removed in v1.7.0 Please use the
                        ``DeviceStatsMonitor`` callback directly instead. (default: None)
  --progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE
                        How often to refresh progress bar (in steps). Value ``0`` disables progress bar. Ignored when a custom progress bar is passed to
                        :paramref:`~Trainer.callbacks`. Default: None, means a suitable value will be chosen based on the environment (terminal, Google COLAB, etc.). ..
                        deprecated:: v1.5 ``progress_bar_refresh_rate`` has been deprecated in v1.5 and will be removed in v1.7. Please pass
                        :class:`~pytorch_lightning.callbacks.progress.TQDMProgressBar` with ``refresh_rate`` directly to the Trainer's ``callbacks`` argument instead. To
                        disable the progress bar, pass ``enable_progress_bar = False`` to the Trainer. (default: None)
  --enable_progress_bar [ENABLE_PROGRESS_BAR]
                        Whether to enable to progress bar by default. (default: True)
  --overfit_batches OVERFIT_BATCHES
                        Overfit a fraction of training data (float) or a set number of batches (int). (default: 0.0)
  --track_grad_norm TRACK_GRAD_NORM
                        -1 no tracking. Otherwise tracks that p-norm. May be set to 'inf' infinity-norm. If using Automatic Mixed Precision (AMP), the gradients will be
                        unscaled before logging them. (default: -1)
  --check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH
                        Check val every n train epochs. (default: 1)
  --fast_dev_run [FAST_DEV_RUN]
                        Runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es) of train, val and test to find any bugs (ie: a sort of unit test). (default: False)
  --accumulate_grad_batches ACCUMULATE_GRAD_BATCHES
                        Accumulates grads every k batches or as set up in the dict. (default: None)
  --max_epochs MAX_EPOCHS
                        Stop training once this number of epochs is reached. Disabled by default (None). If both max_epochs and max_steps are not specified, defaults to
                        ``max_epochs = 1000``. To enable infinite training, set ``max_epochs = -1``. (default: None)
  --min_epochs MIN_EPOCHS
                        Force training for at least these many epochs. Disabled by default (None). If both min_epochs and min_steps are not specified, defaults to ``min_epochs
                        = 1``. (default: None)
  --max_steps MAX_STEPS
                        Stop training after this number of steps. Disabled by default (-1). If ``max_steps = -1`` and ``max_epochs = None``, will default to ``max_epochs =
                        1000``. To enable infinite training, set ``max_epochs`` to ``-1``. (default: -1)
  --min_steps MIN_STEPS
                        Force training for at least these number of steps. Disabled by default (None). (default: None)
  --max_time MAX_TIME   Stop training after this amount of time has passed. Disabled by default (None). The time duration can be specified in the format DD:HH:MM:SS (days,
                        hours, minutes seconds), as a :class:`datetime.timedelta`, or a dictionary with keys that will be passed to :class:`datetime.timedelta`. (default: None)
  --limit_train_batches LIMIT_TRAIN_BATCHES
                        How much of training dataset to check (float = fraction, int = num_batches). (default: 1.0)
  --limit_val_batches LIMIT_VAL_BATCHES
                        How much of validation dataset to check (float = fraction, int = num_batches). (default: 1.0)
  --limit_test_batches LIMIT_TEST_BATCHES
                        How much of test dataset to check (float = fraction, int = num_batches). (default: 1.0)
  --limit_predict_batches LIMIT_PREDICT_BATCHES
                        How much of prediction dataset to check (float = fraction, int = num_batches). (default: 1.0)
  --val_check_interval VAL_CHECK_INTERVAL
                        How often to check the validation set. Use float to check within a training epoch, use int to check every n steps (batches). (default: 1.0)
  --flush_logs_every_n_steps FLUSH_LOGS_EVERY_N_STEPS
                        How often to flush logs to disk (defaults to every 100 steps). .. deprecated:: v1.5 ``flush_logs_every_n_steps`` has been deprecated in v1.5 and will be
                        removed in v1.7. Please configure flushing directly in the logger instead. (default: None)
  --log_every_n_steps LOG_EVERY_N_STEPS
                        How often to log within steps (defaults to every 50 steps). (default: 50)
  --accelerator ACCELERATOR
                        Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "auto") as well as custom accelerator instances. .. deprecated:: v1.5 Passing
                        training strategies (e.g., 'ddp') to ``accelerator`` has been deprecated in v1.5.0 and will be removed in v1.7.0. Please use the ``strategy`` argument
                        instead. (default: None)
  --strategy STRATEGY   Supports different training strategies with aliases as well custom training type plugins. (default: None)
  --sync_batchnorm [SYNC_BATCHNORM]
                        Synchronize batch norm layers between process groups/whole world. (default: False)
  --precision PRECISION
                        Double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16). Can be used on CPU, GPU or TPUs. (default: 32)
  --enable_model_summary [ENABLE_MODEL_SUMMARY]
                        Whether to enable model summarization by default. (default: True)
  --weights_summary WEIGHTS_SUMMARY
                        Prints a summary of the weights when training begins. .. deprecated:: v1.5 ``weights_summary`` has been deprecated in v1.5 and will be removed in v1.7.
                        To disable the summary, pass ``enable_model_summary = False`` to the Trainer. To customize the summary, pass
                        :class:`~pytorch_lightning.callbacks.model_summary.ModelSummary` directly to the Trainer's ``callbacks`` argument. (default: top)
  --weights_save_path WEIGHTS_SAVE_PATH
                        Where to save weights if specified. Will override default_root_dir for checkpoints only. Use this if for whatever reason you need the checkpoints stored
                        in a different place than the logs written in `default_root_dir`. Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/' Defaults to
                        `default_root_dir`. (default: None)
  --num_sanity_val_steps NUM_SANITY_VAL_STEPS
                        Sanity check runs n validation batches before starting the training routine. Set it to `-1` to run all batches in all validation dataloaders. (default:
                        2)
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        Path/URL of the checkpoint from which training is resumed. If there is no checkpoint file at the path, an exception is raised. If resuming from mid-
                        epoch checkpoint, training will start from the beginning of the next epoch. .. deprecated:: v1.5 ``resume_from_checkpoint`` is deprecated in v1.5 and
                        will be removed in v1.7. Please pass the path to ``Trainer.fit(..., ckpt_path=...)`` instead. (default: None)
  --profiler PROFILER   To profile individual steps during training and assist in identifying bottlenecks. (default: None)
  --benchmark [BENCHMARK]
                        If true enables cudnn.benchmark. (default: False)
  --deterministic [DETERMINISTIC]
                        If ``True``, sets whether PyTorch operations must use deterministic algorithms. Default: ``False``. (default: False)
  --reload_dataloaders_every_n_epochs RELOAD_DATALOADERS_EVERY_N_EPOCHS
                        Set to a non-negative integer to reload dataloaders every n epochs. (default: 0)
  --reload_dataloaders_every_epoch [RELOAD_DATALOADERS_EVERY_EPOCH]
                        Set to True to reload dataloaders every epoch. .. deprecated:: v1.4 ``reload_dataloaders_every_epoch`` has been deprecated in v1.4 and will be removed
                        in v1.6. Please use ``reload_dataloaders_every_n_epochs``. (default: False)
  --auto_lr_find [AUTO_LR_FIND]
                        If set to True, will make trainer.tune() run a learning rate finder, trying to optimize initial learning for faster convergence. trainer.tune() method
                        will set the suggested learning rate in self.lr or self.learning_rate in the LightningModule. To use a different key set a string instead of True with
                        the key name. (default: False)
  --replace_sampler_ddp [REPLACE_SAMPLER_DDP]
                        Explicitly enables or disables sampler replacement. If not specified this will toggled automatically when DDP is used. By default it will add
                        ``shuffle=True`` for train sampler and ``shuffle=False`` for val/test sampler. If you want to customize it, you can set ``replace_sampler_ddp=False``
                        and add your own distributed sampler. (default: True)
  --detect_anomaly [DETECT_ANOMALY]
                        Enable anomaly detection for the autograd engine. (default: False)
  --auto_scale_batch_size [AUTO_SCALE_BATCH_SIZE]
                        If set to True, will `initially` run a batch size finder trying to find the largest batch size that fits into memory. The result will be stored in
                        self.batch_size in the LightningModule. Additionally, can be set to either `power` that estimates the batch size through a power search or `binsearch`
                        that estimates the batch size through a binary search. (default: False)
  --prepare_data_per_node [PREPARE_DATA_PER_NODE]
                        If True, each LOCAL_RANK=0 will call prepare data. Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare data .. deprecated:: v1.5 Deprecated in v1.5.0
                        and will be removed in v1.7.0 Please set ``prepare_data_per_node`` in LightningDataModule or LightningModule directly instead. (default: None)
  --plugins PLUGINS     Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins. (default: None)
  --amp_backend AMP_BACKEND
                        The mixed precision backend to use ("native" or "apex"). (default: native)
  --amp_level AMP_LEVEL
                        The optimization level to use (O1, O2, etc...). By default it will be set to "O2" if ``amp_backend`` is set to "apex". (default: None)
  --move_metrics_to_cpu [MOVE_METRICS_TO_CPU]
                        Whether to force internal logged metrics to be moved to cpu. This can save some gpu memory, but can make training slower. Use with attention. (default:
                        False)
  --multiple_trainloader_mode MULTIPLE_TRAINLOADER_MODE
                        How to loop over the datasets when there are multiple train loaders. In 'max_size_cycle' mode, the trainer ends one epoch when the largest dataset is
                        traversed, and smaller datasets reload when running out of their data. In 'min_size' mode, all the datasets reload when reaching the minimum length of
                        datasets. (default: max_size_cycle)
  --stochastic_weight_avg [STOCHASTIC_WEIGHT_AVG]
                        Whether to use `Stochastic Weight Averaging (SWA) <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/>`_. .. deprecated::
                        v1.5 ``stochastic_weight_avg`` has been deprecated in v1.5 and will be removed in v1.7. Please pass
                        :class:`~pytorch_lightning.callbacks.stochastic_weight_avg.StochasticWeightAveraging` directly to the Trainer's ``callbacks`` argument instead.
                        (default: False)
  --terminate_on_nan [TERMINATE_ON_NAN]
                        If set to True, will terminate training (by raising a `ValueError`) at the end of each training batch, if any of the parameters or the loss are NaN or
                        +/-inf. .. deprecated:: v1.5 Trainer argument ``terminate_on_nan`` was deprecated in v1.5 and will be removed in 1.7. Please use ``detect_anomaly``
                        instead. (default: None)
```

## Example

Train 10 folds

```bash
$ export common_args=" --n_fold 10 --gpus 1 --accumulate_grad_batches 4 --max_epochs 150 --split_ratios 0.95 0.05"
$ python main.py $common_args --batch_size 8 --szs 384 128 64 32 --mhas 3 --swa --lr 0.0003  --dropout 0.4
```


