# EVALUATOR - PROGRESS MANAGER
START_CONFIG = "START_CONFIG"
END_CONFIG = "END_CONFIG"
END_FINAL_RUN = "END_FINAL_RUN"
OUTER_FOLD = "outer_fold"
INNER_FOLD = "inner_fold"
CONFIG_ID = "config_id"
RUN_ID = "run_id"
ELAPSED = "elapsed"
EXP_NAME = "exp_name"
SEED = "seed"
GRID_SEARCH = "grid"
RANDOM_SEARCH = "random"
NUM_SAMPLES = "num_samples"
SAMPLE_METHOD = "sample_method"
DATASET = "dataset"
CONFIG = "config"
MODEL_ASSESSMENT = "MODEL_ASSESSMENT"
FOLDS = "folds"
TR_LOSS = "training_loss"
VL_LOSS = "validation_loss"
TR_SCORE = "training_score"
VL_SCORE = "validation_score"
BEST_CONFIG = "best_config"
OUTER_TRAIN = "outer_train"
OUTER_VALIDATION = "outer_validation"
OUTER_TEST = "outer_test"

# SPLITTER
DATA_SPLITTER_BASE_PATH = "mlwiz.data.splitter."

# ENGINE
DEFAULT_ENGINE_CALLBACK = (
    "mlwiz.training.callback.engine_callback.EngineCallback"
)

# STATE
TRAINING = "training"
EVALUATION = "evaluation"
VALIDATION = "validation"
TEST = "test"
MAIN_LOSS = "main_loss"
MAIN_SCORE = "main_score"
LOSSES = "losses"
LOSS = "loss"
SCORES = "scores"
SCORE = "score"
MODEL_STATE = "model_state"
OPTIMIZER_STATE = "optimizer_state"
SCHEDULER_STATE = "scheduler_state"
BEST_EPOCH = "best_epoch"
BEST_EPOCH_RESULTS = "best_epoch_results"
EPOCH = "epoch"
STOP_TRAINING = "stop_training"
LAST_CHECKPOINT_FILENAME = "last_checkpoint.pth"
BEST_CHECKPOINT_FILENAME = "best_checkpoint.pth"
BATCH_LOSS_EXTRA = "batch_loss_extra"

# STRING FORMATTING
TENSORBOARD = "tensorboard"
EMB_TUPLE_SUBSTR = "_embeddings_tuple"
ATOMIC_SAVE_EXTENSION = ".part"
CLASS_NAME = "class_name"
ARGS = "args"
MODEL_STATE_DICT = "model_state_dict"
OPTIMIZER_STATE_DICT = "optimizer_state_dict"
AVG = "avg"
STD = "std"
READOUT = "readout"

# SET UP
CUDA = "cuda"
MLWIZ_RAY_NUM_GPUS_PER_TASK = "MLWIZ_RAY_NUM_GPUS_PER_TASK"
MIN = "min"
MAX = "max"

# CLI
CONFIG_FILE_CLI_ARGUMENT = "--config-file"
CONFIG_FILE = "config_file"
DEBUG_CLI_ARGUMENT = "--debug"

DATA_ROOT = "data_root"
DATA_SPLITS_FILE = "data_splits_file"
DATASET_CLASS = "dataset_class"
DATASET_GETTER = "dataset_getter"
evaluate_every = "evaluate_every"
EVALUATE_EVERY = "evaluate_every"
DATA_LOADER = "data_loader"
DATA_LOADER_ARGS = "data_loader_args"
DATASET_NAME = "dataset_name"
STORAGE_FOLDER = "storage_folder"
DEBUG = "debug"
DEVICE = "device"
EXPERIMENT = "experiment"
RISK_ASSESSMENT_TRAINING_RUNS = "risk_assessment_training_runs"
GPUS_PER_TASK = "gpus_per_task"
HIGHER_RESULTS_ARE_BETTER = "higher_results_are_better"
MAX_CPUS = "max_cpus"
MAX_GPUS = "max_gpus"
GPUS_SUBSET = "gpus_subset"
MODEL = "model"
RESULT_FOLDER = "result_folder"

# TELEGRAM BOT
TELEGRAM_CONFIG_FILE = "telegram_config_file"
TELEGRAM_BOT_TOKEN = "bot_token"
TELEGRAM_BOT_CHAT_ID = "bot_chat_ID"
LOG_MODEL_SELECTION = "log_model_selection"
LOG_FINAL_RUNS = "log_final_runs"