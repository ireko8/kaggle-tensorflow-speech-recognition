TRAIN_AUDIO_PATH = "input/train/audio"
TRAIN_PATH = "input/train"
TRAIN_FILE_META_INFO = "data/train_file_info.csv"
POSSIBLE_LABELS = ['yes',
                   'no',
                   'up',
                   'down',
                   'left',
                   'right',
                   'on',
                   'off',
                   'stop',
                   'go',
                   'silence',
                   'unknown']
TEST_AUDIO_PATH = "input/test/audio"
SILENCE_DATA_PATH = "data/silence"
SILENCE_DATA_VERSION = "2018_01_03_04_30_29"

BATCH_SIZE = 32
EPOCHS = 100

SAMPLE_RATE = 16000

VOLUME_UP = 1.2
VOLUME_DOWN = 0.8
SHIFT_MAX = int(SAMPLE_RATE*0.1)
SHIFT_MIN = -int(SAMPLE_RATE*0.1)
SPEED_UP_MAX = 1.1
SPEED_UP_MIN = 0.9
SPEED_DOWN_MAX = 0.9
SPEED_DOWN_MIN = 0.8
PITCH_MAX = 1
PITCH_MIN = -1
ADD_WN_MIN = 0.005
ADD_WN_MAX = 0.01
MIX_BGN_RATE = 0.05
MIX_BGN_MAX = 0.2
LP_MIN = 2000
LP_MAX = 7999

AUG_LIST = []
# AUG_LIST = ["pitch_up",
            # "speed_up",
            # "pitch_down",
            # "speed_down"]
            
AUG_VERSION = "2018_01_12_05_19_46_param_fixed"
AUG_PATH = "data/augment/"

NOISE_TYPE = ["blue", "brown", "violet", "white", "pink"]

SAMPLING_SIZE = 2000
CV_SAMPLING_SIZE = 1800
