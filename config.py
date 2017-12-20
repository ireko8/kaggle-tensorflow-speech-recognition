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
SILECE_DATA_PATH = "data/silence"

BATCH_SIZE = 128

SAMPLE_RATE = 16000

VOLUME_UP = 1.2
VOLUME_DOWN = 0.8
SHIFT_FORWARD = int(SAMPLE_RATE*0.2) + 1
SHIFT_BACKWARD = -int(SAMPLE_RATE*0.2)
SPEED_UP = 1.2
SPEED_DOWN = 0.8
PITCH_UP = 3
PITCH_DOWN = -3
ADD_WHITENOISE_RATE = 0.005
ADD_WHITENOISE_MIN = 0.005
ADD_WHITENOISE_MAX = 0.02
MIX_BGN_RATE = 0.1
MIX_BGN_MAX = 0.5
LP_MIN = 2000
LP_MAX = 7999

AUG_LIST = ["id",
            "shift_forward",
            "shift_random",
            "shift_backward",
            "speed_up",
            "speed_down",
            "speed_random",
            "pitch_up",
            "pitch_down",
            "pitch_random",
            "add_wn",
            "mix_random"]

