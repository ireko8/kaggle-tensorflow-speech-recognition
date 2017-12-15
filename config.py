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
SHIFT_FORWARD = int(SAMPLE_RATE*0.1)
SHIFT_BACKWARD = -int(SAMPLE_RATE*0.1)
SPEED_UP = 1.2
SPEED_DOWN = 0.8
ADD_WHITENOISE_RATE = 0.005
MIX_BGN_RATE = 0.005

AUG_LIST = ["id",
            "vol_up",
            "vol_down",
            "shift_forward",
            "shift_backward",
            "speed_up",
            "speed_down",
            "add_wn",
            "mix_bgn",
            "lp_2000",
            "lp_4000",
            "lp_6000"]
