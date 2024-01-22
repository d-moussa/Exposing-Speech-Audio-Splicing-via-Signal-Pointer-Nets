random_seed = 42
config_name = "TTS all 0-5 Data Config"
set_name = "tts"
experiment_name = "all"

data_path = "data/prepared/"  # Path to processed speech audio

min_split_points = 0  # Minimal number of splices per sample
max_split_points = 5  # Maximal number of splices per sample
dataset_size_train = 0  # Number of samples in train set
dataset_size_val = 0  # Number of samples in validation set
dataset_size_test = 30000  # Number of samples in test set
package_size = 30000  # Splits the datasets in smaller packages of this size

duplicate_strategy = "allow"  # Duplicate handling "disallow" = no duplicates allowed (slowest), "minimize" = try to minimize duplicate count, "allow" = do nothing (fastest)
max_duplicate_retries = 5  # if disallow - finishes datageneration after N tries, if minimize - tries N times to generate a new sample generation will continue
dataset_root_path = "datasets/tts/all/"  # Path to the actual datasets and to the dataset dictionaries
dataset_pickle_name_train = ""  # Filenames
dataset_pickle_name_val = ""
dataset_pickle_name_test = "test_all_multi"
dataset_path_train = ""
dataset_path_val = ""
dataset_path_test = "datasets/tts/all/dataset_test_all_multi.json"

silence_path_train = ""
silence_path_val = ""
silence_path_test = "datasets/tts/all/silence_test_all_multi.json"
use_same_file = False  # Use same file for datageneration (useful for intersplicing)
#
rir_file_all = [  # Used rirs needed for generation of processed samples
    "none",
    "raw_others_pra/pos_2/pos_2_room_11_1.16_pra",
    "raw_others_pra/pos_2/pos_2_room_7_0.74_pra",
    "raw_others_pra/pos_2/pos_2_room_4_0.53_pra",
    "raw_others_pra/pos_1/pos_1_room_12_1.41_pra",
    "raw_others_pra/pos_1/pos_1_room_1_0.33_pra",
    "paper_pra/pos_2/pos_2_room_1_0.31_pra",
    "paper_pra/pos_2/pos_2_room_5_0.52_pra",
    "paper_pra/pos_1/pos_1_room_2_0.35_pra",
    "paper_ACE/pos_2/Lecture_Room_2_Single_403a_1_7_1.25_ace",
    "paper_ACE/pos_2/Building_Lobby_Single_EE_lobby_1_6_0.65_ace",
    "paper_ACE/pos_2/Office_2_Single_803_1_3_0.39_ace",
    "paper_ACE/pos_1/Building_Lobby_Single_EE_lobby_1_6_0.65_ace",
    "paper_ACE/pos_1/Meeting_Room_2_Single_611_1_2_0.37_ace",
    "paper_ACE/pos_1/Office_1_Single_502_1_1_0.34_ace",
]
rir_file_train = []
rir_file_val = []
rir_file_test = ["none",  # Used rirs for the test set (needed for dataset generation)
                 "raw_others_pra/pos_2/pos_2_room_11_1.16_pra",
                 "raw_others_pra/pos_2/pos_2_room_7_0.74_pra",
                 "raw_others_pra/pos_2/pos_2_room_4_0.53_pra",
                 "raw_others_pra/pos_1/pos_1_room_12_1.41_pra",
                 "raw_others_pra/pos_1/pos_1_room_1_0.33_pra",
                 "paper_pra/pos_2/pos_2_room_1_0.31_pra",
                 "paper_pra/pos_2/pos_2_room_5_0.52_pra",
                 "paper_pra/pos_1/pos_1_room_2_0.35_pra",
                 "paper_ACE/pos_2/Lecture_Room_2_Single_403a_1_7_1.25_ace",
                 "paper_ACE/pos_2/Building_Lobby_Single_EE_lobby_1_6_0.65_ace",
                 "paper_ACE/pos_2/Office_2_Single_803_1_3_0.39_ace",
                 "paper_ACE/pos_1/Building_Lobby_Single_EE_lobby_1_6_0.65_ace",
                 "paper_ACE/pos_1/Meeting_Room_2_Single_611_1_2_0.37_ace",
                 "paper_ACE/pos_1/Office_1_Single_502_1_1_0.34_ace",
                 ]

speaker_file_all = {  # Used speakerfiles needed for generation of processed samples
    "Female1Ex": ["s0f0", "s0f4", "s1f15", "s1f19", "s2f23", "s3f35", "s4f47", "s5f50", "s5f59", "s6f67"],
    "Female2Ex": ["s0f2", "s0f6", "s1f10", "s1f13", "s1f16", "s2f23", "s2f26", "s3f30", "s3f35", "s3f38"],
    "Male1Ex": ["s0f3", "s0f8", "s1f14", "s2f22", "s2f23", "s3f30", "s3f34", "s4f40", "s5f55", "s6f69"],
    "Male2Ex": ["s0f0", "s0f2", "s2f20", "s3f35", "s5f57", "s6f64", "s7f76", "s8f82", "s9f98", "s9f99"],
    "Male4Ex": ["s0f0", "s0f2", "s0f7", "s0f9", "s1f11", "s1f17", "s2f23", "s2f28", "s3f35", "s3f39"],
    "Female3Ex": ["s0f0", "s0f2", "s0f5", "s0f6", "s0f7", "s1f10", "s1f11", "s1f13", "s1f16", "s1f18"],
    "Female4Ex": ["s0f0", "s1f13", "s2f24", "s2f29", "s3f35", "s4f41", "s4f45", "s5f54", "s6f60", "s6f67"],
    "Female5Ex": ["s0f0", "s0f5", "s0f9", "s2f24", "s3f30", "s3f34", "s3f37", "s4f44", "s5f51", "s5f58"],
    "Female6Ex": ["s0f1", "s0f3", "s0f9", "s1f13", "s1f16", "s2f27", "s3f35", "s5f54", "s6f63", "s6f68"],
    "Male3Ex": ["s0f2", "s0f7", "s1f10", "s1f17", "s2f20", "s2f25", "s3f30", "s3f34", "s3f38", "s4f41"],
}
speaker_file_train = {  # Used speakerfiles for the train set (needed for dataset generation)
}
speaker_file_val = {  # Used speakerfiles for the validation set (needed for dataset generation)
}
speaker_file_test = {  # Used speakerfiles for the test set (needed for dataset generation)
    "Female1Ex": ["s0f0", "s0f4", "s1f15", "s1f19", "s2f23", "s3f35", "s4f47", "s5f50", "s5f59", "s6f67"],
    "Female2Ex": ["s0f2", "s0f6", "s1f10", "s1f13", "s1f16", "s2f23", "s2f26", "s3f30", "s3f35", "s3f38"],
    "Male1Ex": ["s0f3", "s0f8", "s1f14", "s2f22", "s2f23", "s3f30", "s3f34", "s4f40", "s5f55", "s6f69"],
    "Male2Ex": ["s0f0", "s0f2", "s2f20", "s3f35", "s5f57", "s6f64", "s7f76", "s8f82", "s9f98", "s9f99"],
    "Male4Ex": ["s0f0", "s0f2", "s0f7", "s0f9", "s1f11", "s1f17", "s2f23", "s2f28", "s3f35", "s3f39"],
    "Female3Ex": ["s0f0", "s0f2", "s0f5", "s0f6", "s0f7", "s1f10", "s1f11", "s1f13", "s1f16", "s1f18"],
    "Female4Ex": ["s0f0", "s1f13", "s2f24", "s2f29", "s3f35", "s4f41", "s4f45", "s5f54", "s6f60", "s6f67"],
    "Female5Ex": ["s0f0", "s0f5", "s0f9", "s2f24", "s3f30", "s3f34", "s3f37", "s4f44", "s5f51", "s5f58"],
    "Female6Ex": ["s0f1", "s0f3", "s0f9", "s1f13", "s1f16", "s2f27", "s3f35", "s5f54", "s6f63", "s6f68"],
    "Male3Ex": ["s0f2", "s0f7", "s1f10", "s1f17", "s2f20", "s2f25", "s3f30", "s3f34", "s3f38", "s4f41"],
}

noise_file = "whitenoise"  # Noise file whitenoise - generates whitenoise or path (e.g. "data/custom_noise/airport.wav") - use custom noise
snr_db_train = [None]  # SNR range for training None or list of integers
snr_db_val = [None]  # SNR range for validation None or list of integers
snr_db_test = list(range(-10, 50))  # SNR range for testing None or list of integers

max_compressions = 1  # Number of iterative compressions
compressions_train = [None]  # Type of compressions for training None or list of dicts
compressions_val = [None]  # Type of compressions for validation None or list of dicts
compressions_test = [  # Type of compressions for testing None or list of dicts
    {"format": "mp3", "compression": list(range(10, 129))},
    {"format": "amr-nb", "compression": [0, 1, 2, 3, 4, 5, 6, 7]}
]

sampling_rate = 16000  # Sampling rate of audio files

energy_window = 300  # Window size for silence detection
ignore_start_sec = 0.0  # Ignore first seconds of the audio file
ignore_end_sec = 0.0  # Ignore last seconds of the audio file
min_time = 5.0  # Minimal spliced audio length
max_time = 45.0  # Maximal spliced audio length

n_fft = 16000  # Size of FFT (fast fourier transform)
window_length = None  # Window size (None or int)
hop_length = None  # Length of hop between STFT windows (None or int)

emb_size = 256  # Melspectrogram bins
input_specs_dims = [256, 20, 1]  # Height of multiinput representations or None for single input
input_specs_types = ["melspec", "mfcc", "centroid"]  # Types of multiinput representations or None for single input
