random_seed = 42
config_name = "Ace all 0-5 Multimodal Data Config"
set_name = "ace"
experiment_name = "allMultimodal"


data_path = "data/prepared/"
#
min_split_points = 0
max_split_points = 5
dataset_size_train = 500000
dataset_size_val = 300000
dataset_size_test = 300000
package_size = 25000

#
duplicate_strategy = "allow" #"disallow", "minimize", "allow"
max_duplicate_retries = 5
dataset_root_path = "datasets/ace/all/"
dataset_pickle_name_train = "train_all_multi"
dataset_pickle_name_val = "val_all_multi"
dataset_pickle_name_test = "test_all_multi"
dataset_path_train = "datasets/ace/all/dataset_train_all_multi_s.json"
dataset_path_val = "datasets/ace/all/dataset_val_all_multi_s.json"
dataset_path_test = "datasets/ace/all/dataset_test_all_multi_s.json"

silence_path_train = "datasets/ace/all/silence_train_all_multi_s.json"
silence_path_val = "datasets/ace/all/silence_val_all_multi_s.json"
silence_path_test = "datasets/ace/all/silence_test_all_multi_s.json"
use_same_file = False
#
rir_file_all = [
    "none",
    #"rir",
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
rir_file_train = ["none",
    #"rir",
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
rir_file_val = ["none",
    #"rir",
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
rir_file_test = ["none",
    #"rir",
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

speaker_file_all = {
    "F1":["s1", "s2", "s3", "s4", "s5"],
    "F2":["s1", "s2", "s3", "s4", "s5"],
    "F3":["s1", "s2", "s3", "s4", "s5"],
    "F4":["s1", "s2", "s3", "s4", "s5"],
    "F5":["s1", "s2", "s3", "s4", "s5"],
    "M1":["s1", "s2", "s3", "s4", "s5"],
    "M2":["s1", "s2", "s3", "s4", "s5"],
    "M3":["s1", "s2", "s3", "s4", "s5"],
    "M4":["s1", "s2", "s3", "s4", "s5"],
    "M5":["s1", "s2", "s3", "s4", "s5"],
    "M6":["s3", "s4"],
    "M7":["s3", "s4"],
    "M8":["s3", "s4"],
    "M9":["s3", "s4"]
}
speaker_file_train = {
    "F1":["s1", "s2", "s3", "s4", "s5"],
    "F2":["s1", "s2", "s3", "s4", "s5"],
    "F3":["s1", "s2", "s3", "s4", "s5"],
    "F4":["s1", "s2", "s3", "s4", "s5"],
    "M2":["s1", "s2", "s3", "s4", "s5"],
    "M3":["s1", "s2", "s3", "s4", "s5"],
    "M4":["s1", "s2", "s3", "s4", "s5"],
    "M5":["s1", "s2", "s3", "s4", "s5"],
    "M6":["s3", "s4"],
    "M7":["s3", "s4"],
}
speaker_file_val = {
    "M8":["s3", "s4"],
    "M9":["s3", "s4"]
}
speaker_file_test = {
    "F5":["s1", "s2", "s3", "s4", "s5"],
    "M1":["s1", "s2", "s3", "s4", "s5"],
}

noise_file = "whitenoise"
snr_db_train = list(range(-10, 50))
snr_db_val = list(range(-10, 50))
snr_db_test = [-5, 10, 30]

max_compressions = 1
compressions_train = [
    {"format":"mp3", "compression":list(range(10,129))}, 
    {"format":"amr-nb", "compression":[0,1,2,3,4,5,6,7]}
    ]
compressions_val = [
    {"format":"mp3", "compression":list(range(10,129))},
    {"format":"amr-nb", "compression":[0,1,2,3,4,5,6,7]}
    ]
compressions_test = [
    {"format":"mp3", "compression":[20, 70, 120]},
    {"format":"amr-nb", "compression":[2,4,6]}
    ]
#
sampling_rate = 16000
energy_window = 300
ignore_start_sec = 0.0
ignore_end_sec = 0.0
min_time = 5.0
max_time = 45.0

n_fft = 16000
window_length = None
hop_length = None

emb_size = 256
input_specs_dims = [256, 20, 1]
input_specs_types = ["melspec", "mfcc", "centroid"]
