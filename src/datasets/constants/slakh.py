SR = 16000 
NFFT = 2048
HOP = 256
NMEL = 80
SEQ_LEN = int((4 * SR) / HOP)

# Slakh instrument classes (based on actual dataset)
INSTRUMENTS = [
    "Pipe",
    "Bass", 
    "Organ",
    "Synth Pad",
    "Synth Lead",
    "Strings (continued)",
    "Strings",
    "Guitar",
    "Reed",
    "Piano",
    "Brass",
    "Ethnic"
]

# Create mapping from instrument names to indices
DICT_INST_TO_IDX = {
    instrument: idx for idx, instrument in enumerate(INSTRUMENTS)
}

DICT_IDX_TO_INST = {v: k for k, v in DICT_INST_TO_IDX.items()}

# Ban list for instruments we don't want to include
BAN_LIST = [
    "Chromatic Percussion",
    "Drums",
    "Percussive",
    "Sound Effects",
    "Sound effects",
] 