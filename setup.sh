git clone https://github.com/viriyadhika/CSC2515-project.git
cd CSC2515-project
mkdir data
# scp the data over
unzip data/audioset_5000.zip -d data
unzip data/esc-50.zip -d data
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install torchcodec

scp data/final_data/audioset_5000.zip vastai:/workspace/CSC2515-project/data
scp data/final_data/esc-50.zip vastai:/workspace/CSC2515-project/data

