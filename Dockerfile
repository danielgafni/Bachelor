FROM continuumio/miniconda3

WORKDIR /bachelor

RUN conda install -y -c conda-forge -c pytorch -c ravelbio -c plotly -c defaults \
   pip numpy pandas scikit-learn matplotlib tqdm pytorch torchvision plotly gym jupyterlab ffmpeg opencv \
   pyperclip awscli cosmos-wfm
RUN pip install tensorboardX

RUN echo ". ~/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate" >> ~/.bashrc

RUN apt install -y libgl1-mesa-glx awscli

COPY bindsnet bindsnet/
COPY thesis thesis/
COPY optimize_awsbatch/evaluate.py .