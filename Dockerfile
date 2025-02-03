# Use TensorFlow GPU base image
FROM tensorflow/tensorflow:2.16.1-gpu-jupyter

# Set working directory
WORKDIR /tf

# Install additional Python packages if needed
RUN pip install --no-cache-dir \
    pandas \
    matplotlib \
    scikit-learn \
    seaborn \
    tensorflow-addons
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python

# Expose Jupyter port
EXPOSE 8888

# Set default command to run Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]