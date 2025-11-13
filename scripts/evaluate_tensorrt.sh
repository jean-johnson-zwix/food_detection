docker run --gpus all -it --rm -v /home/ubuntu:/home/ubuntu -w /home/ubuntu nvcr.io/nvidia/tensorrt:24.10-py3
pip install ultralytics numpy pyyaml
pip install --no-cache-dir opencv-python-headless==4.10.0.84
python -m src.evaluate_engine