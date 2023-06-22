from io import StringIO
from pathlib import Path
import streamlit as st
import time
from ultralytics import YOLO
import os
import sys
import argparse
from PIL import Image

def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result

def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)

if __name__ == '__main__':

    model = YOLO('yolov8n.pt')

    st.title('Demo YOLOv8 & BoT-SORT')

    # Video input
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4',
                                                                   'asf',
                                                                   'avi',
                                                                   'gif',
                                                                   'm4v',
                                                                   'mkv',
                                                                   'mov',
                                                                   'mpeg',
                                                                   'mpg',
                                                                   'ts',
                                                                   'wmv',
                                                                   'webm'])
    if uploaded_file is not None:
        is_valid = True
        with st.spinner(text='Uploading...'):
            st.sidebar.video(uploaded_file)
            # Read from web and save video on server
            with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            # Set path to parser
            source = f'data/videos/{uploaded_file.name}'
    else:
        is_valid = False

    if is_valid:
        if st.button('Tracking Objects'):
            with st.spinner(text='Tracking Objects...'):
                model.track(
                    source=source,
                    stream=False,
                    conf=0.25,
                    iou=0.75,
                    device=1,
                    half=False,
                    show=False,
                    save=True,
                    save_txt=False,
                    save_conf=False,
                    save_crop=False,
                    show_labels=True,
                    show_conf=True,
                    max_det=300,
                    vid_stride=1,
                    line_width=None,
                    visualize=False,
                    augment=False,
                    agnostic_nms=False,
                    retina_masks=False,
                    classes=0,
                    tracker='botsort.yaml',
                    verbose=False
                    )

            st.success('Successful!')

            with st.spinner(text='Preparing Results...'):
                result_path = get_detection_folder()
                for vid in os.listdir(result_path):
                    result = open(str(Path(f'{result_path}') / vid), 'rb')
                    st.video(result.read())
                    result.close()