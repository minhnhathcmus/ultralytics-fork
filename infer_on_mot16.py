from ultralytics import YOLO

model = YOLO('yolov8n.pt')

for id, imgsz in [['02', (1920, 1080)],
           ['04', (1920, 1080)],
           ['05', (640, 480)],
           ['09', (1920, 1080)],
           ['10', (1920, 1080)],
           ['11', (1920, 1080)],
           ['13', (1920, 1080)]]:
# for id, imgsz in [['01', (1920, 1080)],
#            ['03', (1920, 1080)],
#            ['06', (640, 480)],
#            ['07', (1920, 1080)],
#            ['08', (1920, 1080)],
#            ['12', (1920, 1080)],
#            ['14', (1920, 1080)]]:
    
    source = f"../mot16/train/MOT16-{id}/img1"
    result_path = f"../TrackEval/data/trackers/mot_challenge/MOT16-train/yolov8_botsort/data/MOT16-{id}.txt"
    # result_path = f"../mot16_testset_results/MOT16-{id}.txt"

    results = model.track(
        source=source,
        stream=True,
        imgsz=imgsz,
        conf=0.25,
        iou=0.75,
        half=False,
        device=0,
        show=False,
        save=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        show_labels=False,
        show_conf=False,
        max_det=300,
        vid_stride=False,
        line_width=None,
        visualize=False,
        augment=False,
        agnostic_nms=False,
        retina_masks=False,
        classes=0,
        tracker='botsort.yaml',
        verbose=False
        )

    with open (result_path, 'w') as f:
        for frame_id, result in enumerate(results):
            for d in result.boxes:
                if d.id is not None:
                    line = (frame_id + 1, int(d.id.item()), *d.xyxy[0, :2].view(-1), *d.xywh[0, -2:].view(-1), float(d.conf), -1, -1, -1)
                    f.write((('%g,' * len(line)).rstrip(',') % line) + '\n')