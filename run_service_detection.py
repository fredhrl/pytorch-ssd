import sys

import cv2

import pandas as pd

from vision.ssd.mobilenet_v2_ssd_lite import (
    create_mobilenetv2_ssd_lite,
    create_mobilenetv2_ssd_lite_predictor,
)
from vision.ssd.mobilenetv1_ssd import (
    create_mobilenetv1_ssd,
    create_mobilenetv1_ssd_predictor,
)
from vision.ssd.mobilenetv1_ssd_lite import (
    create_mobilenetv1_ssd_lite,
    create_mobilenetv1_ssd_lite_predictor,
)
from vision.ssd.squeezenet_ssd_lite import (
    create_squeezenet_ssd_lite,
    create_squeezenet_ssd_lite_predictor,
)
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.utils.misc import Timer

if len(sys.argv) < 4:
    print(
        "Usage: python run_ssd_example.py <net type>  <model path> <label path> <video file>"
    )
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]

cap = cv2.VideoCapture(sys.argv[4])  # capture from file
fps = cap.get(cv2.CAP_PROP_FPS)  # get the video fps

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)


if net_type == "vgg16-ssd":
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == "mb1-ssd":
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == "mb1-ssd-lite":
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == "mb2-ssd-lite":
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == "sq-ssd-lite":
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print(
        "The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite."
    )
    sys.exit(1)
net.load(model_path)

if net_type == "vgg16-ssd":
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == "mb1-ssd":
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == "mb1-ssd-lite":
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == "mb2-ssd-lite":
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif net_type == "sq-ssd-lite":
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    print(
        "The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite."
    )
    sys.exit(1)

main_wnd_name = "annotated"
cv2.namedWindow(main_wnd_name, cv2.WINDOW_NORMAL)

signal = []

c1, c2 = 30, 120

timer = Timer()
while True:
    ret, orig_image = cap.read()
    if not ret:
        df = pd.DataFrame(signal, columns=["Signal"])
        print(df.keys())
        df.to_csv(
            "./" + sys.argv[4].split("/")[-1].split(".")[0] + ".csv",
            index=None,
            sep=" ",
            mode="a",
        )
        break

    m, n, c = orig_image.shape
    x, xf, y, yf = c1, m - c1, c2, n - c2
    orig_image = orig_image[x:xf, y:yf]
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.8)

    if len(boxes) is 0:
        signal.append(0)
        continue

    interval = timer.end()
    # print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        if labels[i].numpy() == 15:
            signal.append(1)
            box = boxes[i, :]
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.rectangle(
                orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4
            )

            cv2.putText(
                orig_image,
                label,
                (box[0] + 20, box[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # font scale
                (255, 0, 255),
                2,
            )  # line type
    cv2.imshow(main_wnd_name, orig_image)

    if len(box) == 0:
        signal.append(0)

    if cv2.waitKey(int(1 / fps * 1000)) & 0xFF == ord("q"):
        df = pd.DataFrame(signal, columns=["Signal"])
        df.to_csv(
            "./" + sys.argv[4].split("/")[-1].split(".")[0] + ".csv",
            index=None,
            sep=" ",
            mode="a",
        )
        break
cap.release()
cv2.destroyAllWindows()
