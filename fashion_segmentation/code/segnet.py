from keras_segmentation.models.segnet import segnet
import matplotlib.pyplot as plt
import os

here = os.path.abspath(os.path.dirname(__file__))
path_to_repo = os.path.join(here.split("fashion_segmentation")[0], "fashion_segmentation")

model = segnet(n_classes=46, input_height=1024, input_width=1024)

model.train(
    train_images=os.path.join(path_to_repo, "data/train"),
    train_annotations=os.path.join(path_to_repo, "data/train_annotations"),
    checkpoints_path="/tmp/segnet_1", epochs=5,
    val_images=os.path.join(path_to_repo, "data/val_train"),
    val_annotations=os.path.join(path_to_repo, "data/val_annotations"),
)

out = model.predict_segmentation(
    inp=os.path.join(path_to_repo, "data/test", "00000663ed1ff0c4e0132b9b9ac53f6e.jpg"),
    out_fname=os.path.join(path_to_repo, "data", "out_segnet.png")
)

plt.imshow(out)
plt.show()

# evaluating the model
print(model.evaluate_segmentation(inp_images_dir=os.path.join(path_to_repo, "data/test"),
                                  annotations_dir=os.path.join(path_to_repo, "data/test_annotations")))
