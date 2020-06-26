import torch
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import os

from hyper_parameters import NUM_CLASSES
from hyper_parameters import MODEL_SAVE_DIR
from hyper_parameters import WRONG_IMG_DIR

from data_set_loader import data_loaders
import network


def predict(model, data_loaders, device, wrong_img_dir):
    model.to(device)
    model.eval()
    with torch.no_grad():
        wrong_count = 0
        right_count = 0
        for i, data in enumerate(data_loaders['test']):
            inputs = data['image'].to(device)
            labels_classes = data['class_id'].to(device)

            out_classes = model(inputs)
            out_classes = out_classes.view(-1, NUM_CLASSES)
            _, preds_classes = torch.max(out_classes, 1)

            if preds_classes != labels_classes:  # 只看没测对的
                wrong_count += 1
                plt.imshow(transforms.ToPILImage()(inputs.to(
                    torch.device("cpu")).squeeze(0)))
                plt.title(
                    f'predicted classes: {preds_classes}\n ground-truth classes:{labels_classes}'
                )
                if wrong_img_dir is None or wrong_img_dir is False:
                    plt.show()
                else:
                    plt.savefig(
                        os.path.join(
                            wrong_img_dir,
                            f'{wrong_count}--predicted-{preds_classes.item()}--ground-truth-{labels_classes.item()}.png'
                        ))

            else:
                right_count += 1
    return right_count, wrong_count


net = network.resnet18(num_classes=NUM_CLASSES, pretrained=False)
load_path = os.path.join(MODEL_SAVE_DIR,
                         "2020-04-17-21-02-26----best_model.pt")
net.load_state_dict(torch.load(load_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# 重新准备一个关于test的shuffle的数据集
# shuffle看的时候更随机
# test_loader = torch.utils.data.DataLoader(dataset=data_set_loader.test_dataset,
#                                           shuffle=True)
# data_loaders = {'test': test_loader}
# 不过现在不需要了, 因为只看错的

right_count, wrong_count = predict(model=net,
                                   data_loaders=data_loaders,
                                   device=device,
                                   wrong_img_dir=WRONG_IMG_DIR)
print(right_count, wrong_count)
