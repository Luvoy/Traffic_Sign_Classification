import torch
import copy
import os
import matplotlib.pyplot as plt
import time

from data_set_loader import data_loaders
import network

from hyper_parameters import NUM_CLASSES
from hyper_parameters import LEARNING_RATE
from hyper_parameters import MOMENTUM
from hyper_parameters import NUM_EPOCHS
from hyper_parameters import PHASE
from hyper_parameters import MODEL_SAVE_DIR
from hyper_parameters import STEP_SIZE
from hyper_parameters import GAMMA
from hyper_parameters import TRAIN_LOG_SAVE_PATH
from hyper_parameters import LOG_EPOCH_MOD
from hyper_parameters import LOG_TIME_FORMAT


def get_time_str(format_str=LOG_TIME_FORMAT):
    return time.strftime(format_str, time.localtime(time.time()))


def train_model(model,
                device,
                criterion,
                optimizer,
                scheduler,
                num_epochs,
                model_save_dir,
                save_log=None):
    if save_log:
        f = open(save_log, "a")

    loss_list = {'train': [], 'test': []}
    accu_list = {'train': [], 'test': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # epoch循环训练
    for epoch in range(num_epochs):
        print('epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-*' * 10)
        f.write(get_time_str())
        f.write('epoch {}/{}\n'.format(epoch, num_epochs - 1))
        f.write('-*' * 10 + '\n')

        # 每个epoch都有train(训练)和test(测试)两个阶段
        for phase in PHASE:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects_classes = 0

            for idx, data in enumerate(data_loaders[phase]):

                if idx + 1 % LOG_EPOCH_MOD == LOG_EPOCH_MOD:
                    print(f"{phase} processing: {idx} th batch.")
                    f.write(get_time_str())
                    f.write(f"{phase} processing: {idx} th batch.\n")

                inputs = data['image'].to(device)
                labels_classes = data['class_id'].to(device)

                optimizer.zero_grad()

                # 训练阶段
                with torch.set_grad_enabled(phase == 'train'):
                    out_classes = model(inputs)
                    out_classes = out_classes.view(-1, NUM_CLASSES)
                    _, preds_classes = torch.max(out_classes, 1)  # TODO
                    # 计算训练误差
                    loss = criterion(out_classes, labels_classes)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # TODO
                running_loss += loss.item() * inputs.size(0)

                corrects_classes += torch.sum(preds_classes == labels_classes)

            # TODO
            epoch_loss = running_loss / len(data_loaders[phase].dataset)

            loss_list[phase].append(epoch_loss)

            epoch_acc = corrects_classes.double() / len(
                data_loaders[phase].dataset)

            accu_list[phase].append(epoch_acc)

            print(
                f'{phase} Loss: {epoch_loss:.4f}  Acc_classes: {epoch_acc:.2%}'
            )
            f.write(get_time_str())
            f.write(
                '{phase} Loss: {epoch_loss:.4f}  Acc_classes: {epoch_acc:.2%}\n'
            )

            # 测试阶段
            if phase == 'test' and epoch_acc > best_acc:
                # 如果当前epoch下的准确率总体提高或者误差下降，则认为当下的模型最优

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'Best test classes Acc: {best_acc}')
                f.write(get_time_str())
                f.write(f'Best test classes Acc: {best_acc}\n')

    # 保存模型
    best_model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(),
               os.path.join(model_save_dir,
                            get_time_str() + 'best_model.pt'))

    print(f'Best_model:  classes Accu: {best_acc}')
    f.write(get_time_str())
    f.write(f'Best_model:  classes Accu: {best_acc}\n')
    f.close()

    # 保存这次训练的超参数们
    with open("hyper_parameters.py", "r", encoding='utf-8') as f1:
        hypers = f1.readlines()
    f1.close()
    with open(os.path.join(model_save_dir,
                           get_time_str() + 'parameters.txt'),
              "w",
              encoding='utf-8') as f2:
        f2.writelines(hypers)
    f2.close()
    return model, loss_list, accu_list


def visualize(loss_list, accu_list, epoch, fig_save_dir):
    x = range(0, NUM_EPOCHS)
    y1 = loss_list["test"]
    y2 = loss_list["train"]
    plt.figure(figsize=(19, 14))

    plt.subplot(211)
    plt.plot(x,
             y1,
             color="r",
             linestyle="-",
             marker="o",
             linewidth=1,
             label="test")
    plt.plot(x,
             y2,
             color="b",
             linestyle="-",
             marker="o",
             linewidth=1,
             label="train")
    plt.legend()
    plt.title('train and test loss vs. epoches')
    plt.xlabel('epoches')
    plt.ylabel('loss')

    plt.subplot(212)
    y3 = accu_list["train"]
    y4 = accu_list["test"]
    plt.plot(x,
             y3,
             color="y",
             linestyle="-",
             marker=".",
             linewidth=1,
             label="train_class")
    plt.plot(x,
             y4,
             color="g",
             linestyle="-",
             marker=".",
             linewidth=1,
             label="val_class")
    plt.legend()
    plt.title('train and test vs. epoches')
    plt.xlabel('epoches')
    plt.ylabel('accuracy')
    plt.savefig(os.path.join(fig_save_dir, get_time_str() + 'loss-accu.png'))
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_tensor_type(torch.FloatTensor)

net = network.resnet18(num_classes=NUM_CLASSES, pretrained=False)
# net = network.SimpleNet(num_classes=NUM_CLASSES)

total_num = sum(p.numel() for p in net.parameters())
trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f'Total: {total_num}, Trainable: {trainable_num}')

net = net.to(device)

optimizer = torch.optim.SGD(net.parameters(),
                            lr=LEARNING_RATE,
                            momentum=MOMENTUM)

criterion = torch.nn.CrossEntropyLoss()

exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=STEP_SIZE,
                                                   gamma=GAMMA)

model, loss_list, accu_list = train_model(model=net,
                                          device=device,
                                          criterion=criterion,
                                          optimizer=optimizer,
                                          scheduler=exp_lr_scheduler,
                                          num_epochs=NUM_EPOCHS,
                                          model_save_dir=MODEL_SAVE_DIR,
                                          save_log=TRAIN_LOG_SAVE_PATH)

visualize(loss_list, accu_list, epoch=NUM_EPOCHS, fig_save_dir=MODEL_SAVE_DIR)
