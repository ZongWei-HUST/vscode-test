from PIL import Image
from cv2 import rotate
from timm.data.dataset import ImageDataset
from torch import no_grad
from torch.nn.functional import softmax
from numpy import argmax
from sklearn.metrics import confusion_matrix, recall_score
from numpy import array
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

def cm_plot(true_label, predict_label, cls_num, cls_name, pic=None):
    cm = confusion_matrix(true_label, predict_label) # 由预测标签和真实标签生成混淆矩阵
    print("precision: ", precision_score(true_label, predict_label, average="macro"))
    print("recall: ", recall_score(true_label, predict_label, average="macro"))
    print("f1-score: ", f1_score(true_label, predict_label, average="macro"))
    cm = cm.T
    # print(cm)
    plt.figure()
    plt.matshow(cm, cmap=plt.cm.Blues) # 绘制混淆矩阵, 用个人觉得好看的蓝色风格
    plt.colorbar() # 颜色标签
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.xlabel('Predicted label') # 坐标轴标签
    plt.ylabel('True label') 
    num_local = array(range(cls_num))
    plt.xticks(num_local, cls_name) # 将标签印在x,y轴坐标上
    plt.yticks(num_local, cls_name) 
    plt.title('Confusion Matrix')
    if pic is not None:
        plt.savefig(str(pic) + '.jpg', dpi=300, bbox_inches="tight") # 保证显示的图片清晰度和位置
    plt.show()

def model_evaluate(root, model, config, transform, show_confusion_matrix=None):
    correct_num = 0
    
    dataset_eval = ImageDataset(root=root) 
    samples = dataset_eval.parser.samples # list as (file_path, target)
    idx2cls = {v:k for k, v in dataset_eval.parser.class_to_idx.items()} # dict as {idx : cls}
    
    model.eval() # important before using model to evaluate
    with no_grad():
        y_true = []
        y_pred = []
        for (filename, target) in samples:
            img = Image.open(filename).convert("RGB")
            tensor = transform(img).unsqueeze(0) # add batch dimension
            tensor = tensor.to("cuda") # cuda compute, need put model in cuda device
            out = model(tensor)
            
            pred = softmax(out[0], dim=0)
            pred_idx = argmax(pred.cpu()) # convert to cpu
            
            # confusion matrix
            if show_confusion_matrix:
                y_true.append(target)
                y_pred.append(int(pred_idx))
            
            if pred_idx == target:
                correct_num += 1
            else:
                print(filename.split('/')[-1], end='\t')
                print('y_true: ' + idx2cls[target] + '\t', 'y_pred: ' + idx2cls[int(pred_idx)])
        print("evaluate images num: " + str(len(samples)), "correct num: " + str(correct_num))
        print("accuracy:{:.2%}".format(correct_num / len(samples)))
    
        if show_confusion_matrix:
            cm_plot(y_true, y_pred, len(idx2cls), list(idx2cls.values()), root.split('/')[-2])
    