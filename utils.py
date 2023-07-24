import os
import shutil

import torch
import yaml

# 保存模型的 检查点 检查是否为最好的模型
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # torch.save 保存state 存储地址为filename
    torch.save(state, filename)
    if is_best:
        # shutil.copyfile将filename中文件复制到'model_best.pth.tar'文件夹中
        shutil.copyfile(filename, 'model_best.pth.tar')

# 保存配置文件
def save_config_file(model_checkpoints_folder, args):
    # os.path.exists（） 括号内文件存在则标注1
    # 总语句：如果model_checkpoints_folder为空/不存在，则执行：
    if not os.path.exists(model_checkpoints_folder):
        # os.makedirs（）创建括号内文件夹
        os.makedirs(model_checkpoints_folder)
        # with用于对资源进行访问的场合，确保不管使用过程中是否发生异常都会执行必要的“清理”操作，释放资源
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            # yaml.dump 将yaml文件一次性全部写入你创建的文件
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    # 计算指定值 k 的 k 个顶级预测的准确性
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # with torch.no_grad() 设置requires_grad参数为False 不进行自动求导
    with torch.no_grad():
        maxk = max(topk)
        # size统计输入矩阵的元素的个数
        batch_size = target.size(0)

        # topk 输入maxk的tensor数据 得到前1个数据 指定在从大到小按照顺序返回
        # _得到的是tensor中具体数值 pred得到第几个数据为真的可能性最高
        _, pred = output.topk(maxk, 1, True, True)
        # .t 对矩阵进行转置
        pred = pred.t()
        # .eq？？？不知道
        # view 将tensor以1行 -1列的形式展示出来
        # expand_as 将矩阵扩展为pred的形式
        correct = pred.eq(target.view(1, -1).expand_as(pred)) ######### -1什么意思

        res = []
        for k in topk:
            # [:k]截取correct中元素到下标k
            # reshape(-1) 将元素改成一行 没有行列
            # float() 转为浮点数
            # sum 求和？
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # mul_ 将correct_k中数据和（100.0 / batch_size）中数据点对点相乘
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
