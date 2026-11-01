# 模型训练
使用官方 PaddleOCR release/2.7 源码，放在 training/PaddleOCR。兼容 PaddlePaddle 2.6.2。
将用户含嵌入图片的标注压缩包放在 training/lable.zip，在项目根目录依次运行：
python training/prepare_dataset.py
python training/recover_weights.py
python training/finetune.py

识别网络采用 CRNN MobileNetV3-small 0.5、BiLSTM96、CTC43。
保持48×320输入，Adam学习率0.00001，batch32，1epoch，按验证集选择权重。训练需要额外CPU时间与训练依赖。
注意：若从本包已更新模型运行，是继续训练；精确重现起点需原仓库的原始识别模型。
按整张图片哈希划分160/20/20，避免同页字符跨集合。不能证明原始模型未见过这些图，也不能证明不同集合来自不同书写者。
指标是标注裁剪文字准确率，不是整页成功率。
