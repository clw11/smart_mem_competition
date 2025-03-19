# 代码说明
当前代码，包含三个主要模块：generate feature、train model、inference；

generate feature模块：使用原始数据以及维修单生成特征文件

    --data_path: 原始数据集文件路径，不需要指定type_A
    --ticket_path: 维修单路径
    --feature_path： 保存的特征文件路径


train model模块：使用特征文件生成正负样本，然后训练并保存模型

    --feature_path： 特征文件路径
    --ticket_path： 维修单路径
    --model_path： 模型保存路径


inference模块：使用特征文件生成测试集，并加载模型进行推理

    --feature_path： 特征文件路径
    --model_path： 模型保存路径
    --output_path：result输出路径
    --test_stage：评估阶段，1表示推理0601-0801期间的样本，2表示推理0801-1001期间的样本


note：当前模型只对type_A类型的文件进行模型训练和推理


# 参考运行脚本
## step1 generate feature
python .\main.py --process generate_features --data_path "D:/competition_data_release_feather" --ticket_path "D:/competition_data_release_feather/ticket.csv" --feature_path "D:/release_features/combined_sn_feature"
## step2 train model
python .\main.py --process train_model --feature_path "D:/release_features/combined_sn_feature" --ticket_path "D:/competition_data_release_feather/ticket.csv" --model_path "model.pkl"
## step3 inference
python .\main.py --process inference --feature_path "D:/release_features/combined_sn_feature" --model_path "model.pkl" --output_path "submission.csv" --test_stage 2
