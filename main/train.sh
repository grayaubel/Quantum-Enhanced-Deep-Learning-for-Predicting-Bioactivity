#!/bin/bash

# dataset=CAD

dataset=CAD

mlp_hidden_dim=64
mlp_output_dim=32
attention_dim=32
attention_output_dim=100

batch_train=32
batch_test=32
lr=1e-3
lr_decay=0.9
dropout_rate=0.5
decay_interval=10
weight_decay=1e-6
iteration=100

setting=$dataset--mlp_hidden_dim$mlp_hidden_dim--mlp_output_dim$mlp_output_dim--attention_dim$attention_dim--attention_output_dim$attention_output_dim--batch_train$batch_train--batch_test$batch_test--lr$lr--lr_decay$lr_decay--dropout_rate$dropout_rate--decay_interval$decay_interval--weight_decay$weight_decay--iteration$iteration
python train.py $dataset $mlp_hidden_dim $mlp_output_dim $attention_input_dim $attention_dim $attention_output_dim $batch_train $batch_test $lr $lr_decay $dropout_rate $decay_interval $weight_decay $iteration $setting