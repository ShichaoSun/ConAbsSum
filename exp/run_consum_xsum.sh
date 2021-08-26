
python src/consum.py \
    --do_train \
    --do_predict \
    --max_source_length 512 --max_target_length 64 \
    --learning_rate 1e-6 \
    --freeze_embeds --freeze_encoder --adafactor --task summarization_xsum \
    --model_name_or_path google/pegasus-xsum \
    --data_dir xsum/ \
    --decoder_layerdrop 0.1 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --weight_decay 1e-8 \
    --gpus 1 \
    --num_train_epochs 1 \
    --train_batch_size 8 \
    --eval_batch_size 4 \
    --warmup_steps 500 \
    --n_val 1000 \
    --output_dir=xsumConSum \
    --logger_name wandb \
    --val_check_interval 0.01 \
    --lr_scheduler cosine \
    --log_every_n_steps 100 \
    --num_sanity_val_steps 0 \
    --early_stopping_patience 4 \
    --loss_length_penalty 0.6 \
    --margin_value 0.0
