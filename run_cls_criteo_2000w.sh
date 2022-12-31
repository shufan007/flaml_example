
python3 ./train.py \
      --train_input ./criteo_1w.csv \
      --output_dir ./outputs \
      --data_resample_size 20000000 \
      --label_cols "['label']" \
      --feature_cols "['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','d1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14','d15','d16','d17','d18','d19','d20','d21','d22','d23','d24','d25','d26']" \
      --time_budget 600 \
      --metric roc_auc \
      --estimator_list '["lgbm"]' \
      --seed 1 \
      --estimator_kwargs '{"n_jobs":4,"n_concurrent_trials":2,"sample":1,"min_sample_size":100000,"retrain_full":0,"log_type":"all"}'
