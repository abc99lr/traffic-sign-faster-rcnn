rm keras_frcnn/*.pyc
rm results_imgs/*
#python train_frcnn.py --network=simple --parser=simple --path=../dataset/PNG_train/gt.txt --output_weight_path='./model_frcnn_zfnet.hdf5'
#python test_frcnn.py --network=simple --path=../dataset/PNG_test --config_filename=simple_config.pickle
python train_frcnn.py --output_weight_path='./model_frcnn_zfnet.hdf5'
#python test_frcnn.py  --config_filename=simple_config.pickle