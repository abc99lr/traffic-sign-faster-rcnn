rm keras_frcnn/*.pyc
rm results_imgs/*
#python train_frcnn.py --network=simple --parser=simple --path=../faster-rcnn/dataset/PNG_train/gt.txt --config_filename=simple_config.pickle --output_weight_path='./model_frcnn_zfnet.hdf5'
#python test_frcnn.py --network=simple --path=../faster-rcnn/dataset/PNG_test --config_filename=simple_config.pickle
#python train_frcnn.py --path=../faster-rcnn/dataset/PNG_train/gt.txt --config_filename=simple_config.pickle --output_weight_path='./model_frcnn_zfnet.hdf5'
#python test_frcnn.py --path=../faster-rcnn/dataset/PNG_test --config_filename=simple_config.pickle

python train_frcnn.py --network=simple --parser=simple --path=../faster-rcnn/dataset/PNG_train/gt.txt --config_filename=config.pickle --output_weight_path='./model_frcnn_simple.hdf5'
python test_frcnn.py --network=simple --path=../faster-rcnn/dataset/PNG_test --config_filename=config.pickle