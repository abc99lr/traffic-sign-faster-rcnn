rm keras_frcnn/*.pyc
rm results_imgs/*
python train_frcnn.py --network=fcnet --parser=simple --num_epochs=12 --epoch_length=600 --path=../faster-rcnn/dataset/PNG_train/gt.txt --config_filename=fcnet_config.pickle --output_weight_path='./model_fcnet.hdf5'
#python test_frcnn.py --network=fcnet --path=../faster-rcnn/dataset/PNG_train/gt.txt --config_filename=fcnet_config.pickle
#python train_frcnn.py --network=zfnet --parser=simple --num_epochs=12 --epoch_length=600 --path=../faster-rcnn/dataset/PNG_train/gt.txt --config_filename=zfnet_config.pickle --output_weight_path='./model_zfnet.hdf5'
#python test_frcnn.py --network=zfnet --path=../faster-rcnn/dataset/PNG_train/gt.txt --config_filename=zfnet_config.pickle
#python train_frcnn.py --network=vgg --parser=simple --num_epochs=12 --epoch_length=600 --path=../faster-rcnn/dataset/PNG_train/gt.txt --config_filename=vgg_config.pickle --output_weight_path='./model_vgg.hdf5'
#python test_frcnn.py --network=vgg --path=../faster-rcnn/dataset/PNG_train/gt.txt --config_filename=vgg_config.pickle