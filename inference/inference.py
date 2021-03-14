import tensorflow.compat.v1 as tf
import numpy as np
#from models.fastpose import FastPose
from utils.pose_input_preprocess import transform_detections
from utils.convert_results_format import prepare_detection_results
from utils.pose_postproccessing import post_process
from utils.visualization_utils import visualize_poses
from builders.builder import build_detection_model
import tensorflow as tf
import cv2


def main():
    #detection_model = build_detection_model(name, config)
    #detection_model.load_model(model_path, label_map)
    model = tf.saved_model.load("inference/saved_model")
    infr = model.signatures['serving_default']

    inp = cv2.imread('inference/sample.jpg')
    inp_rgb = im_rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    #preprocessed_image = detector.preprocess(inp)
    #detections_raw = detector.inference(preprocessed_image)
    #detections = prepare_detection_results(detections_raw, w, h)
    
#    detections = np.array([[  0.0000,   0.0000,   7.9978,  71.6218, 123.5006,   0.9864,   0.9994,
#           0.0000]])
    
    detections = np.array([[0.0000e+00, 4.8360e+02, 8.0461e+02, 1.0931e+03, 2.0388e+03, 9.9820e-01,
         9.9999e-01, 0.0000e+00],
        [0.0000e+00, 1.0499e+03, 7.5240e+02, 1.6743e+03, 2.4084e+03, 9.9078e-01,
         9.9999e-01, 0.0000e+00],
        [0.0000e+00, 1.5949e+03, 9.9634e+02, 2.3587e+03, 2.2222e+03, 9.8668e-01,
         9.9995e-01, 0.0000e+00],
        [0.0000e+00, 3.2557e+03, 9.0536e+02, 3.5294e+03, 1.6164e+03, 9.8638e-01,
         9.9998e-01, 0.0000e+00],
        [0.0000e+00, 2.6197e+03, 7.6679e+02, 3.1344e+03, 2.1086e+03, 9.6865e-01,
         9.9996e-01, 0.0000e+00],
        [0.0000e+00, 2.2635e+03, 9.3226e+02, 2.3718e+03, 1.3805e+03, 9.3844e-01,
         9.9998e-01, 0.0000e+00],
        [0.0000e+00, 4.4076e+02, 9.0047e+02, 6.3583e+02, 1.5971e+03, 9.2874e-01,
         9.9998e-01, 0.0000e+00],
        [0.0000e+00, 2.8621e+03, 7.0690e+02, 3.2694e+03, 1.8603e+03, 9.0747e-01,
         9.9999e-01, 0.0000e+00],
        [0.0000e+00, 8.6789e+02, 8.1764e+02, 1.1657e+03, 1.7274e+03, 8.5713e-01,
         9.9999e-01, 0.0000e+00],
        [0.0000e+00, 2.3431e+03, 9.5201e+02, 2.4418e+03, 1.3627e+03, 6.8212e-01,
         9.9990e-01, 0.0000e+00],
        [0.0000e+00, 2.3730e+03, 1.0087e+03, 2.4474e+03, 1.3088e+03, 4.0227e-01,
         9.9926e-01, 0.0000e+00],
        [0.0000e+00, 2.2329e+03, 9.8045e+02, 2.2894e+03, 1.3390e+03, 3.7365e-01,
         9.9814e-01, 0.0000e+00],
        [0.0000e+00, 1.7698e+03, 9.6615e+02, 1.8724e+03, 1.1957e+03, 3.0504e-01,
         9.9141e-01, 0.0000e+00],
        [0.0000e+00, 1.7155e+03, 9.3078e+02, 1.8016e+03, 1.2231e+03, 3.0181e-01,
         9.9887e-01, 0.0000e+00],
        [0.0000e+00, 3.4508e+03, 9.8130e+02, 3.5637e+03, 1.1761e+03, 2.8387e-01,
         4.7732e-01, 0.0000e+00],
        [0.0000e+00, 2.5856e+03, 9.3704e+02, 2.6338e+03, 1.0048e+03, 2.6423e-01,
         9.9398e-01, 0.0000e+00],
        [0.0000e+00, 1.7922e+03, 9.8538e+02, 1.8853e+03, 1.1632e+03, 2.3618e-01,
         9.7348e-01, 0.0000e+00],
        [0.0000e+00, 2.4469e+03, 9.4039e+02, 2.4986e+03, 1.0229e+03, 2.3030e-01,
         9.9648e-01, 0.0000e+00],
        [0.0000e+00, 2.4280e+03, 9.3910e+02, 2.4968e+03, 1.1061e+03, 1.6140e-01,
         9.9904e-01, 0.0000e+00],
        [0.0000e+00, 3.2621e+03, 9.3852e+02, 3.3417e+03, 1.1096e+03, 1.5570e-01,
         9.9791e-01, 0.0000e+00],
        [0.0000e+00, 3.2928e+03, 9.4538e+02, 3.3444e+03, 1.0281e+03, 1.3924e-01,
         9.0198e-01, 0.0000e+00],
        [0.0000e+00, 3.5420e+03, 9.8530e+02, 3.5634e+03, 1.0593e+03, 1.0013e-01,
         7.0085e-01, 0.0000e+00]])

    
    inps, cropped_boxes, boxes, scores, ids = transform_detections(inp_rgb, detections, (256, 192))
    raw_output = infr(inps)
    hm = raw_output['conv_out'].numpy()
    poses = post_process(hm, cropped_boxes, boxes, scores, ids)
    output_image = visualize_poses(inp_rgb, poses) 
    cv2.imwrite("test_out.jpg", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    # TODO: image decoding
    print("Done!")


if __name__ == '__main__':
    main()
