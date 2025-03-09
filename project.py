import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_detector(model_path):
    return cv2.FaceDetectorYN.create(
        model_path, "", (320, 320), 0.8, 0.3, 5000
    )

def load_recognizer(model_path):
    return cv2.FaceRecognizerSF.create(model_path, "")

def detect_faces(detector, image):
    detector.setInputSize((image.shape[1], image.shape[0]))
    faces = detector.detect(image)
    return faces[1] if faces[1] is not None else None

def draw_results(image, faces):
    for face in faces:
        coords = face[:-1].astype(np.int32)
        cv2.rectangle(image, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), 2)
        for i, color in zip(range(4, 14, 2), [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255)]):
            cv2.circle(image, (coords[i], coords[i + 1]), 2, color, -1)

def recognize_faces(recognizer, img1, img2, faces1, faces2):
    face1_align = recognizer.alignCrop(img1, faces1[0])
    face2_align = recognizer.alignCrop(img2, faces2[0])
    feature1 = recognizer.feature(face1_align)
    feature2 = recognizer.feature(face2_align)
    return recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_NORM_L2)

def display_results(img1, img2, similarity, threshold=1.128):
    ## threshold = 1.128  زیر این عدد 2 عکس برار هستند
    result_text = "Same Identity" if similarity <= threshold else "Different Identities"
    color = "green" if similarity <= threshold else "red"
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1].axis("off")
    
    plt.suptitle(f"{result_text} (Distance: {similarity:.3f})", color=color, fontsize=14)
    plt.show()

def main(image_path1, image_path2):
    detector = load_detector("model/face_detection_yunet_2023mar.onnx")
    recognizer = load_recognizer("model/face_recognition_sface_2021dec.onnx")
    
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    faces1 = detect_faces(detector, img1)
    faces2 = detect_faces(detector, img2)
    
    if faces1 is None or faces2 is None:
        print("No faces detected in one or both images.")
        return
    
    draw_results(img1, faces1)
    draw_results(img2, faces2)
    
    similarity = recognize_faces(recognizer, img1, img2, faces1, faces2)
    display_results(img1, img2, similarity)




# Example Usage
main("image/image1.jpg", "image/image2.jpg")
