import os
import shutil
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import mediapipe as mp
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


class Face():
    def __init__(self, image_path, bbox, global_id):
        self.image_path = image_path
        self.bbox = bbox
        self.features = None
        self.cluster_id = None
        self.global_id = global_id
    
    def set_features(self, features):
        self.features = features
    
    def set_cluster(self, cluster_id):
        self.cluster_id = cluster_id

    def save_face(self, output_dir):
        img = cv.imread(self.image_path)
        x, y, w, h = self.bbox
        if self.cluster_id is not None:
            output_path = os.path.join(output_dir, str(self.cluster_id)).replace('\\', '/')
            if not os.path.isdir(output_path):
                os.mkdir(output_path)
            cv.imwrite(os.path.join(output_path, str(self.global_id) + '.png').replace('\\', '/'), img[y:y+h, x:x+w])


class MTCNNDetector():
    def __init__(self, device):
        self.face_detector = MTCNN(post_process=True, select_largest=False, keep_all=True, device=device)
    
    def detect(self, img):
        bboxes, prob = self.face_detector.detect(img)
        return bboxes, prob

class MPFaceDetector():
    def __init__(self):
        self.FACE_DETECTION_CONFIDENCE = 0.7
        self.face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=self.FACE_DETECTION_CONFIDENCE)
    
    def detect(self, img):
        bboxes = []
        prob = []
        h_i, w_i = img.shape[:2]
        face_detection_results = self.face_detector.process(img)
        if face_detection_results.detections is not None:
            for detection in face_detection_results.detections:
                x = int(detection.location_data.relative_bounding_box.xmin * w_i)
                y = int(detection.location_data.relative_bounding_box.ymin * h_i)
                w = int(detection.location_data.relative_bounding_box.width * w_i)
                h = int(detection.location_data.relative_bounding_box.height * h_i)
                score = detection.score[0]
                bboxes.append([x, y, w, h])
                prob.append(score)

        return bboxes, prob

def align_face(image):
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        # Detect landmarks
        results = face_mesh.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        
        for face_landmarks in results.multi_face_landmarks:
            # Get key landmarks: left eye, right eye, and nose tip
            left_eye = np.array([face_landmarks.landmark[33].x, face_landmarks.landmark[33].y])  # Adjust index if needed
            right_eye = np.array([face_landmarks.landmark[263].x, face_landmarks.landmark[263].y])
            nose_tip = np.array([face_landmarks.landmark[1].x, face_landmarks.landmark[1].y])
            
            # Calculate the center point between eyes
            eye_center = (left_eye + right_eye) / 2.0
            
            # Compute the angle of rotation
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))
            
            # Compute the rotation matrix
            h, w = image.shape[:2]
            rot_mat = cv.getRotationMatrix2D(tuple(eye_center * [w, h]), angle, 1.0)
            
            # Apply the rotation
            aligned_face = cv.warpAffine(image, rot_mat, (w, h), flags=cv.INTER_CUBIC)
            return aligned_face
    return None



if __name__ == "__main__":
    
    # init params
    k = 3  # number of clusters.
    clustering_results = {}  # key: img path, value: list of class Face
    images_paths = []
    face_counter = 0
    input_dir = "./data"
    output_dir = "./output"
    # remove existing output folder at start of run
    if os.path.isdir(output_dir):
        try:
            shutil.rmtree(output_dir)
        except Exception as e:
            print(f"Warning: failed to remove existing output dir: {e}")
    os.mkdir(output_dir)

    # init face detection model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # face_detector = MTCNNDetector(device)
    face_detector = MPFaceDetector()

    # Init emmbedings pretrained model on CASIA-Webface
    resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
    pca = PCA(n_components=0.95)
    
    # init kmeans
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
    # dbscan = DBSCAN(eps=0.1, min_samples=5, metric='euclidean')
    
    # load input images
    list_of_images = os.listdir(input_dir)
    for image_name in list_of_images:
        images_paths.append(os.path.join(input_dir, image_name).replace('\\', '/'))
    for img_pth in images_paths:
        img = cv.imread(img_pth)
        h, w = img.shape[:2]

        # detect faces
        bboxes, prob = face_detector.detect(img)

        # init Face class
        n = len(bboxes)
        for i in range(n):
            x, y, w, h = bboxes[i]
            x, y, w, h = [int(x), int(y), int(w), int(h)]
            face_counter+=1
            face = Face(img_pth, [x, y, w, h], face_counter)

            # pre-process each face image - 160x160
            face_crop = img[y:y+h, x:x+w]
            resized = cv.resize(face_crop, (160, 160))
            aligned = align_face(resized)
            if aligned is not None:
                face_img = aligned
            else:
                face_img = resized

            # convert BGR->RGB, normalize and move channels to C,H,W
            face_rgb = cv.cvtColor(face_img, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
            face_rgb = np.transpose(face_rgb, (2, 0, 1))

            # prewhiten per-image
            def prewhiten(x: np.ndarray):
                mean = np.mean(x)
                std = np.std(x)
                std_adj = max(std, 1.0/np.sqrt(x.size))
                return (x - mean) / std_adj

            face_pre = prewhiten(face_rgb)
            img_tensor = torch.from_numpy(face_pre).unsqueeze(0).to(device)

            # extract and save face features
            embeding = resnet(img_tensor).to("cpu").detach().numpy()
            normalized_features = normalize(embeding, norm='l2')
            face.set_features(normalized_features)
    
            # save Face class to clustering_results
            if img_pth not in clustering_results.keys():
                clustering_results[img_pth] = [face]
            else:
                clustering_results[img_pth].append(face)
    
    # prepare for clustering
    features = []
    for img_pth in clustering_results.keys():
        for face in clustering_results[img_pth]:
            features.append(face.features)
    features = np.array(features).reshape(-1, 512)

    # reduce dimensionality
    reduced_features = pca.fit_transform(features)

    # Determine outliers by 2D PCA visualization y-axis threshold
    # compute a 2D projection for deciding outliers (and for plotting)
    pca2 = PCA(n_components=2)
    points2d = pca2.fit_transform(reduced_features)

    # Mark as 'not_face' any point with PCA y-coordinate > 0.8 (user threshold)
    outlier_idx = np.where(points2d[:, 1] > 0.8)[0]

    # Initialize final labels as -1
    final_labels = np.full(reduced_features.shape[0], -1, dtype=int)

    # Inliers are points below the threshold
    inlier_idx = np.array([i for i in range(reduced_features.shape[0]) if i not in set(outlier_idx)])

    # Decide number of person clusters automatically using silhouette on KMeans
    if inlier_idx.size >= 2:
        features_inliers = reduced_features[inlier_idx]
        max_k = min(5, features_inliers.shape[0])
        best_k = 1
        best_score = -1.0
        best_labels = None
        for k_try in range(2, max_k + 1):
            km = KMeans(n_clusters=k_try, random_state=0, n_init="auto")
            labels_try = km.fit_predict(features_inliers)
            try:
                score = silhouette_score(features_inliers, labels_try)
            except Exception:
                score = -1.0
            if score > best_score:
                best_score = score
                best_k = k_try
                best_labels = labels_try

        if best_labels is None:
            # fallback: single cluster
            mapped = np.zeros(features_inliers.shape[0], dtype=int)
            num_persons = 1
        else:
            # map labels to sequential 0..num_persons-1
            unique_person_labels = np.unique(best_labels)
            label_map = {old: new for new, old in enumerate(unique_person_labels)}
            mapped = np.array([label_map[l] for l in best_labels])
            num_persons = len(unique_person_labels)

        # merge tiny clusters into nearest larger cluster
        counts = np.bincount(mapped)
        min_size = max(2, int(0.03 * features_inliers.shape[0]))
        if counts.size > 1:
            # compute centroids
            centroids = []
            for cl in range(counts.size):
                centroids.append(features_inliers[mapped == cl].mean(axis=0))
            centroids = np.vstack(centroids)
            for small_cl in np.where(counts < min_size)[0]:
                if counts[small_cl] == 0:
                    continue
                # find nearest non-small cluster
                other_cl = [c for c in range(centroids.shape[0]) if c != small_cl and counts[c] >= min_size]
                if not other_cl:
                    continue
                dists = np.linalg.norm(centroids[other_cl] - centroids[small_cl], axis=1)
                nearest = other_cl[int(np.argmin(dists))]
                mapped[mapped == small_cl] = nearest
                counts[nearest] += counts[small_cl]
                counts[small_cl] = 0

        final_labels[inlier_idx] = mapped
    else:
        # fallback: treat all as single person cluster
        final_labels[:] = 0
        num_persons = 1

    # Assign not-face label as the last index
    not_face_label = num_persons
    final_labels[outlier_idx] = not_face_label

    # Build a color map for clusters (person clusters + not-face)
    num_total_clusters = num_persons + 1
    cmap = plt.cm.get_cmap('tab20', num_total_clusters)
    color_map = {}
    for cl in range(num_persons):
        rgb = cmap(cl)[:3]
        # convert RGB float to BGR 0-255 for OpenCV
        bgr = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
        color_map[cl] = bgr
    # not-face -> red
    color_map[not_face_label] = (0, 0, 255)

    # save results: annotate original images with bboxes and class ids
    for img_pth in clustering_results.keys():
        img = cv.imread(img_pth)
        if img is None:
            continue
        for face in clustering_results[img_pth]:
            cid = int(final_labels[face.global_id - 1])
            face.set_cluster(cid)
            x, y, w, h = face.bbox
            # choose color per cluster (from color_map), fallback to red
            color = color_map.get(cid, (0, 0, 255))
            # draw bbox and larger label text
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            label = f'class:{cid}'
            cv.putText(img, label, (x, max(0, y - 12)), cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        out_path = os.path.join(output_dir, os.path.basename(img_pth)).replace('\\', '/')
        cv.imwrite(out_path, img)

    # visualization: 2D projection of reduced features colored by final label
    try:
        plt.figure(figsize=(8, 6))
        unique_labels = sorted(list(set(final_labels.tolist())))
        colors = plt.cm.get_cmap('tab10', len(unique_labels))
        for idx, lab in enumerate(unique_labels):
            mask = final_labels == lab
            plt.scatter(points2d[mask, 0], points2d[mask, 1], c=[colors(idx)], label=f'cluster_{lab}', s=40)
        # annotate points with global_id
        for i_pt in range(points2d.shape[0]):
            plt.text(points2d[i_pt, 0], points2d[i_pt, 1], str(i_pt+1), fontsize=8)
        plt.axhline(0.8, color='red', linestyle='--', linewidth=1, label='y=0.8 threshold')
        plt.legend()
        plt.title('Face feature clustering (2D PCA)')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'cluster_scatter.png').replace('\\', '/')
        plt.savefig(plot_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Visualization skipped: {e}")




