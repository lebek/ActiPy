import numpy as np
import cv2

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib import cm

from actipy.video_features import VideoFeatures
from actipy.plan import good_cells, vid_dims


class DissertationPredictor:

    def __init__(self, training_feature_path, training_category_path,
                 probabalistic=False):
        feature_vectors = np.load(training_feature_path)
        categories = np.load(training_category_path)

        self.probabalistic = probabalistic

        # Calculate PCA features
        self.pca = PCA(n_components=6)
        pca_feature_vectors = self.pca.fit_transform(feature_vectors)

        # Fit SVM
        self.classifier = svm.SVC(
            class_weight='auto', probability=self.probabalistic)
        self.classifier.fit(pca_feature_vectors, categories)

    def predict(self, path):
        x_cells, y_cells = good_cells(path, 3, 3)

        avg_hists, bin_edges, avg_magnitudes, variances, flow = VideoFeatures(
            path).calc_window_features(x_cells, y_cells, None)
        fv = np.concatenate((avg_hists.flatten(), avg_magnitudes.flatten(),
                            variances.flatten()))

        prediction = self.classifier.predict(self.pca.transform(fv))[0]
        return prediction

    def read_labels(self, path):
        labels = []
        last_end_frame = 0
        with open(path, 'r') as p:
            for segment in p:
                category, end_frame = segment.strip().split(',')
                end_frame = int(end_frame)
                duration = end_frame-last_end_frame
                last_end_frame = end_frame
                labels.extend([category]*duration)
        return labels

    def draw_confmat(self, confmat, labels):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.matshow(confmat, cmap=cm.gray)
        plt.title('Confusion matrix')
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def roc_analysis(self, predictions, true_labels):
        # thresholds = np.arange(0,1,0.1)
        # predictions_at_thresh = []
        # for threshold in thresholds:
        #     for prediction, true_label in zip(predictions, true_labels):
        #         if prediction["running"] > threshold:
        #             predictions_at_thresh.append(1)
        #         else:
        #             predictions_at_thresh.append(0)
        #         if true_label == "running":
        #             predictions_at_thresh.append(1)
        #         else:
        #             predictions_at_thresh.append(0)

        #import pdb; pdb.set_trace()

        for pos, label in enumerate(self.classifier.classes_):
            binary_running_truths = [1 if l == label else 0 for l in true_labels]
            binary_running_predictions = [p[pos] for p in predictions]
            fpr, tpr, thresholds = roc_curve(binary_running_truths, binary_running_predictions)
            roc_auc = auc(fpr, tpr)
            print "Area under the ROC curve : %f" % (roc_auc,)

            # Plot ROC curve
            plt.clf()
            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic (%s)' % label)
            plt.legend(loc="lower right")
            plt.show()



    def realtime_predict(self, test_path, label_path, window_size, start, end,
                         output):
        labels = self.read_labels(label_path)
        x_cells, y_cells = good_cells(test_path, 3, 3)

        v = cv2.VideoWriter()
        dims = vid_dims(test_path)
        codec = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
        v.open(output, codec, 15, dims, 1)

        predictions = []
        true_labels = []
        for pos, agg_features in enumerate(VideoFeatures(
                test_path, persist=True).aggregate_features(
                x_cells, y_cells, window_size)):
            if window_size+pos < start:
                continue
            if window_size+pos > end:
                break

            avg_hists, bin_edges, avg_magnitudes, variances, flow = agg_features
            fv = np.concatenate((avg_hists.flatten(), avg_magnitudes.flatten(),
                                variances.flatten()))

            if self.probabalistic:
                prediction = self.classifier.predict_proba(
                        self.pca.transform(fv))[0]
            else:
                prediction = self.classifier.predict(self.pca.transform(fv))[0]
                vis = flow.show("Prediction", flow=False, text=prediction,
                                display=True)
                flow.show("Flow", flow=True, display=True)
                v.write(vis)

            # prediction is centered on current_time-window_size/2
            # and current_time=window_size+pos
            # => prediction is for (window_size/2)+pos
            predictions.append(prediction)
            true_labels.append(labels[(window_size/2)+pos])

        print

        if self.probabalistic:
            self.roc_analysis(predictions, true_labels)
        else:
            confmat_labels = sorted(set(predictions+true_labels))
            confmat = confusion_matrix(true_labels, predictions, confmat_labels)
            self.draw_confmat(confmat, confmat_labels)
            print "Accuracy: ", accuracy_score(true_labels, predictions)

        v.release()

if __name__ == "__main__":
    from sys import argv
    training_feature_path = argv[1]
    training_category_path = argv[2]
    test_path = argv[3]
    label_path = argv[4]
    window_size = int(argv[5])
    start = int(argv[6])
    end = int(argv[7])
    output = argv[8]
    dp = DissertationPredictor(training_feature_path, training_category_path,
                               probabalistic=True)
    #print dp.predict(test_path)
    dp.realtime_predict(test_path, label_path, window_size, start, end, output)
